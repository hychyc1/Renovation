import pandas as pd
import numpy as np
import random
from typing import List, Callable
import numpy as np
import pandas as pd
import numpy as np
import yaml
import time
import geopandas as gpd
from models.agent import PPOAgent
import torch
from env.env import RenovationEnv
from utils.config import Config
import pandas as pd
import argparse

def evaluate(env: RenovationEnv, plan, manual_signal = True):
    env.reset()
    grouped = plan.groupby('year')
    info_list = []
    nested_list = [(year, group[['ID', 'r_c', 'r_r', 'r_poi', 'FAR']].apply(tuple, axis=1).tolist()) for year, group in grouped]
    # print(nested_list, flush=True)
    # return 
    curr_year = 1
    for year, actions in nested_list:
        while manual_signal and curr_year <= year:
            curr_year += 1
            env.signal_year_end()
        _, _, _, info = env.renovate(actions)
        info_list.append(info)
    infos = pd.DataFrame(info_list)
    sum_rewards = infos['weighted_R_M'].sum() + infos['weighted_R_P'].sum() + infos['weighted_R_T'].sum()
    return infos, (infos['weighted_R_M'].sum(), infos['weighted_R_P'].sum(), infos['weighted_R_T'].sum(), sum_rewards)


def parse_df_to_env_state(df, village_df):
    """
    Parses a GeoDataFrame with `row` and `column` entries into a dictionary
    representing the environment's state.

    Args:
    - gdf (DataFrame): GeoDataFrame with `row` and `column` entries and relevant attributes.
    - attribute_columns (list of str): List of columns in `gdf` representing the attributes to include.

    Returns:
    - dict: A dictionary where keys are attribute names and values are 2D numpy arrays (grids).
    """
    # Determine grid dimensions
    max_row = df['row'].max() + 1
    max_col = df['column'].max() + 1
    attribute_columns = df.columns.drop(['row', 'column', 'AREA'])

    # Initialize the dictionary to store grids for each attribute
    env_state = {attr: np.zeros((max_row, max_col), dtype=np.float32) for attr in attribute_columns}

    # Populate the grids
    for _, row in df.iterrows():
        r, c = row['row'].astype(int), row['column'].astype(int)
        for attr in attribute_columns:
            env_state[attr][r, c] = row[attr]

    env_state['AREA'] = np.zeros((max_row, max_col), dtype=np.float32)
    for _, row in village_df.iterrows():
        r, c = row['assign_row'].astype(int), row['assign_col'].astype(int)
        # r, c = int(row['assign_row']), int(row['assign_col'])
        env_state['AREA'][r, c] += row['area']
    return env_state

def restrict(district, plan):
    village_path = './data/' + district + "/villages.shp"
    village_in_district = gpd.read_file(village_path)
    plan_restricted = plan[plan['ID'].isin(village_in_district['ID'])]
    return plan_restricted


if __name__ == "__main__":
    config_path = 'cfg/cfg_eval.yaml'
    cfg = Config.from_yaml(config_path)
    list_to_test = [
        'district_own', 'district_global',
        'global', 
    ]
    for name in list_to_test:
        plan_path = 'plans/' + name + ".csv"
        
        dtype = torch.float32
        torch.set_default_dtype(dtype)
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        torch.set_default_device(device)

        # checkpoint = int(FLAGS.iteration) if FLAGS.iteration.isnumeric() else FLAGS.iteration

        villages = gpd.read_file('data/urban_villages.shp')
        villages = villages.dropna()
        # print(villages, flush=True)
        villages['area'] = villages.geometry.area
        villages = villages.drop(columns=['geometry', 'Area'])
        villages = villages.reindex(columns=['assign_row', 'assign_col', 'area', 'ID'])
        
        grid_info = pd.read_csv('data/updated_grid_info.csv')
        grid_info = parse_df_to_env_state(grid_info, villages)

        extra_population = pd.read_csv('data/whole_population.csv')
        extra_population = extra_population.reindex(columns=['row', 'column', 'population'])
        extra_population_array = extra_population.to_numpy()
        # print(villages)
        # print(extra_population_array)

        plan = pd.read_csv(plan_path)

        districts = ['Fengtai', 'Remaining', 'Daxing', 'Fangshan', 'Changping', 'Chaoyang', 'Haidian', 'Tongzhou', 'Shunyi']

        districts_and_plans = [(district, restrict(district, plan)) for district in districts]

        # print("Own Reward")
        results = []
        save_path = 'Eval_results/District_stats/' + name + '/'
        for district, dist_plan in districts_and_plans:
            # print(f"Running {district}")
            village_path = './data/' + district + "/villages.shp"
            num_village_in_district = len(gpd.read_file(village_path))
            cfg.village_per_year = (num_village_in_district + 25) // 50
            mask = torch.tensor(np.loadtxt('data/'+district+'/mask.txt', delimiter=',', dtype=np.uint8))

            env_dist = RenovationEnv(cfg=cfg, device=device, grid_info=grid_info, village_array=villages.to_numpy(), extra_population=extra_population_array, mask=mask)
            info, curr_result = evaluate(env_dist, dist_plan, manual_signal=True)
            results.append((district, *curr_result))
            info.to_csv(save_path + district + '.csv')
            print(results[-1])
        results_df = pd.DataFrame(results, columns = ['district', 'r_m', 'r_p', 'r_t', 'r'])
        results_df['r_m (raw)'] = results_df['r_m'] / cfg.monetary_weight
        results_df['r_p (raw)'] = results_df['r_p'] / cfg.POI_weight
        results_df['r_t (raw)'] = results_df['r_t'] / cfg.transportation_weight
        results_df.to_csv(save_path + 'summary.csv')
        results_df.to_excel(save_path + 'summary.xlsx')

    # print("Random perm")
    # print(estimate_shapley_random_permutations(env, districts_and_plans), flush=True) 