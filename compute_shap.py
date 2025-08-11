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
from math import factorial
import itertools


def evaluate(env, plan, manual_signal=True):
    if plan.empty:
        return pd.DataFrame(), (0, 0, 0, 0)
    
    env.reset()
    grouped = plan.groupby('year')
    info_list = []
    nested_list = [
        (year, group[['ID', 'r_c', 'r_r', 'r_poi', 'FAR']].apply(tuple, axis=1).tolist())
        for year, group in grouped
    ]
    
    curr_year = 1
    for year, actions in nested_list:
        while manual_signal and curr_year <= year:
            curr_year += 1
            env.signal_year_end()
        _, _, _, info = env.renovate(actions)
        info_list.append(info)
    
    infos = pd.DataFrame(info_list)
    R_M = infos['weighted_R_M'].sum()
    R_P = infos['weighted_R_P'].sum()
    R_T = infos['weighted_R_T'].sum()
    R_total = R_M + R_P + R_T
    
    return infos, (R_M, R_P, R_T, R_total)

def compute_shapley_exact(env, districts_and_plans) -> pd.DataFrame:
    n = len(districts_and_plans)
    district_names = [d for d, _ in districts_and_plans]
    plans_dict = dict(districts_and_plans)

    # 初始化每个 district 的 shapley 值字典
    shapley_values = {
        district: {'R_M': 0.0, 'R_P': 0.0, 'R_T': 0.0, 'R_total': 0.0}
        for district in district_names
    }

    all_districts = set(district_names)

    for district in district_names:
        print(district, flush=True)
        others = all_districts - {district}
        for r in range(n):  # 0 到 n-1 个其他元素的子集
            print(r, flush=True)
            for subset in itertools.combinations(others, r):
                subset_set = set(subset)

                # Subset without district
                subset_df = pd.concat([plans_dict[d] for d in subset_set], ignore_index=True) if subset_set else pd.DataFrame()
                _, v_S = evaluate(env, subset_df)

                # Subset with district
                subset_with_i_df = pd.concat([subset_df, plans_dict[district]], ignore_index=True)
                _, v_Si = evaluate(env, subset_with_i_df)

                # 边际贡献
                marginal = tuple(v_si - v_s for v_si, v_s in zip(v_Si, v_S))
                weight = factorial(len(subset)) * factorial(n - len(subset) - 1) / factorial(n)

                shapley_values[district]['R_M']     += weight * marginal[0]
                shapley_values[district]['R_P']     += weight * marginal[1]
                shapley_values[district]['R_T']     += weight * marginal[2]
                shapley_values[district]['R_total'] += weight * marginal[3]

    # 转换为 DataFrame 格式
    shapley_df = pd.DataFrame.from_dict(shapley_values, orient='index')
    shapley_df.index.name = 'District'
    return shapley_df

def estimate_shapley_random_permutations(env, districts_and_plans, n_trials: int = 10000) -> dict:
    n = len(districts_and_plans)
    contributions = {district: [] for district, _ in districts_and_plans}
    full_df = pd.concat([plan for _, plan in districts_and_plans], ignore_index=True)
    full_value = evaluate(env, full_df)[1]  # assuming evaluate returns (details, value)

    for i in range(n_trials):
        start_time = time.time()
        perm = random.sample(districts_and_plans, n)
        cumulative_df = pd.DataFrame()
        prev_value = 0

        for district, plan in perm[:-1]:
            cumulative_df = pd.concat([cumulative_df, plan], ignore_index=True)
            _, curr_value = evaluate(env, cumulative_df)
            marginal = curr_value - prev_value
            contributions[district].append(marginal)
            prev_value = curr_value

        # Last district's marginal contribution
        last_district, _ = perm[-1]
        contributions[last_district].append(full_value - prev_value)
        end_time = time.time()

        if i % 100 == 0:
            print(f"FINISH TRIAL {i}", flush=True)
        # print(f"a trial in {end_time-start_time} sec")

    return {district: np.mean(vals) for district, vals in contributions.items()}

def estimate_shapley_leave_one_out(env, districts_and_plans) -> dict:
    shapley_values = {}

    full_df = pd.concat([plan for _, plan in districts_and_plans], ignore_index=True)
    full_value = evaluate(env, full_df)[1]  # assuming evaluate returns (details, value)

    for district, _ in districts_and_plans:
        subset = [(d, p) for (d, p) in districts_and_plans if d != district]
        subset_df = pd.concat([plan for _, plan in subset], ignore_index=True)
        subset_value = evaluate(env, subset_df)[1]
        shapley_values[district] = full_value - subset_value

    return shapley_values

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        default='normal'
    )
    parser.add_argument(
        "--save_result",
        type=bool,
        default=True
    )
    args = parser.parse_args()
    # config_path = 'cfg/cfg_' + args.name + ".yaml"
    config_path = 'cfg/cfg_eval.yaml'
    plan = 'inferred_plan/' + args.name + "/plan.csv"
    
    cfg = Config.from_yaml(config_path)
    if args.name is not None:
        cfg.set_name(args.name)

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
    env = RenovationEnv(cfg=cfg, device=device, grid_info=grid_info, village_array=villages.to_numpy(), extra_population=extra_population_array)

    name = 'global'
    plan = pd.read_csv(f'plans/{name}.csv')

    districts = ['丰台区', '剩余五区', '大兴区', '房山区', '昌平区', '朝阳区', '海淀区', '通州区', '顺义区']

    districts_and_plans = [(district, restrict(district, plan)) for district in districts]

  
    # print("Leave one out")
    # print(estimate_shapley_leave_one_out(env, districts_and_plans), flush=True) 

    # print("Own Reward")
    # dist_reward = {}
    # for district in districts:
    #     mask = torch.tensor(np.loadtxt('data/'+district+'/mask.txt', delimiter=',', dtype=np.uint8))
    #     env_dist = RenovationEnv(cfg=cfg, device=device, grid_info=grid_info, village_array=villages.to_numpy(), extra_population=extra_population_array, mask=mask)
    #     _, dist_reward[district] = evaluate(env_dist, plan, manual_signal=False)
    # print(dist_reward)

    # print("Shap")
    Shap_results = compute_shapley_exact(env, districts_and_plans)
    Shap_results.to_csv(f'plans/{name}_shap.csv')