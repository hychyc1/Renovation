import numpy as np
import pandas as pd
import numpy as np
import yaml
import os
import geopandas as gpd
from models.agent import PPOAgent
import torch
from env.env import RenovationEnv
# from env.env_3 import RenovationEnv
from utils.config import Config
import pandas as pd
import argparse

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

def save_grid(save_path, plan):
    grid_gdf = gpd.read_file('data/grid/raw_data_with_geometry.shp')
    
    original = env.get_state()

    for actions in plan:
        env.renovate(actions)
    finished = env.get_state()

    env.reset()
    for _ in range(cfg.max_year):
        env.signal_year_end()
    natural = env.get_state()

    finished["price_r"] *= (1 + cfg.inflation_rate) ** cfg.max_year
    finished["price_c"] *= (1 + cfg.inflation_rate) ** cfg.max_year

    natural["price_r"] *= (1 + cfg.inflation_rate) ** cfg.max_year
    natural["price_c"] *= (1 + cfg.inflation_rate) ** cfg.max_year

    for idx, row in grid_gdf.iterrows():
        i = row['row']  # row index in the matrix
        j = row['column']  # column index in the matrix
        
        # Assign the collected lists back to the GeoDataFrame
        grid_gdf.at[idx, 'POI_begin'] = original["POI"][i, j].item()
        grid_gdf.at[idx, 'POI_end'] = finished["POI"][i, j].item()

        grid_gdf.at[idx, 'Trans_begin'] = original["Trans"][i, j].item()
        grid_gdf.at[idx, 'Trans_end'] = finished["Trans"][i, j].item()

        grid_gdf.at[idx, 'Price_r_natural'] = natural["price_r"][i, j].item()
        grid_gdf.at[idx, 'Price_r_begin'] = original["price_r"][i, j].item()
        grid_gdf.at[idx, 'Price_r_end'] = finished["price_r"][i, j].item()
        grid_gdf.at[idx, 'Price_c_natural'] = natural["price_c"][i, j].item()
        grid_gdf.at[idx, 'Price_c_begin'] = original["price_c"][i, j].item()
        grid_gdf.at[idx, 'Price_c_end'] = finished["price_c"][i, j].item()

    if save_path is None:
        print(grid_gdf)
    else:
        import os
        os.makedirs(save_path, exist_ok=True)
        grid_gdf.to_file(save_path + 'grid_changes.shp')

def evaluate(save_path, save_name, nested_list):
    info_list = []
    env.reset()
    sum_reward = 0
    # print(len(nested_list))
    for actions in nested_list:
        # print(len(actions))
        _, reward, _, info = env.renovate(actions)
        info_list.append(info)
        sum_reward += reward
    infos = pd.DataFrame(info_list)
    # infos['weighted_R_M'] *= 5
    raw_sum_rewards = infos['weighted_R_M'].sum() + infos['weighted_R_P'].sum() + infos['weighted_R_T'].sum()
    # print(sum_rewards)
    if save_path is not None:
        infos.to_csv(save_path + 'report_' + save_name)
    return (infos['weighted_R_M'].sum(), infos['weighted_R_P'].sum(), infos['weighted_R_T'].sum(), sum_reward)

def parse_baseline(gdf):
    gdf = gdf.rename(columns={'批次': 'year', '容积率': 'FAR'})
    gdf[['r_c', 'r_r', 'r_poi']] = gdf['模式'].str.split(':', expand=True).astype(float) / 10
    gdf = gdf.drop(columns=['geometry', '模式'])
    # gdf[['r_c', 'r_r', 'r_poi']] = gdf['改造模'].str.split(':', expand=True).astype(float) / 10
    # gdf = gdf.drop(columns=['geometry', '改造模'])
    return gdf

def parse_csv(df):
    df = df.rename(columns={'顺序': 'year', '容积率': 'FAR'})
    # df[['x', 'y', 'z']] = df['改造plan'].astype(str).apply(lambda s: pd.Series(list(s)))
    df[['r_c', 'r_r', 'r_poi']] = df['改造plan'].astype(str).apply(lambda s: pd.Series([int(d)/10 for d in s]))
    df = df.drop(columns=['改造plan'])
    return df

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
    parser.add_argument(
        "--plan_name",
        type=str,
        default=None
    )
    parser.add_argument(
        "--chaoyang",
        type=bool,
        default = False
    )
    args = parser.parse_args()
    # config_path = 'cfg/cfg_' + args.name + ".yaml"
    if args.chaoyang:
        print("Eval Chaoyang")
        config_path = 'cfg/cfg_eval_cy.yaml'
        mask = torch.tensor(np.loadtxt('data/'+'Chaoyang'+'/mask.txt', delimiter=',', dtype=np.uint8))
        baseline_path = './baseline_csv/Chaoyang'
        # our_path = 'inferred_plan/朝阳区/plan_attn.csv'
        our_path = 'inferred_plan/normal_gnn_cy/plan.csv'
        # our_path = 'inferred_plan/朝阳区.csv'
        save_path = 'Eval_results/Chaoyang/'

    else:
        config_path  = 'cfg/cfg_eval.yaml'
        mask = None
        baseline_path = './baseline_csv'
        our_path = 'inferred_plan/plan_new.csv'
        save_path = 'Eval_results/Global/'
    # plan = 'inferred_plan/' + args.name + "/plan.csv"
    
    plan_name = args.plan_name

    cfg = Config.from_yaml(config_path)
    if args.name is not None:
        cfg.set_name(args.name)

    if args.chaoyang:
        cfg.village_per_year = 6

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
    # mask = torch.tensor(np.loadtxt('data/'+'朝阳区'+'/mask.txt', delimiter=',', dtype=np.uint8))
    env = RenovationEnv(cfg=cfg, device=device, grid_info=grid_info, village_array=villages.to_numpy(), extra_population=extra_population_array, mask=mask)

    # baseline_path = './baseline_csv/朝阳区'
    
    # for filename in os.listdir(baseline_path):
        # if filename[0] == '.':
            # continue

    results = []
    
    # for filename in ['plan1.csv', 'plan2.csv', 'plan3.csv', 'plan4.csv', 'plan5.csv', 'plan6.csv', 
    #                  'rule1.csv', 'rule2.csv', 'rule3.csv', 
    #                  'district.csv', 'district_own.csv', 
    #                  'greedy.csv', 'greedy2.csv', 'greedy_MC.csv', 'ga.csv']:
    for filename in ['ga.csv']:
        file_path = os.path.join(baseline_path, filename)
        # print(file_path)
        if not os.path.exists(file_path):
            continue
        plan = pd.read_csv(file_path)
        grouped = plan.groupby('year')
        # print(grouped)
        nested_list = [group[['ID', 'r_c', 'r_r', 'r_poi', 'FAR']].apply(tuple, axis=1).tolist() for _, group in grouped]

        r_m, r_p, r_t, _ = evaluate(save_path, filename, nested_list)
        r = r_m + r_p + r_t
        results.append((filename[:-4], r_m, r_p, r_t, r))
        print(f"{filename[:-4]}, {r_m: .2f}, {r_p: .2f}, {r_t: .2f}, {r: .2f}", flush=True)

    # file_path = 'inferred_plan/朝阳区.csv'
    # if True:
    # if False:
    plan = pd.read_csv(our_path)
    grouped = plan.groupby('year')
    # print(grouped)
    nested_list = [group[['ID', 'r_c', 'r_r', 'r_poi', 'FAR']].apply(tuple, axis=1).tolist() for _, group in grouped]
    # save_grid('Eval_results/Global', nested_list)
    r_m, r_p, r_t, r = evaluate(save_path, 'ours.csv', nested_list)
    r = r_m + r_p + r_t
    print(f"ours, {r_m: .2f}, {r_p: .2f}, {r_t: .2f}, {r: .2f}", flush=True)
    # print(("ours", f"{r_m: .2f}", f"{r_p: .2f}", f"{r_t: .2f}"), flush=True)
    results.append(('ours', r_m, r_p, r_t, r))

    results_df = pd.DataFrame(results, columns=['name', 'r_m', 'r_p', 'r_t', 'r'])
    results_df['r_m (raw)'] = results_df['r_m'] / cfg.monetary_weight
    results_df['r_p (raw)'] = results_df['r_p'] / cfg.POI_weight
    results_df['r_t (raw)'] = results_df['r_t'] / cfg.transportation_weight
    results_df.to_csv(save_path+'summary.csv')