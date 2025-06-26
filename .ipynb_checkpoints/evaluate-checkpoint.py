import numpy as np
import pandas as pd
import numpy as np
import yaml
import geopandas as gpd
from models.agent import PPOAgent
import torch
from env.env import RenovationEnv
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

def save_grid(save_path):
    grid_gdf = gpd.read_file('data_use/geometry/raw_data_with_geometry.shp')
    
    original = env.get_state()

    for actions in nested_list:
        env.renovate(actions)
    finished = env.get_state()

    env.reset()
    for _ in range(cfg.max_step):
        env.renovate([])
    natural = env.get_state()

    finished["price_r"] *= (1 + cfg.inflation_rate) ** cfg.max_step
    finished["price_c"] *= (1 + cfg.inflation_rate) ** cfg.max_step

    natural["price_r"] *= (1 + cfg.inflation_rate) ** cfg.max_step
    natural["price_c"] *= (1 + cfg.inflation_rate) ** cfg.max_step

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


def save_result(save_path):
    info_list = []
    for actions in nested_list:
        _, _, _, info = env.renovate(actions)
        info_list.append(info)
    infos = pd.DataFrame(info_list)
    sum_rewards = infos['weighted_R_M'].sum() + infos['weighted_R_P'].sum() + infos['weighted_R_T'].sum()
    print(sum_rewards)
    if save_path is None:
        print(infos)
    else:
        import os
        os.makedirs(save_path, exist_ok=True)
        infos.to_csv(save_path + 'report.csv')

def parse_baseline(gdf):
    gdf = gdf.rename(columns={'批次': 'year', '容积率': 'FAR'})
    gdf[['r_c', 'r_r', 'r_poi']] = gdf['模式'].str.split(':', expand=True).astype(float) / 10
    gdf = gdf.drop(columns=['geometry', '模式'])
    # gdf[['r_c', 'r_r', 'r_poi']] = gdf['改造模'].str.split(':', expand=True).astype(float) / 10
    # gdf = gdf.drop(columns=['geometry', '改造模'])
    return gdf

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
    config_path = 'cfg/cfg_' + args.name + ".yaml"
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

    # print(villages)
    env = RenovationEnv(cfg=cfg, device=device, grid_info=grid_info, village_array=villages.to_numpy())

    # plan = gpd.read_file('baseline/方案四/360方案村庄统一格式.shp')
    # plan = gpd.read_file('baseline/方案五/Export_Output_4.shp')
    plan = gpd.read_file('baseline/方案六/ghy改格式.shp')
    print(plan)
    # plan = gpd.read_file('baseline/方案三/改造村.shp')
    # print(plan[plan['批次']==4], flush=True)
    plan = parse_baseline(plan)
    # print(plan)
    # plan = gpd.read_csv(plan)

    grouped = plan.groupby('year')
    nested_list = [group[['ID', 'r_c', 'r_r', 'r_poi', 'FAR']].apply(tuple, axis=1).tolist() for _, group in grouped]
    print(nested_list[3])

    save_result(None)
    
    # save_path = 'inferred_plan/'
    # if cfg.name is not None:
    #     save_path += cfg.name + '/'

    # if args.save_result:
    #     save_result(save_path)
