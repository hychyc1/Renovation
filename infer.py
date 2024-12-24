import numpy as np
import yaml
from models.agent import PPOAgent
import torch
from env.env import RenovationEnv
from Config.config import Config
import pandas as pd
from utils.parse_df import parse_df_to_env_state
import geopandas as gpd
import pandas as pd

def plan_to_df(plan, cfg):
    """
    Converts a renovation plan into a DataFrame.

    Args:
    - plan (list of list): The plan, where the i-th list contains the places to be renovated in the i-th year.
                           Each entry in the inner list is a tuple (row, column, comb, FAR_index).
    - cfg (object): Configuration object containing combinations and FAR values:
        - cfg.combinations: List of tuples [(r_c, r_r, r_poi), ...]
        - cfg.FAR_values: List of FAR values [1.8, 2.0, ...]

    Returns:
    - pd.DataFrame: DataFrame with columns ['year', 'row', 'column', 'r_c', 'r_r', 'r_poi', 'FAR'].
    """
    rows = []
    
    for year, renovations in enumerate(plan):
        for (row, column, comb, FAR_index) in renovations:
            # Extract r_c, r_r, r_poi from the combination index
            r_c, r_r, r_poi = cfg.combinations[comb]
            
            # Extract FAR value from the FAR index
            FAR = cfg.FAR_values[FAR_index]
            
            # Append the data to rows
            rows.append({
                'year': year + 1,  # Use 1-based indexing for years
                'row': row,
                'column': column,
                'r_c': r_c,
                'r_r': r_r,
                'r_poi': r_poi,
                'FAR': FAR
            })
    
    # Convert rows to a DataFrame
    df = pd.DataFrame(rows, columns=['year', 'row', 'column', 'r_c', 'r_r', 'r_poi', 'FAR'])
    return df


def generate_plan_gdf(plan, gdf):
    """
    Generates a GeoDataFrame containing geometries from `gdf` whose assigned grid appears in the `plan`.

    Args:
    - plan (pd.DataFrame): A DataFrame containing the plan with columns:
                           ['year', 'row', 'column', 'r_r', 'r_c', 'r_poi', 'FAR'].
    - gdf (gpd.GeoDataFrame): A GeoDataFrame with columns:
                              ['geometry', 'assigned_grid'], where 'assigned_grid' is a tuple (row, column).

    Returns:
    - gpd.GeoDataFrame: A GeoDataFrame with geometries from the plan, including columns:
                        ['geometry', 'year', 'row', 'column', 'r_r', 'r_c', 'r_poi', 'FAR'].
    """
    # Merge plan and gdf based on the grid coordinates
    plan['assigned_grid'] = list(zip(plan['row'], plan['column']))
    gdf['assigned_grid'] = list(zip(gdf['assign_row'], gdf['assign_col']))
    merged = gdf.merge(plan, on='assigned_grid', how='inner')

    # Select relevant columns
    output_columns = ['geometry', 'year', 'row', 'column', 'r_r', 'r_c', 'r_poi', 'FAR']
    result_gdf = merged[output_columns]

    return gpd.GeoDataFrame(result_gdf, geometry='geometry')

if __name__ == '__main__':
    # flags.mark_flags_as_required([
    #   'cfg'
    # ])
    cfg_path = 'cfg/cfg1.yaml'
    cfg = Config.from_yaml(cfg_path)

    dtype = torch.float32
    torch.set_default_dtype(dtype)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # checkpoint = int(FLAGS.iteration) if FLAGS.iteration.isnumeric() else FLAGS.iteration

    data_path = 'data/updated_grid_info.csv'
    grid_info = pd.read_csv(data_path)
    grid_info = parse_df_to_env_state(grid_info)

    env = RenovationEnv(cfg=cfg, grid_info=grid_info)

    """create agent"""
    # agent = PPOAgent(cfg=cfg, dtype=dtype, device=device)
    agent = PPOAgent(cfg=cfg, env=env)
    
    weight_path = 'checkpoint_iter_100_reward_ 2383.68.pt'
    agent.load_checkpoint(weight_path)
    plan, reward, info = agent.infer()

    # print(info)
    info = pd.DataFrame(info)
    save_path = 'inferred_plan/'
    result = plan_to_df(plan, cfg)
    urban_villages = gpd.read_file('data/urban_villages.shp')

    plan_gdf = generate_plan_gdf(result, urban_villages)

    if save_path is None:
        print(result)
        print(info)
    else:
        result.to_csv(save_path + 'plan.csv')
        plan_gdf.to_csv(save_path + 'plan.shp', index=False)
        info.to_csv(save_path + 'report.csv', index_label='year')