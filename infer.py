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
import argparse

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

def process_trajectory_info(info_list, grid_gdf):
    """
    1) Reads 'POI_change' and 'Transportation_change' from each dictionary in info_list.
    2) Appends the values to 'grid_gdf' based on its (row, col) columns.
    3) Removes 'POI_change' and 'Transportation_change' from the dictionaries in info_list.
    
    Args:
        info_list (list of dict): Each element is a dictionary of info from a trajectory step/episode.
                                  Some dictionaries contain "POI_change" and "Transportation_change",
                                  which are (n x m) matrices.
        grid_gdf (GeoDataFrame): Must have columns "row" and "col" that indicate
                                 the grid cell index (i, j) for that row in the geodataframe.
    
    Returns:
        None. (Modifies grid_gdf in-place and cleans up info_list.)
    """

    # Iterate over each row in the GeoDataFrame
    for idx, row in grid_gdf.iterrows():
        i = row['row']  # row index in the matrix
        j = row['column']  # column index in the matrix
        
        # We'll build up lists of changes for this cell across all dictionaries
        poi_list = []
        trans_list = []

        for info_dict in info_list:
            # If 'POI_change' or 'Transportation_change' exist, read them
            poi_list.append(info_dict['POI_change'][i, j].item())

            trans_list.append(info_dict['Transportation_change'][i, j].item())

        # Assign the collected lists back to the GeoDataFrame
        grid_gdf.at[idx, 'poi_changes'] = " ".join(f"{x:.3f}" for x in poi_list)
        grid_gdf.at[idx, 'trans_changes'] = " ".join(f"{x:.3f}" for x in trans_list)

    for info_dict in info_list:
        if 'POI_change' in info_dict:
            del info_dict['POI_change']
        if 'Transportation_change' in info_dict:
            del info_dict['Transportation_change']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default='cfg/cfg1.yaml',
        help="Path to the config file."
    )
    args = parser.parse_args()
    cfg = Config.from_yaml(args.config)

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
    
    weight_path = 'checkpoint_iter_60_reward_14259.17.pt'
    agent.load_checkpoint(weight_path)
    plan, reward, info = agent.infer()

    grid_gdf = gpd.read_file('data/raw_data_with_geometry.shp')

    process_trajectory_info(info, grid_gdf)

    info = pd.DataFrame(info)
    save_path = 'inferred_plan/'
    result = plan_to_df(plan, cfg)
    urban_villages = gpd.read_file('data/urban_villages.shp')

    plan_gdf = generate_plan_gdf(result, urban_villages)

    if save_path is None:
        print(result)
        print(info)
    else:
        result.to_csv(save_path + 'plan.csv', index=False)
        plan_gdf.to_file(save_path + 'plan.shp')
        info.to_csv(save_path + 'report.csv', index_label='year')
        grid_gdf.to_file(save_path + 'grid_changes.shp')