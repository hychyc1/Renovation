import numpy as np
import yaml
import os
from models.agent import PPOAgent
import torch
from env.env import RenovationEnv
from utils.config import Config
import pandas as pd
from utils.setup import setup_agent
import geopandas as gpd
import pandas as pd
import argparse

def plan_to_df(plan, cfg, villages):
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
        for (idx, comb, FAR_index) in renovations:
            # Extract r_c, r_r, r_poi from the combination index
            r_c, r_r, r_poi = cfg.combinations[comb]
            
            # Extract FAR value from the FAR index
            FAR = cfg.FAR_values[FAR_index]
            # Append the data to rows
            rows.append({
                'year': year + 1,  # Use 1-based indexing for years
                'ID': villages.iloc[idx]['ID'],
                'r_c': r_c,
                'r_r': r_r,
                'r_poi': r_poi,
                'FAR': FAR
            })
    
    # Convert rows to a DataFrame
    df = pd.DataFrame(rows, columns=['year', 'ID', 'r_c', 'r_r', 'r_poi', 'FAR'])
    return df


def generate_plan_gdf(plan, gdf: gpd.GeoDataFrame):
    """
    Generates a GeoDataFrame containing geometries from `gdf` whose assigned grid appears in the `plan`.

    Args:
    - plan (pd.DataFrame): A DataFrame containing the plan with columns:
                           ['year', 'idx', 'r_r', 'r_c', 'r_poi', 'FAR'].
    - gdf (gpd.GeoDataFrame): A GeoDataFrame with columns:
                              ['geometry']

    Returns:
    - gpd.GeoDataFrame: A GeoDataFrame with geometries from the plan, including columns:
                        ['geometry', 'year', 'r_r', 'r_c', 'r_poi', 'FAR'].
    """
    # Merge plan and gdf based on the grid coordinates
    merged = gdf.merge(plan, left_index=True, right_on='ID', how='right')

    # Select relevant columns
    output_columns = ['geometry', 'year', 'r_r', 'r_c', 'r_poi', 'FAR', 'ID']
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
            poi_list.append(info_dict['POI'][i, j].item())

            trans_list.append(info_dict['Trans'][i, j].item())

        # Assign the collected lists back to the GeoDataFrame
        grid_gdf.at[idx, 'POI'] = " ".join(f"{x:.3f}" for x in poi_list)
        grid_gdf.at[idx, 'Trans'] = " ".join(f"{x:.3f}" for x in trans_list)

    for info_dict in info_list:
        if 'POI' in info_dict:
            del info_dict['POI']
        if 'Trans' in info_dict:
            del info_dict['Trans']

if __name__ == '__main__':
    agent, cfg = setup_agent()
    
    plan, rewards, info = agent.infer(mean_action=False)
    
    # grid_gdf = gpd.read_file('data_use/geometry/raw_data_with_geometry.shp')
    # process_trajectory_info(info, grid_gdf)

    info = pd.DataFrame(info)
    save_path = 'inferred_plan/'
    # save_path = None
    if save_path is not None and cfg.name is not None:
        save_path += cfg.name + '/'

    village_path = 'data/urban_villages.shp' if cfg.district is None else 'data/' + cfg.district + '/villages.shp'
    villages = gpd.read_file(village_path)
    # villages = gpd.read_file('data/urban_villages.shp')
    villages = villages.dropna()
    # print(villages)
    # print(plan)
    result = plan_to_df(plan, cfg, villages)

    plan_gdf = generate_plan_gdf(result, villages)

    if save_path is None:
        print(plan)
        print(result)
        print(info)
        sum_rewards = info['weighted_R_M'].sum() + info['weighted_R_P'].sum() + info['weighted_R_T'].sum()
        print(sum_rewards)
        # print(plan_gdf)
    else:
        print(info)
        sum_rewards = info['weighted_R_M'].sum() + info['weighted_R_P'].sum() + info['weighted_R_T'].sum()
        print(sum_rewards)
        os.makedirs(save_path, exist_ok=True)
        result.to_csv(save_path + 'plan.csv', index=False)
        plan_gdf.to_file(save_path + 'plan.shp')
        info.to_csv(save_path + 'report.csv', index_label='year')
        # grid_gdf.to_file(save_path + 'grid_changes.shp')