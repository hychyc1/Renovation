import numpy as np
import pandas as pd
import geopandas as gpd
import numpy as np
import yaml
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

def setup_agent():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default='cfg/cfg_normal_gnn.yaml',
        help="Path to the config file."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint file."
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None
    )
    parser.add_argument(
        "--district",
        type=str,
        default=None
    )
    args = parser.parse_args()
    cfg = Config.from_yaml(args.config)
    if args.name is not None:
        cfg.set_name(args.name)
    print(f"Loaded config: {cfg.name}", flush=True)

    dtype = torch.float32
    torch.set_default_dtype(dtype)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    torch.set_default_device(device)

    # checkpoint = int(FLAGS.iteration) if FLAGS.iteration.isnumeric() else FLAGS.iteration

    # data_path = 
    cfg.district = args.district
    village_path = 'data/urban_villages.shp' if args.district is None else 'data/' + args.district + '/villages.shp'
    villages = gpd.read_file(village_path)
    villages = villages.dropna()
    # print(villages.sort_values(by='ID'))
    cfg.total_villages = len(villages)
    if args.district is not None:
        cfg.village_per_year = (cfg.total_villages + 25) // 50
        if cfg.village_per_step > 1:
            cfg.village_per_step = (cfg.total_villages + 25) // 50
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
    env = RenovationEnv(cfg=cfg, device=device, grid_info=grid_info, village_array=villages.to_numpy(), extra_population=extra_population_array)

    checkpoint_path = args.checkpoint
    # agent = PPOAgent(cfg=cfg, device=device, env=env)
    # if cfg.use_parallel:
    #     agent = PPOAgentParallel(cfg=cfg, device=device, env=env)
    # else:
    agent = PPOAgent(cfg=cfg, device=device, env=env)
    if checkpoint_path is not None:
        agent.load_checkpoint(checkpoint_path)

    return agent, cfg
