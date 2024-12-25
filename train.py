import numpy as np
import yaml
from models.agent import PPOAgent
import torch
from env.env import RenovationEnv
from Config.config import Config
import pandas as pd
from utils.parse_df import parse_df_to_env_state

# def parse_df_to_env_state(df):
#     """
#     Parses a GeoDataFrame with `row` and `column` entries into a dictionary
#     representing the environment's state.

#     Args:
#     - gdf (DataFrame): GeoDataFrame with `row` and `column` entries and relevant attributes.
#     - attribute_columns (list of str): List of columns in `gdf` representing the attributes to include.

#     Returns:
#     - dict: A dictionary where keys are attribute names and values are 2D numpy arrays (grids).
#     """
#     # Determine grid dimensions
#     max_row = df['row'].max() + 1
#     max_col = df['column'].max() + 1
#     attribute_columns = df.columns.drop(['row', 'column'])

#     # Initialize the dictionary to store grids for each attribute
#     env_state = {attr: np.zeros((max_row, max_col), dtype=np.float32) for attr in attribute_columns}

#     # Populate the grids
#     for _, row in df.iterrows():
#         r, c = row['row'].astype(int), row['column'].astype(int)
#         for attr in attribute_columns:
#             env_state[attr][r, c] = row[attr]

#     return env_state

# def train_one_iteration(agent: PPOAgent, iteration: int) -> None:
#     """Train one iteration"""
#     agent.optimize(iteration)
#     agent.save_checkpoint(iteration)

#     """clean up gpu memory"""
#     torch.cuda.empty_cache()


# def main_loop(_):

#     # setproctitle.setproctitle(f'road_planning_{FLAGS.cfg}_{FLAGS.global_seed}@suhy')


#     # cfg = Config(FLAGS.cfg, FLAGS.slum_name, FLAGS.global_seed, FLAGS.tmp, FLAGS.root_dir, FLAGS.agent)
#     cfg_addr = 'cfg/cfg1.yaml'
    
#     with open(cfg_addr, 'r') as stream:
#         cfg = yaml.safe_load(stream)
#     dtype = torch.float32
#     torch.set_default_dtype(dtype)
#     if torch.cuda.is_available():
#         device = torch.device('cuda')
#     else:
#         device = torch.device('cpu')

#     # checkpoint = int(FLAGS.iteration) if FLAGS.iteration.isnumeric() else FLAGS.iteration

#     """create agent"""
#     agent = PPOAgent(cfg=cfg, dtype=dtype, device=device)
    
#     for iteration in range(cfg.max_num_iterations):
#         train_one_iteration(agent, iteration)

#     agent.logger.info('training done!')


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

    # checkpoint_path = None
    checkpoint_path = 'checkpoint_iter_60_reward_ 9133.70.pt'

    """create agent"""
    # agent = PPOAgent(cfg=cfg, dtype=dtype, device=device)
    agent = PPOAgent(cfg=cfg, env=env)
    if checkpoint_path is not None:
        agent.load_checkpoint(checkpoint_path)

    agent.train(cfg.max_num_iterations)

    # for iteration in range(cfg.max_num_iterations):
    #     train_one_iteration(agent, iteration)

    agent.logger.info('training done!')
