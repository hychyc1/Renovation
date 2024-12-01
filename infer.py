import numpy as np
import yaml
from models.agent import PPOAgent
import torch
from env.env import RenovationEnv
from Config.config import Config
import pandas as pd
from utils.parse_df import parse_df_to_env_state

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
    
    weight_path = 'checkpoint_iter_80.pt'
    agent.load_checkpoint(weight_path)
    plan, reward = agent.infer()

    save_path = 'inferred_plan.csv'
    result = plan_to_df(plan, cfg)
    if save_path is None:
        print(result)
    else:
        result.to_csv(save_path, index=False)
    
