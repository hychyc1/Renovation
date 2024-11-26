import numpy as np
import yaml
from models.agent import PPOAgent
import torch

def parse_df_to_env_state(df, attribute_columns):
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

    # Initialize the dictionary to store grids for each attribute
    env_state = {attr: np.zeros((max_row, max_col), dtype=np.float32) for attr in attribute_columns}

    # Populate the grids
    for _, row in df.iterrows():
        r, c = row['row'], row['column']
        for attr in attribute_columns:
            env_state[attr][r, c] = row[attr]

    return env_state

def train_one_iteration(agent: PPOAgent, iteration: int) -> None:
    """Train one iteration"""
    agent.optimize(iteration)
    agent.save_checkpoint(iteration)

    """clean up gpu memory"""
    torch.cuda.empty_cache()


def main_loop(_):

    # setproctitle.setproctitle(f'road_planning_{FLAGS.cfg}_{FLAGS.global_seed}@suhy')


    # cfg = Config(FLAGS.cfg, FLAGS.slum_name, FLAGS.global_seed, FLAGS.tmp, FLAGS.root_dir, FLAGS.agent)
    cfg_addr = 'cfg/cfg1.yaml'
    
    with open(cfg_addr, 'r') as stream:
        cfg = yaml.safe_load(stream)
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # checkpoint = int(FLAGS.iteration) if FLAGS.iteration.isnumeric() else FLAGS.iteration

    """create agent"""
    agent = PPOAgent(cfg=cfg, dtype=dtype, device=device)
    
    if FLAGS.infer:
        agent.infer(visualize=FLAGS.visualize)
    else:

        start_iteration = agent.start_iteration
        for iteration in range(start_iteration, cfg.max_num_iterations):
            train_one_iteration(agent, iteration)

    agent.logger.info('training done!')


if __name__ == '__main__':
    # flags.mark_flags_as_required([
    #   'cfg'
    # ])
    app.run(main_loop)
