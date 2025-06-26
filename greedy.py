import random
from typing import List, Tuple, Callable, Sequence
import argparse
import copy
import yaml
from tqdm import tqdm
from utils.config import Config
import sys
import torch
import pandas as pd
import numpy as np
import geopandas as gpd
from env.env import RenovationEnv

VillageGene = Tuple[int, int, int]  # (village_id, comb_idx, far_idx)
YearPlan = List[VillageGene]        # 30 genes = 1 year
Plan = List[YearPlan]               # 12 years

VILLAGE_IDX = 0
COMB_IDX = 1
FAR_IDX = 2


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

class GreedyPlanner:
    """One‑step greedy planner that maximises immediate reward.

    At each step (renovating exactly one village) we iterate over **all**
    `(village, comb_idx, far_idx)` combinations that have not yet been picked
    and choose the action that yields the highest *single‑step* reward when
    applied to a *deep‑copied* environment. The selected action is then
    committed to the real environment. We repeat until 360 villages are
    renovated, collecting them into the 12×30 plan structure.
    """

    def __init__(
        self,
        env: RenovationEnv,
        n_villages: int,
        far_list: Sequence[float | int],
        comb_list: Sequence[Sequence[float | int] | dict],
    ) -> None:
        self.env = env
        self.n_villages = n_villages
        self.far_list = far_list
        self.comb_list = comb_list

    # -------------------------------------------------------------
    def run(self) -> Tuple[Plan, float]:
        plan: Plan = []
        current_year: YearPlan = []
        total_reward = 0.0
        renovated: set[int] = set()

        # 360 single‑village steps --------------------------------
        for step in range(12 * 30):
            best_reward = -float("inf")
            best_action: VillageGene | None = None

            # Exhaustive search over remaining villages & params ---
            for village in range(self.n_villages):
                # if village % 100 == 0:
                #     print(f"Attempting village {village}", flush=True)
                if village in renovated:
                    continue
                for comb_idx in range(len(self.comb_list)):
                    for far_idx in range(len(self.far_list)):
                        # if best_action is not None and random.randint(1, 10) <= 3: 
                        #     continue
                        # trial_env = copy.deepcopy(env)
                        # _, reward, _, _ = trial_env.step([(village, comb_idx, far_idx)])
                        reward = env.compute_reward([(village, comb_idx, far_idx)])
                        # print((village, comb_idx, far_idx, reward), flush=True)
                        if reward > best_reward:
                            best_reward = reward
                            best_action = (village, comb_idx, far_idx)

            assert best_action is not None, "No valid action found!"

            # Commit the chosen action to the *real* environment ----
            _, reward, _, _ = self.env.step([best_action])
            total_reward += reward
            renovated.add(best_action[VILLAGE_IDX])
            current_year.append(best_action)

            # Roll year boundary -----------------------------------
            if len(current_year) == 30:
                plan.append(current_year)
                current_year = []

            print(f"[Greedy] Step {step+1:3d}/360 - reward {reward:.4f} - cumulative {total_reward:.4f}", file=sys.stderr)
            print(f"Action: {best_action}")

        if current_year:
            plan.append(current_year)
        return plan, total_reward
    
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

def setup_env():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default='cfg/cfg_normal_gnn_3.yaml',
        help="Path to the config file."
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

    return env, cfg, villages

if __name__ == "__main__":
    env, cfg, villages = setup_env()
    # print(global_cfg)

    def evaluate(plan: Plan) -> float:
        env.reset()
        sum_rewards = 0
        for actions in plan:
            # print(actions, flush=True)
            _, reward, _, _ = env.step(actions)
            sum_rewards += reward
        # print(sum_rewards, flush=True)
        return sum_rewards

    def generate_df(plan):
        return plan_to_df(plan, cfg, villages)

    gp = GreedyPlanner(
        env=env,
        n_villages=cfg.total_villages,
        far_list=cfg.FAR_values,
        comb_list=cfg.combinations
    )

    best_plan, best_score = gp.run()
    print("\nBest score:", best_score)
    # print(best_plan)

    best_plan_df = generate_df(best_plan)
    print(best_plan_df)
    best_plan_df.to_csv(f'greedy_{best_score: .4f}.csv')
