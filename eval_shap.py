import pandas as pd
import numpy as np
import random
from typing import List, Callable

def load_plan(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def evaluate()

def estimate_shapley_random_permutations(plan_paths: List[str], eval_fn: Callable[[pd.DataFrame], float], n_trials: int = 100000) -> dict:
    n = len(plan_paths)
    contributions = {path: [] for path in plan_paths}
    plans = {path: load_plan(path) for path in plan_paths}

    for _ in range(n_trials):
        perm = random.sample(plan_paths, n)
        cumulative_df = pd.DataFrame()
        prev_value = 0

        for i, path in enumerate(perm):
            cumulative_df = pd.concat([cumulative_df, plans[path]], ignore_index=True)
            curr_value = evaluate(cumulative_df)
            marginal = curr_value - prev_value
            contributions[path].append(marginal)
            prev_value = curr_value

    return {path: np.mean(vals) for path, vals in contributions.items()}

def estimate_shapley_leave_one_out(plan_paths: List[str], eval_fn: Callable[[pd.DataFrame], float]) -> dict:
    full_df = pd.concat([load_plan(path) for path in plan_paths], ignore_index=True)
    full_value = eval_fn(full_df)
    shapley_values = {}

    for path in plan_paths:
        subset_paths = [p for p in plan_paths if p != path]
        subset_df = pd.concat([load_plan(p) for p in subset_paths], ignore_index=True)
        subset_value = eval_fn(subset_df)
        shapley_values[path] = full_value - subset_value

    return shapley_values
