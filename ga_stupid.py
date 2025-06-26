import random
from typing import List, Tuple, Callable, Sequence
import argparse
import yaml
from utils.config import Config
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

class GeneticPlanner:
    """Baseline GA that evolves *max_year × village_per_year* renovation schedules."""

    def __init__(
        self,
        cfg: "Config",
        n_villages: int,
        far_list: Sequence[float | int],
        comb_list: Sequence[str | int],
        evaluate: Callable[["Plan"], float],
        generate_df,
        population_size: int = 100,
        generations: int = 400,
        tournament_k: int = 1,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.2,
        elite_frac: float = 0.05,
        random_seed: int | None = None,
        save_path: str = "ga",
    ) -> None:
        # --- schedule dimensions ------------------------------------------------
        self.max_year: int = cfg.max_year
        self.village_per_year: int = cfg.village_per_year

        # --- user‑supplied parameters ------------------------------------------
        self.n_villages = n_villages
        self.cfg = cfg
        self.far_list = far_list
        self.comb_list = comb_list
        self.evaluate_fn = evaluate
        self.generate_df = generate_df
        self.population_size = population_size
        self.generations = generations
        self.tournament_k = tournament_k
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_count = max(1, int(population_size * elite_frac))
        self.save_path = save_path
        if random_seed is not None:
            random.seed(random_seed)

        # --- internal state -----------------------------------------------------
        self.population: List["Plan"] = []
        self.fitness_cache: dict[int, float] = {}

    # ---------------------------------------------------------------------
    # High‑level GA loop
    # ---------------------------------------------------------------------
    def run(self) -> Tuple["Plan", float]:
        """Evolve and return ``(best_plan, best_score)``."""
        # ---------------- initial population --------------------------
        self.population = [self._random_plan() for _ in range(self.population_size)]

        # ---------------- evolutionary loop ---------------------------
        for gen in range(1, self.generations + 1):
            fitnesses = self._evaluate_population(self.population)

            # ---- logging: best score this generation -----------------
            best_gen_idx = max(range(len(self.population)), key=fitnesses.__getitem__)
            best_gen_score = fitnesses[best_gen_idx]
            best_gen_plan_df = self.generate_df(self.population[best_gen_idx])
            if gen % 50 == 0:
                best_gen_plan_df.to_csv(f"{self.save_path}/{gen}_{best_gen_score:.2f}.csv")
            print(f"Generation {gen}: best fitness = {best_gen_score:.4f}", flush=True)

            # ---- elitism --------------------------------------------
            elite_indices = sorted(
                range(len(self.population)),
                key=fitnesses.__getitem__,
                reverse=True,
            )[: self.elite_count]
            elites = [self.population[i] for i in elite_indices]

            # ---- offspring generation -------------------------------
            offspring: List["Plan"] = []
            while len(offspring) + len(elites) < self.population_size:
                parent1 = self._tournament_select()
                parent2 = self._tournament_select()

                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2

                child1 = self._maybe_mutate(child1)
                child2 = self._maybe_mutate(child2)

                offspring.extend([child1, child2])

            # clip offspring to fill remaining slots (in case of overflow)
            self.population = elites + offspring[: self.population_size - len(elites)]

        # ---------------- wrap‑up -------------------------------------
        final_fitnesses = self._evaluate_population(self.population)
        best_idx = max(range(len(self.population)), key=final_fitnesses.__getitem__)
        best_plan = self.population[best_idx]
        best_score = final_fitnesses[best_idx]
        return best_plan, best_score

    # ------------------------------------------------------------------
    # Population helpers
    # ------------------------------------------------------------------
    def _evaluate_population(self, pop: List["Plan"]) -> List[float]:
        """Memoized fitness evaluation."""
        scores: List[float] = []
        for individual in pop:
            key = id(individual)
            if key not in self.fitness_cache:
                self.fitness_cache[key] = self.evaluate_fn(individual)
            scores.append(self.fitness_cache[key])
        return scores

    def _tournament_select(self) -> "Plan":
        """Pick the best out of *k* random individuals."""
        contestants = random.sample(self.population, self.tournament_k)
        fitnesses = self._evaluate_population(contestants)
        winner_idx = max(range(len(contestants)), key=fitnesses.__getitem__)
        return contestants[winner_idx]

    # ------------------------------------------------------------------
    # Genetic operators
    # ------------------------------------------------------------------
    def _crossover(self, p1: "Plan", p2: "Plan") -> Tuple["Plan", "Plan"]:
        """Year‑wise 1‑point crossover preserving *village_per_year* structure."""
        point = random.randint(1, len(p1) - 1)
        child1 = [*p1[:point], *p2[point:]]
        child2 = [*p2[:point], *p1[point:]]
        return self._repair(child1), self._repair(child2)

    def _maybe_mutate(self, plan: "Plan") -> "Plan":
        if random.random() > self.mutation_rate:
            return plan
        mutation_type = random.choice(["swap", "replace", "param", "shuffle_years"])
        if mutation_type == "swap":
            return self._mutate_swap(plan)
        if mutation_type == "replace":
            return self._mutate_replace(plan)
        if mutation_type == "param":
            return self._mutate_param(plan)
        if mutation_type == "shuffle_years":
            return self._mutate_shuffle_years(plan)
        return plan  # pragma: no cover

    # ----------------------- mutation helpers ---------------------------
    def _mutate_swap(self, plan: "Plan") -> "Plan":
        """Swap two villages between two random years."""
        y1, y2 = random.sample(range(self.max_year), 2)
        i1, i2 = random.randrange(self.village_per_year), random.randrange(self.village_per_year)
        plan = [year.copy() for year in plan]
        plan[y1][i1], plan[y2][i2] = plan[y2][i2], plan[y1][i1]
        return self._repair(plan)

    def _mutate_replace(self, plan: "Plan") -> "Plan":
        """Replace one gene with a fresh, unused village."""
        used = {g[VILLAGE_IDX] for yr in plan for g in yr}
        unused = [v for v in range(self.n_villages) if v not in used]
        if not unused:
            return plan  # already using all villages
        y, i = random.randrange(self.max_year), random.randrange(self.village_per_year)
        new_gene = (
            random.choice(unused),
            random.randrange(len(self.comb_list)),
            random.randrange(len(self.far_list)),
        )
        plan = [yr.copy() for yr in plan]
        plan[y][i] = new_gene
        return plan

    def _mutate_param(self, plan: "Plan") -> "Plan":
        """Alter comb or FAR index of a single gene."""
        y, i = random.randrange(self.max_year), random.randrange(self.village_per_year)
        gene = list(plan[y][i])
        if random.random() < 0.5:
            gene[COMB_IDX] = random.randrange(len(self.comb_list))
        else:
            gene[FAR_IDX] = random.randrange(len(self.far_list))
        plan = [yr.copy() for yr in plan]
        plan[y][i] = tuple(gene)  # type: ignore[arg-type]
        return plan

    def _mutate_shuffle_years(self, plan: "Plan") -> "Plan":
        """Swap two whole years."""
        y1, y2 = random.sample(range(self.max_year), 2)
        plan = plan.copy()
        plan[y1], plan[y2] = plan[y2], plan[y1]
        return plan

    # ----------------------- repair & helpers ---------------------------
    def _repair(self, plan: "Plan") -> "Plan":
        """Ensure all village IDs are unique (hard constraint)."""
        seen: set[int] = set()
        all_villages = list(range(self.n_villages))
        random.shuffle(all_villages)

        plan_out: "Plan" = []
        pool_idx = 0
        for year in plan:
            new_year: "YearPlan" = []
            for gene in year:
                v = gene[VILLAGE_IDX]
                if v in seen:
                    # find next unused village
                    while pool_idx < len(all_villages) and all_villages[pool_idx] in seen:
                        pool_idx += 1
                    if pool_idx >= len(all_villages):
                        raise RuntimeError("Ran out of replacement villages during repair!")
                    v = all_villages[pool_idx]
                    pool_idx += 1
                seen.add(v)
                new_year.append((v, gene[COMB_IDX], gene[FAR_IDX]))
            # guarantee exactly self.village_per_year genes (defensive)
            new_year = new_year[: self.village_per_year]
            plan_out.append(new_year)
        return plan_out

    # ------------------------------------------------------------------
    # Plan initialisation
    # ------------------------------------------------------------------
    def _random_plan(self) -> "Plan":
        """Create a random plan with ``max_year × village_per_year`` unique villages."""
        villages = random.sample(
            range(self.n_villages), self.max_year * self.village_per_year
        )
        plan: "Plan" = []
        for y in range(self.max_year):
            year_genes: "YearPlan" = []
            for i in range(self.village_per_year):
                village_id = villages[y * self.village_per_year + i]
                comb_idx = random.randrange(len(self.comb_list))
                far_idx = random.randrange(len(self.far_list))
                year_genes.append((village_id, comb_idx, far_idx))
            plan.append(year_genes)
        return plan

    
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

    save_path = 'ga' if cfg.district is None else 'ga_' + cfg.district

    gp = GeneticPlanner(
        # env=env,
        n_villages=cfg.total_villages,
        cfg=cfg,
        far_list=cfg.FAR_values,
        comb_list=cfg.combinations,
        evaluate=evaluate,
        population_size=100,
        generations=2000,
        random_seed=42,
        generate_df=generate_df,
        save_path = save_path
    )

    best_plan, best_score = gp.run()
    print("\nBest score:", best_score)
    # print(best_plan)

    best_plan_df = generate_df(best_plan)
    # print(best_plan_df)
    best_plan_df.to_csv(f'{save_path}/all_best_{best_score:.4f}.csv')
