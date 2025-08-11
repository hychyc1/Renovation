import random
import sys
from typing import Sequence, Tuple, List, Set

# ----------------------------- type aliases -----------------------------
VillageGene = Tuple[int, int, int]      # (village, comb_idx, far_idx)
YearPlan    = List[VillageGene]
Plan        = List[YearPlan]

VILLAGE_IDX = 0                         # helper

# ------------------------------------------------------------------------
class GreedyPlanner:
    """
    Random-sampling yearly greedy planner.

    For each year (≤ max_year):
      1. Uniformly sample **num_sample** candidate year-plans,
         each containing `village_per_year` distinct villages that
         have not been renovated yet, and with (comb_idx, far_idx)
         chosen uniformly at random.
      2. Evaluate the *whole* candidate plan using `env.compute_reward`.
      3. Keep the best-scoring candidate and commit it with `env.step`.
    """

    def __init__(
        self,
        env,                                # RenovationEnv
        n_villages: int,
        far_list: Sequence[float | int],
        comb_list: Sequence[Sequence[float | int] | dict],
        max_year: int = 12,
        village_per_year: int = 30,
        num_sample: int = 2_000,
        seed: int | None = None,
    ) -> None:
        self.env              = env
        self.n_villages       = n_villages
        self.far_list         = tuple(far_list)
        self.comb_list        = tuple(comb_list)
        self.max_year         = max_year
        self.village_per_year = village_per_year
        self.num_sample       = num_sample
        self.rng              = random.Random(seed)

    # ------------------------------------------------------------------
    def _random_year_plan(self, pool: Sequence[int]) -> YearPlan:
        """Return a random list of `village_per_year` genes."""
        villages = self.rng.sample(pool, self.village_per_year)
        plan: YearPlan = []
        for v in villages:
            comb_idx = self.rng.randrange(len(self.comb_list))
            far_idx  = self.rng.randrange(len(self.far_list))
            plan.append((v, comb_idx, far_idx))
        return plan

    # ------------------------------------------------------------------
    def run(self) -> Tuple[Plan, float]:
        plan: Plan          = []
        renovated: Set[int] = set()
        total_reward        = 0.0

        for year in range(self.max_year):
            remaining = [v for v in range(self.n_villages) if v not in renovated]
            if len(remaining) < self.village_per_year:
                break  # not enough villages for another full year plan

            best_plan: YearPlan | None = None
            best_reward = -float("inf")

            # --------------- Monte-Carlo search over candidate plans -----------
            for _ in range(self.num_sample):
                candidate = self._random_year_plan(remaining)
                r = self.env.compute_reward(candidate)
                if r > best_reward:
                    best_reward = r
                    best_plan   = candidate

            assert best_plan is not None, "Sampling produced no candidate!"

            # --------------- Commit the best candidate -------------------------
            _, realised_r, _, _ = self.env.step(best_plan)
            total_reward += realised_r
            renovated.update(g[VILLAGE_IDX] for g in best_plan)
            plan.append(best_plan)

            print(
                f"[Greedy] Year {year+1:2d}/{self.max_year} –"
                f" reward {realised_r:.4f} – cumulative {total_reward:.4f}",
                file=sys.stderr,
            )

        return plan, total_reward
