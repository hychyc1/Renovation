import numpy as np
from utils.transportation import Transportation
from utils.config import Config
import torch
import torch.nn.functional as F
from scipy.ndimage import convolve

class RenovationEnv:
    def __init__(self, cfg: Config, grid_info, village_array, extra_population, device):
        """
        Initializes the environment.

        Args:
        - cfg (object): Configuration object containing:
            - cfg.plot_ratio (float): Plot ratio for renovation costs.
            - cfg.POI_plot_ratio (float): POI plot ratio for renovation costs.
            - cfg.monetary_compensation_ratio (float): Compensation ratio for renovation costs.
            - cfg.speed_c (float): Annual increase rate for commercial prices.
            - cfg.speed_r (float): Annual increase rate for rental prices.
            - cfg.max_year (int): Maximum number of years before termination.
            - cfg.POI_affect_range (int): Range within which POI affects property prices.
            - cfg.inflation_rate (float): Annual inflation rate.
            - cfg.space_per_person (float): Space required per person.
            - cfg.occupation_rate (float): Occupation rate for rental properties.
            - cfg.combinations (list of tuples): List of combinations as (r_c, r_r, r_poi).
            - cfg.FAR_values (list of float): List of FAR values.
            - cfg.reward_specs.monetary_weight (float): Weight applied to monetary rewards.
            - cfg.reward_specs.transportation_weight (float): Weight applied to transportation rewards.
            - cfg.reward_specs.POI_weight (float): Weight applied to POI rewards.

        - grid_info (dict): Grid attributes with keys:
            - "pop" (np.ndarray): Population distribution on the grid.
            - "price_c" (np.ndarray): Commercial property prices on the grid.
            - "price_r" (np.ndarray): Rental property prices on the grid.
            - "POI" (np.ndarray): Points of Interest (POI) values on the grid.
            - "AREA" (np.ndarray): Available area for renovation on the grid.
            - "r_b" (np.ndarray): Baseline adjustment factor for the grid.

        - village_list (list): Each element is (i, j, area)
        """
        self.device = device
        self.n, self.m = next(iter(grid_info.values())).shape
        self.Transportation = Transportation(self.n, self.m, grid_info['pop'], extra_population)
        self.original_villages = village_array.copy()
        self.current_villages = self.original_villages.copy()
        self.idx_to_position = {int(idx): int(number) for number, (_, _, _, idx) in enumerate(village_array)}
        # print(self.current_villages, flush=True)
        # print(self.idx_to_position, flush=True)
        self.original_state = {key: torch.tensor(value, device=device).clone() for key, value in grid_info.items()}
        self.current_state = {key: torch.tensor(value, device=device).clone() for key, value in grid_info.items()}

        # Global parameters
        self.plot_ratio = cfg.plot_ratio
        self.POI_plot_ratio = cfg.POI_plot_ratio
        self.monetary_compensation_ratio = cfg.monetary_compensation_ratio
        self.construction_cost_ratio = cfg.construction_cost_ratio
        self.speed_c = cfg.speed_c
        self.speed_r = cfg.speed_r
        self.price_changes = cfg.price_changes
        self.max_year = cfg.max_year
        self.village_per_year = cfg.village_per_year
        self.POI_affect_range = cfg.POI_affect_range
        self.inflation_rate = cfg.inflation_rate
        self.space_per_person = cfg.space_per_person
        self.POI_per_space = cfg.POI_per_space
        self.occupation_rate = cfg.occupation_rate
        self.combinations = cfg.combinations
        self.FAR_values = cfg.FAR_values
        self.balance_alpha = cfg.balance_alpha
        self.balance_func = cfg.balance_func
        self.balance_range = cfg.balance_upper
        
        # areas = np.sort([village[2] for village in self.current_villages])[::-1]
        # max_area = areas[:cfg.village_per_year * cfg.max_year].sum()
        # self.recommended_per_year = max_area / cfg.max_year * cfg.balance_upper
        avg_area = np.average([village[2] for village in self.current_villages])
        self.recommended_per_year = avg_area * self.village_per_year

        self.repetitive_penalty = cfg.repetitive_penalty

        # Reward weights
        self.monetary_weight = cfg.monetary_weight
        self.transportation_weight = cfg.transportation_weight
        self.POI_weight = cfg.POI_weight

        # Counter
        self.current_year = 0
        self.num_villages_this_year = 0
        self.area_this_year = 0

        self.scaling = {key: torch.max(self.current_state[key]) for key in self.current_state.keys()}

    def reset(self):
        """
        Resets the environment to its initial state.

        Returns:
        - state (dict, list): The initial state of the grid, including all attributes.
        """
        self.current_state = {key: value.clone() for key, value in self.original_state.items()}
        self.current_villages = self.original_villages.copy()
        self.Transportation.reset()
        self.current_year = 0
        self.num_villages_this_year = 0
        self.area_this_year = 0
        return (self.scale(self.current_state), self.current_villages, 0, 0)


    def get_avg_poi(self, pop, poi):
        R = self.POI_affect_range
        K = 2*R + 1
        device = poi.device

        idx = torch.arange(-R, R+1, device=device)
        dy = idx.view(-1, 1).abs()
        dx = idx.view(1, -1).abs()
        mask = (dy + dx <= R).float()        
        kernel = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]
        poi_in = poi.unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
        poi_sum = F.conv2d(poi_in, kernel, padding=R)  # [1,1,H,W]
        poi_sum = poi_sum.squeeze(0).squeeze(0)         # back to [H,W]

        # 4) compute population-weighted average
        total_pop       = pop.sum()
        weighted_poi_tot = (poi_sum * pop).sum()
        avg_poi_per_person = weighted_poi_tot / total_pop
        return avg_poi_per_person

    # -------------------------------------------------------------------------
    # Read-only reward probe (no persistent side-effects)
    # -------------------------------------------------------------------------
    def compute_reward(self, actions):
        """
        Compute the one-step reward for *actions* without changing the
        environment state.  It mimics `step()` but rolls back every update
        (grid tensors, Transportation object, bookkeeping counters).

        Parameters
        ----------
        actions : list[tuple]
            Each tuple is (village_idx, comb_idx, far_idx).

        Returns
        -------
        float
            Immediate reward identical to what `step()` would have returned.
        dict
            The same `info` dictionary (values are detached from the env).
        """
        # ------------------------------------------------------------------
        # Local mirrors of counters – never write to the real ones
        # ------------------------------------------------------------------
        area_so_far   = self.area_this_year
        villages_done = self.num_villages_this_year

        # Pre-step snapshots for ratio calculations
        grid          = self.current_state

        # Transportation baseline
        _, trans_avg0 = self.Transportation.calc_transport_time()
        avg_poi_old = self.get_avg_poi(poi=grid['POI'],       pop=grid['pop'])
        
        # Lists for rolling-back state
        pop_deltas    = []               # (i, j, +Δpop)
        poi_deltas    = []               # (i, j, +Δpoi)
        area_deltas   = []               # (i, j, –Δarea)
        move_log      = []               # record movein for moveout
        villages_reset = []              # (v_idx, original_AREA)

        # ------------------ reward accumulators ---------------------------
        R_M = 0.0
        repetitive_penalty = 0.0

        # ------------------------------------------------------------------
        # Apply the actions *temporarily*
        # ------------------------------------------------------------------
        for v_idx, comb_idx, far_idx in actions:
            i, j = self.current_villages[v_idx][0].astype(int), self.current_villages[v_idx][1].astype(int)
            AREA   = float(self.current_villages[v_idx][2])

            if AREA == 0:
                repetitive_penalty += float(self.repetitive_penalty)
                continue

            r_c, r_r, r_poi = self.combinations[comb_idx]
            FAR             = self.FAR_values[far_idx]
            r_b             = grid["r_b"][i, j]

            # Log original village area so we can restore it
            villages_reset.append((v_idx, AREA))
            self.current_villages[v_idx][2] = 0.0            # mark as renovated (temp)

            # Space calculations
            sell_space = AREA * FAR * r_c * r_b
            rent_space = AREA * FAR * r_r * r_b
            poi_space  = AREA * r_poi * r_b

            # ---------------- monetary reward -----------------------------
            sell_reward = sell_space * grid["price_c"][i, j]
            rent_reward = rent_space * grid["price_r"][i, j] * 12 * self.max_year
            cost = (
                AREA * self.plot_ratio * (0.15 + self.monetary_compensation_ratio) * grid["price_c"][i, j]
                + (sell_space + rent_space) * self.construction_cost_ratio * grid["price_c"][i, j]
                + poi_space * self.POI_plot_ratio * grid["price_c"][i, j]
            )
            R_M += sell_reward + rent_reward - cost

            # ---------------- grid updates (temp) -------------------------
            #   • pop   += Δ
            #   • POI   += Δ
            #   • AREA  -= AREA
            move_in_people = (
                sell_space / self.space_per_person
                + self.occupation_rate * rent_space / self.space_per_person
            )

            grid["pop"][i, j]  += move_in_people
            pop_deltas.append((i, j, move_in_people))

            grid["POI"][i, j]  += poi_space * self.POI_per_space
            poi_deltas.append((i, j, poi_space * self.POI_per_space))

            grid["AREA"][i, j] -= AREA
            area_deltas.append((i, j, AREA))                 # remember to add back

            # Transportation side-effect (temp)
            self.Transportation.movein(i, j, move_in_people)
            move_log.append((i, j, move_in_people))

            # Yearly accounting (local copy only)
            area_so_far   += AREA
            villages_done += 1

        # ------------------------------------------------------------------
        # Derived rewards that require the mutated state
        # ------------------------------------------------------------------
        _, trans_avg1 = self.Transportation.calc_transport_time()
        R_T        = trans_avg0 - trans_avg1

        avg_poi_new = self.get_avg_poi(poi=grid["POI"], pop=grid["pop"])
        R_P        = avg_poi_new - avg_poi_old

        # Year-end balance penalty (local)
        cost_balance = 0
        if villages_done >= self.village_per_year:
            cost_balance = self.compute_cost_balance(area_so_far)

        # ------------------------------------------------------------------
        # Aggregate weighted reward
        # ------------------------------------------------------------------
        weighted_R_M = self.monetary_weight       * R_M
        weighted_R_T = self.transportation_weight * R_T
        weighted_R_P = self.POI_weight            * R_P

        total_reward = (
            weighted_R_M + weighted_R_T + weighted_R_P
            - cost_balance
            - repetitive_penalty
        )

        # ------------------------------------------------------------------
        # ROLL BACK – restore pristine env -------------------------------
        # ------------------------------------------------------------------
        # 1. grids
        for i, j, delta in pop_deltas:
            grid["pop"][i, j] -= delta
        for i, j, delta in poi_deltas:
            grid["POI"][i, j] -= delta
        for i, j, area in area_deltas:
            grid["AREA"][i, j] += area

        # 2. Transportation
        for i, j, delta_people in move_log:
            self.Transportation.moveout(i, j, delta_people)

        # 3. Village availability
        for v_idx, original_area in villages_reset:
            self.current_villages[v_idx][2] = original_area

        # (self.area_this_year, num_villages_this_year, current_year etc.
        # were never mutated, so nothing else to reset.)

        return float(total_reward)


    def step(self, actions):
        """
        Executes a step in the environment based on the given actions.

        Args:
        - actions (list of tuples): Each tuple represents a renovation action as (x, r_c, r_r, r_poi, FAR),

        Returns:
        - next_state (dict, list): The updated grid attributes after the actions.
        - reward (float): The total reward from this step.
        - done (bool): Whether the episode has ended.
        - info (dict): Additional information (e.g., breakdown of rewards).
        """
        grid = self.current_state
        # old_population = grid["pop"].clone()
        old_POI = grid["POI"].clone()

        # Calculate rewards
        R_M = 0  # Monetary reward
        repetitive_penalty = 0

        old_transportation, old_avg = self.Transportation.calc_transport_time()
        avg_POI_old = self.get_avg_poi(poi=grid['POI'], pop=grid['pop'])

        for x, comb, f in actions:
            # Renovation parameters
            i, j = self.current_villages[x][0].astype(int), self.current_villages[x][1].astype(int)
            AREA = self.current_villages[x][2]
            if AREA == 0:
                repetitive_penalty += self.repetitive_penalty
            r_c, r_r, r_poi = self.combinations[comb]
            FAR = self.FAR_values[f]

            r_b = grid["r_b"][i, j]

            self.area_this_year += AREA
            self.current_villages[x][2] = 0

            # Sell and rent areas
            sell_space = AREA * FAR * r_c * r_b
            rent_space = AREA * FAR * r_r * r_b
            POI_space = AREA * r_poi * r_b

            # Update grid attributes
            grid["AREA"][i, j] -= AREA  # Reduce renovatable area in this grid
            move_in_people = sell_space / self.space_per_person + self.occupation_rate * rent_space / self.space_per_person
            grid["pop"][i, j] += move_in_people
            self.Transportation.movein(i, j, move_in_people)
            grid["POI"][i, j] += POI_space * self.POI_per_space

            # Monetary reward components
            sell_reward = sell_space * grid["price_c"][i, j]
            rent_reward = rent_space * grid["price_r"][i, j] * 12 * self.max_year # Annual rent revenue
            cost = (
                AREA * self.plot_ratio * (0.15 + self.monetary_compensation_ratio) * grid["price_c"][i, j]
                + (sell_space + rent_space) * self.construction_cost_ratio * grid["price_c"][i, j]
                + POI_space * self.POI_plot_ratio * grid["price_c"][i, j]
            )
            R_M += sell_reward + rent_reward - cost
            # if sell_reward + rent_reward - cost < -1:
                # print(f"INFO {AREA}, {FAR}, {r_c}, {r_r}, {r_poi}, {sell_space}, {rent_space}, {POI_space}, {sell_reward}, {rent_reward}, {cost}, {R_M}")

            # Adjust adjacent grids' prices

            # print(f"INFO {i}, {j}, {AREA}, {FAR}, {r_c}, {r_r}, {r_poi}, {sell_space}, {rent_space}, {POI_space}, {sell_reward}, {rent_reward}, {cost}, {R_M}")

            # self.update_adjacent_prices(i, j, grid, prev_POI)

        self.update_prices(grid, old_POI)

        cost_balance = 0
        area_so_far = self.area_this_year
        self.num_villages_this_year += len(actions)
        if self.num_villages_this_year >= self.village_per_year:
            self.num_villages_this_year = 0
            self.current_year += 1
            cost_balance = self.compute_cost_balance(self.area_this_year)
            self.area_this_year = 0

            # Global changes
            if self.price_changes is not None:
                grid["price_c"] *= 1 + self.price_changes[self.current_year-1]
                grid["price_r"] *= 1 + self.price_changes[self.current_year-1]
            else:
                grid["price_c"] *= 1 + self.speed_c
                grid["price_r"] *= 1 + self.speed_r
            grid["price_c"] /= 1 + self.inflation_rate
            grid["price_r"] /= 1 + self.inflation_rate

        # Transportation reward
        new_transportation, new_avg = self.Transportation.calc_transport_time()
        R_T = old_avg - new_avg
        R_T_ratio = R_T / old_avg

        # POI reward
        avg_POI_new = self.get_avg_poi(poi=grid['POI'], pop=grid['pop'])
        R_P = avg_POI_new - avg_POI_old
        R_P_ratio = R_P / avg_POI_new

        # print(f'Popu_old: {old_population.sum()}, POI_old: {old_POI.sum()}, Popu_new: {grid["pop"].sum()}, POI_new: {grid["POI"].sum()}, R_P: {R_P}')
        # Apply reward weights
        weighted_R_M = self.monetary_weight * R_M
        weighted_R_T = self.transportation_weight * R_T
        weighted_R_P = self.POI_weight * R_P

        # Total reward
        total_reward = weighted_R_M + weighted_R_T + weighted_R_P - cost_balance - repetitive_penalty
        # print(f"{(weighted_R_M, weighted_R_T, weighted_R_P, cost_balance, repetitive_penalty)}", flush=True)

        # Check if the episode is done
        done = self.current_year >= self.max_year

        # old_transportation[old_transportation == 0] = float('inf')
        # old_POI[old_POI < 1.0e-7] = float('inf')

        # Return next state, reward, done, and info

        info = {
            "AREA": area_so_far,
            "weighted_R_M": weighted_R_M.item(),
            "weighted_R_T": weighted_R_T.item(),
            "weighted_R_P": weighted_R_P.item(),
            "cost_balance": cost_balance,
            "R_T_ratio": R_T_ratio.item(),
            "R_P_ratio": R_P_ratio.item()
            # "repetitive_penalty": repetitive_penalty
            # "POI_change": grid["POI"] - old_POI,
            # "Transportation_change": R_T_2d / old_transportation
        }
        # if grid_info:
        #     info.update(info_grid)
        return (self.scale(grid), self.current_villages, self.current_year / self.max_year, self.area_this_year / self.recommended_per_year), total_reward.item(), done, info

    def scale(self, grid):
        scaled_grid = {key: grid[key]/self.scaling[key] for key in grid.keys()}
        return scaled_grid

    def renovate(self, actions):
        """
        Executes a step in the environment based on the given actions.

        Args:
        - actions (list of tuples): Each tuple represents a renovation action as (x, r_c, r_r, r_poi, FAR),

        Returns:
        - next_state (dict, list): The updated grid attributes after the actions.
        - reward (float): The total reward from this step.
        - done (bool): Whether the episode has ended.
        - info (dict): Additional information (e.g., breakdown of rewards).
        """
        grid = self.current_state
        old_population = grid["pop"].clone()
        old_POI = grid["POI"].clone()
        old_transportation, old_avg = self.Transportation.calc_transport_time()

        # Track renovated area
        area_this_step = 0

        # Calculate rewards
        R_M = torch.tensor(0.0)  # Monetary reward
        repetitive_penalty = 0

        for idx, r_c, r_r, r_poi, FAR in actions:
            # Renovation parameters
            x = self.idx_to_position[int(idx)]
            i, j = self.current_villages[x][0].astype(int), self.current_villages[x][1].astype(int)
            AREA = self.current_villages[x][2]
            if AREA == 0:
                repetitive_penalty += self.repetitive_penalty
            # r_c, r_r, r_poi = self.combinations[comb]
            # FAR = self.FAR_values[f]

            r_b = grid["r_b"][i, j]

            area_this_step += AREA
            self.current_villages[x][2] = 0

            # Sell and rent areas
            sell_space = AREA * FAR * r_c * r_b
            rent_space = AREA * FAR * r_r * r_b
            POI_space = AREA * r_poi * r_b

            # Update grid attributes
            grid["AREA"][i, j] -= AREA  # Reduce renovatable area in this grid
            move_in_people = sell_space / self.space_per_person + self.occupation_rate * rent_space / self.space_per_person
            grid["pop"][i, j] += move_in_people
            self.Transportation.movein(i, j, move_in_people)
            grid["POI"][i, j] += POI_space * self.POI_per_space

            # Monetary reward components
            sell_reward = sell_space * grid["price_c"][i, j]
            rent_reward = rent_space * grid["price_r"][i, j] * 12 * self.max_year # Annual rent revenue
            cost = (
                AREA * self.plot_ratio * (0.15 + self.monetary_compensation_ratio) * grid["price_c"][i, j]
                + (sell_space + rent_space) * self.construction_cost_ratio * grid["price_c"][i, j]
                + POI_space * self.POI_plot_ratio * grid["price_c"][i, j]
            )
            R_M += sell_reward + rent_reward - cost
            # if sell_reward + rent_reward - cost < -1:
                # print(f"INFO {AREA}, {FAR}, {r_c}, {r_r}, {r_poi}, {sell_space}, {rent_space}, {POI_space}, {sell_reward}, {rent_reward}, {cost}, {R_M}")

            # Adjust adjacent grids' prices

            # print(f"INFO {i}, {j}, {AREA}, {FAR}, {r_c}, {r_r}, {r_poi}, {sell_space}, {rent_space}, {POI_space}, {sell_reward}, {rent_reward}, {cost}, {R_M}")

            # self.update_adjacent_prices(i, j, grid, prev_POI)

        self.update_prices(grid, old_POI)
        # Global changes
        self.current_year += 1
        if self.price_changes is not None:
            grid["price_c"] *= 1 + self.price_changes[self.current_year-1]
            grid["price_r"] *= 1 + self.price_changes[self.current_year-1]
        else:
            grid["price_c"] *= 1 + self.speed_c
            grid["price_r"] *= 1 + self.speed_r
        grid["price_c"] /= 1 + self.inflation_rate
        grid["price_r"] /= 1 + self.inflation_rate

        # Transportation reward
        new_transportation, new_avg = self.Transportation.calc_transport_time()
        R_T = old_avg - new_avg
        R_T_ratio = R_T / old_avg

        # POI reward
        avg_POI_new = self.get_avg_poi(poi=grid['POI'], pop=grid['pop'])
        avg_POI_old = self.get_avg_poi(poi=old_POI, pop=old_population)
        R_P = avg_POI_new - avg_POI_old
        R_P_ratio = R_P / avg_POI_new

        # print(f'Popu_old: {old_population.sum()}, POI_old: {old_POI.sum()}, Popu_new: {grid["pop"].sum()}, POI_new: {grid["POI"].sum()}, R_P: {R_P}')
        # Apply reward weights
        weighted_R_M = self.monetary_weight * R_M
        weighted_R_T = self.transportation_weight * R_T
        weighted_R_P = self.POI_weight * R_P

        # diff = max(area_this_step - self.recommended_per_year, 0)
        cost_balance = self.compute_cost_balance(area_this_step)

        # Total reward
        total_reward = weighted_R_M + weighted_R_T + weighted_R_P - cost_balance - repetitive_penalty

        # Check if the episode is done
        done = self.current_year >= self.max_year

        # old_transportation[old_transportation == 0] = float('inf')
        # old_POI[old_POI < 1.0e-7] = float('inf')

        # Return next state, reward, done, and info

        info = {
            "AREA": area_this_step,
            "R_M": R_M.item(),
            "R_T": R_T.item(),
            "R_P": R_P.item(),
            "weighted_R_M": weighted_R_M.item(),
            "weighted_R_T": weighted_R_T.item(),
            "weighted_R_P": weighted_R_P.item(),
            "cost_balance": cost_balance,
            "R_T_ratio": R_T_ratio.item(),
            "R_P_ratio": R_P_ratio.item()
            # "repetitive_penalty": repetitive_penalty
            # "POI_change": grid["POI"] - old_POI,
            # "Transportation_change": R_T_2d / old_transportation
        }
        # if grid_info:
        #     info.update(info_grid)
        return (grid, self.current_villages), total_reward.item(), done, info


    # def get_natural_price(self):
    #     price_c = self.original_state['price_c'].clone().cpu().numpy()
    #     price_r = self.original_state['price_r'].clone().cpu().numpy()
    #     if self.price_changes is not None:
    #         price_c *= 1 + self.price_changes[self.current_year-1]
    #         price_r *= 1 + self.price_changes[self.current_year-1]
    #     else:
    #         price_c *= 1 + self.speed_c
    #         price_r *= 1 + self.speed_r
    #     grid["price_c"] /= 1 + self.inflation_rate
    #     grid["price_r"] /= 1 + self.inflation_rate
        
    def get_state(self):
        trans, _ = self.Transportation.calc_transport_time(self.current_state['pop'])
        grid = {key: value.clone() for key, value in self.current_state.items()}
        grid.update({"Trans": trans})
        return grid
        

    def compute_poi(self, POI: torch.Tensor):
        kernel_size = 2 * self.POI_affect_range + 1
        kernel = torch.ones((1, 1, kernel_size, kernel_size), 
                            dtype=POI.dtype, device=POI.device)

        expanded_poi = POI.unsqueeze(0).unsqueeze(0)
        poi_sum = F.conv2d(expanded_poi, kernel, padding=self.POI_affect_range)
        return poi_sum


    def update_prices(self, grid, old_POI: torch.Tensor):
        """
        Updates price_c and price_r in-place based on the ratio of local sums of new_POI vs old_POI,
        using 2D convolution. If old_poi_sum == 0 in a neighborhood, we set ratio=1 (no change).
        
        Args:
        old_POI:  [n, m] float32 on GPU
        """

        old_flat = self.compute_poi(old_POI).view(-1)
        new_flat = self.compute_poi(grid["POI"]).view(-1)

        # ratio_flat => if old_flat[i] > 0 => new_flat[i]/old_flat[i], else => 1
        ratio_flat = torch.where(
            old_flat > 1e-8,
            1 + (new_flat / old_flat - 1) / 50,      # normal ratio
            torch.ones_like(new_flat) # old=0 => ratio=1
        )

        ratio = ratio_flat.view_as(old_POI)  # [n,m]

        # 5) Multiply prices in-place
        grid["price_c"] *= ratio
        grid["price_r"] *= ratio

    def compute_cost_balance(self, area):
        diff         = np.max([area - self.recommended_per_year * self.balance_range, self.recommended_per_year / self.balance_range - area, 0.0])
        cost_balance = diff if self.balance_func == 'l_1' else (diff ** 2)
        cost_balance *= self.balance_alpha 
        # print(f"BALANCE {(area, diff, cost_balance)}", flush=True)
        return cost_balance
            

    # def update_prices(self, grid, old_POI):
    #     """
    #     Updates the prices of adjacent grids within the POI affect range.

    #     Args:
    #     - i, j (int): Indices of the renovated grid.
    #     - grid (dict): Current grid attributes.
    #     - old_POI (numpy array): POI values before renovation.
    #     """
    #     for x in range(self.n):
    #         for y in range(self.m):
    #             POI_before = old_POI[max(0, x - self.POI_affect_range):min(self.n, x + self.POI_affect_range + 1),
    #                                     max(0, y - self.POI_affect_range):min(self.m, y + self.POI_affect_range + 1)].sum()
    #             POI_after = grid["POI"][max(0, x - self.POI_affect_range):min(self.n, x + self.POI_affect_range + 1),
    #                                         max(0, y - self.POI_affect_range):min(self.m, y + self.POI_affect_range + 1)].sum()
    #             if POI_before > 0:
    #                 # abc = POI_after / POI_before
    #                 # if np.isinf(abc):
    #                 #     print(f"{x} {y} {POI_before} {POI_after}")
    #                 grid["price_c"][x, y] *= POI_after / POI_before
    #                 grid["price_r"][x, y] *= POI_after / POI_before

    # def update_adjacent_prices(self, i, j, grid, old_POI):
    #     """
    #     Updates the prices of adjacent grids within the POI affect range.

    #     Args:
    #     - i, j (int): Indices of the renovated grid.
    #     - grid (dict): Current grid attributes.
    #     - old_POI (numpy array): POI values before renovation.
    #     """
    #     for x in range(max(0, i - self.POI_affect_range), min(self.n, i + self.POI_affect_range + 1)):
    #         for y in range(max(0, j - self.POI_affect_range), min(self.m, j + self.POI_affect_range + 1)):
    #             if (x, y) != (i, j):
    #                 POI_before = old_POI[max(0, x - self.POI_affect_range):min(self.n, x + self.POI_affect_range + 1),
    #                                      max(0, y - self.POI_affect_range):min(self.m, y + self.POI_affect_range + 1)].sum()
    #                 POI_after = grid["POI"][max(0, x - self.POI_affect_range):min(self.n, x + self.POI_affect_range + 1),
    #                                          max(0, y - self.POI_affect_range):min(self.m, y + self.POI_affect_range + 1)].sum()
    #                 if POI_before > 0:
    #                     grid["price_c"][x, y] *= POI_after / POI_before
    #                     grid["price_r"][x, y] *= POI_after / POI_before