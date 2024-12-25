import numpy as np
from utils.calc_mean_transportation import Transportation
from Config.config import Config

class RenovationEnv:
    def __init__(self, cfg: Config, grid_info):
        """
        Initializes the environment.

        Args:
        - cfg (object): Configuration object containing:
            - cfg.plot_ratio (float): Plot ratio for renovation costs.
            - cfg.POI_plot_ratio (float): POI plot ratio for renovation costs.
            - cfg.monetary_compensation_ratio (float): Compensation ratio for renovation costs.
            - cfg.speed_c (float): Annual increase rate for commercial prices.
            - cfg.speed_r (float): Annual increase rate for rental prices.
            - cfg.max_yr (int): Maximum number of years before termination.
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
        """
        self.n, self.m = next(iter(grid_info.values())).shape
        self.Transportation = Transportation(self.n, self.m)
        self.original_state = {key: value.copy() for key, value in grid_info.items()}
        self.current_state = {key: value.copy() for key, value in grid_info.items()}

        # Global parameters
        self.plot_ratio = cfg.plot_ratio
        self.POI_plot_ratio = cfg.POI_plot_ratio
        self.monetary_compensation_ratio = cfg.monetary_compensation_ratio
        self.speed_c = cfg.speed_c
        self.speed_r = cfg.speed_r
        self.max_yr = cfg.max_yr
        self.POI_affect_range = cfg.POI_affect_range
        self.inflation_rate = cfg.inflation_rate
        self.space_per_person = cfg.space_per_person
        self.POI_per_space = cfg.POI_per_space
        self.occupation_rate = cfg.occupation_rate
        self.combinations = cfg.combinations
        self.FAR_values = cfg.FAR_values
        self.balance_alpha = cfg.balance_alpha
        
        areas = np.sort(self.current_state['AREA'].flatten())[::-1]
        max_area = areas[:cfg.grid_per_year * cfg.max_yr].sum()
        self.recommended_per_year = max_area / cfg.max_yr * cfg.balance_upper

        # Reward weights
        self.monetary_weight = cfg.monetary_weight
        self.transportation_weight = cfg.transportation_weight
        self.POI_weight = cfg.POI_weight

        # Counter
        self.current_year = 0

    def reset(self):
        """
        Resets the environment to its initial state.

        Returns:
        - state (dict): The initial state of the grid, including all attributes.
        """
        self.current_state = {key: value.copy() for key, value in self.original_state.items()}
        self.current_year = 0
        return self.current_state

    def step(self, actions):
        """
        Executes a step in the environment based on the given actions.

        Args:
        - actions (list of tuples): Each tuple represents a renovation action as (i, j, comb, f),
                                    where i, j are grid indices, comb is a renovation combination index,
                                    and f is a FAR value index.

        Returns:
        - next_state (dict): The updated grid attributes after the actions.
        - reward (float): The total reward from this step.
        - done (bool): Whether the episode has ended.
        - info (dict): Additional information (e.g., breakdown of rewards).
        """
        grid = self.current_state
        old_population = grid["pop"].copy()
        old_POI = grid["POI"].copy()

        # Track renovated area
        area_this_step = 0

        # Calculate rewards
        R_M = 0  # Monetary reward
        for i, j, comb, f in actions:
            # Renovation parameters
            r_c, r_r, r_poi = self.combinations[comb]
            FAR = self.FAR_values[f]
            AREA = grid["AREA"][i, j]
            r_b = grid["r_b"][i, j]

            area_this_step += AREA

            # Sell and rent areas
            sell_space = AREA * FAR * r_c * r_b
            rent_space = AREA * FAR * r_r * r_b
            POI_space = AREA * r_poi * r_b

            # Update grid attributes
            grid["AREA"][i, j] = 0  # No area left to renovate
            grid["pop"][i, j] += sell_space / self.space_per_person + self.occupation_rate * rent_space / self.space_per_person
            grid["POI"][i, j] += POI_space * self.POI_per_space

            # Monetary reward components
            sell_reward = sell_space * grid["price_c"][i, j]
            rent_reward = rent_space * grid["price_r"][i, j] * 12 * self.max_yr # Annual rent revenue
            cost = (
                AREA * self.plot_ratio * self.monetary_compensation_ratio * grid["price_c"][i, j]
                + POI_space * self.POI_plot_ratio * grid["price_c"][i, j]
            )
            R_M += sell_reward + rent_reward - cost
            # if sell_reward + rent_reward - cost < -1:
                # print(f"INFO {AREA}, {FAR}, {r_c}, {r_r}, {r_poi}, {sell_space}, {rent_space}, {POI_space}, {sell_reward}, {rent_reward}, {cost}, {R_M}")

            # Adjust adjacent grids' prices

            self.update_adjacent_prices(i, j, grid, old_POI)
            # print(f"INFO {i}, {j}, {AREA}, {FAR}, {r_c}, {r_r}, {r_poi}, {sell_space}, {rent_space}, {POI_space}, {sell_reward}, {rent_reward}, {cost}, {R_M}")

        # Global changes
        self.current_year += 1
        grid["price_c"] *= 1 + self.speed_c
        grid["price_r"] *= 1 + self.speed_r
        grid["price_c"] /= 1 + self.inflation_rate
        grid["price_r"] /= 1 + self.inflation_rate

        # Transportation reward
        old_transportation = self.Transportation.calc_transport_time(old_population)
        R_T_2d = old_transportation - self.Transportation.calc_transport_time(grid["pop"])
        R_T = R_T_2d.sum().item()

        # POI reward
        avg_POI_new = grid["POI"].sum() / grid["pop"].sum()
        avg_POI_old = old_POI.sum() / old_population.sum()
        R_P = avg_POI_new - avg_POI_old

        # print(f'Popu_old: {old_population.sum()}, POI_old: {old_POI.sum()}, Popu_new: {grid["pop"].sum()}, POI_new: {grid["POI"].sum()}, R_P: {R_P}')
        # Apply reward weights
        weighted_R_M = self.monetary_weight * R_M
        weighted_R_T = self.transportation_weight * R_T
        weighted_R_P = self.POI_weight * R_P

        diff = max(area_this_step - self.recommended_per_year, 0)
        cost_balance = self.balance_alpha * (diff ** 2)

        # Total reward
        total_reward = weighted_R_M + weighted_R_T + weighted_R_P - cost_balance

        # Check if the episode is done
        done = self.current_year >= self.max_yr

        # Return next state, reward, done, and info

        old_transportation[old_transportation == 0] = float('inf')
        old_POI[old_POI < 1.0e-7] = float('inf')

        return grid, total_reward, done, {
            "AREA": area_this_step,
            "R_M": R_M,
            "R_T": R_T,
            "R_P": R_P,
            "weighted_R_M": weighted_R_M,
            "weighted_R_T": weighted_R_T,
            "weighted_R_P": weighted_R_P,
            "cost_balance": cost_balance,
            "POI_change": (grid["POI"] - old_POI) / old_POI,
            "Transportation_change": R_T_2d / old_transportation
        }

    def update_adjacent_prices(self, i, j, grid, old_POI):
        """
        Updates the prices of adjacent grids within the POI affect range.

        Args:
        - i, j (int): Indices of the renovated grid.
        - grid (dict): Current grid attributes.
        - old_POI (numpy array): POI values before renovation.
        """
        for x in range(max(0, i - self.POI_affect_range), min(self.n, i + self.POI_affect_range + 1)):
            for y in range(max(0, j - self.POI_affect_range), min(self.m, j + self.POI_affect_range + 1)):
                if (x, y) != (i, j):
                    POI_before = old_POI[max(0, x - self.POI_affect_range):min(self.n, x + self.POI_affect_range + 1),
                                         max(0, y - self.POI_affect_range):min(self.m, y + self.POI_affect_range + 1)].sum()
                    POI_after = grid["POI"][max(0, x - self.POI_affect_range):min(self.n, x + self.POI_affect_range + 1),
                                             max(0, y - self.POI_affect_range):min(self.m, y + self.POI_affect_range + 1)].sum()
                    if POI_before > 0:
                        grid["price_c"][x, y] *= POI_after / POI_before
                        grid["price_r"][x, y] *= POI_after / POI_before