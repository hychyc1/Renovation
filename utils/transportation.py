import numpy as np
import torch

class Transportation:
    def __init__(self, n, m, grid_population, extra_grid):
        """
        Initializes the Transportation class.

        Args:
            n (int): Number of rows of the grid.
            m (int): Number of columns of the grid.
            extra_grid (list of tuples): Each tuple is (x, y, population). 
                grid with x and y falling inside the grid are dropped.
        """
        self.n, self.m = n, m

        # 1. Create grid coordinates: these are grid for every grid cell.
        grid_coords = np.array([(x, y) for x in range(n) for y in range(m)], dtype=np.float32)
        N_grid = grid_coords.shape[0]

        # 2. Process extra grid: keep only those outside the grid.
        extra_coords = []
        extra_pops = []
        for (x, y, pop) in extra_grid:
            # Drop grid that lie within the grid boundaries.
            if (not (0 <= x < n and 0 <= y < m)) and pop > 0:
                extra_coords.append([x, y])
                extra_pops.append(pop)
        if extra_coords:
            extra_coords = np.array(extra_coords, dtype=np.float32)
            N_extra = extra_coords.shape[0]
            # Combined coordinates: grid first, then extra grid.
            all_coords = np.concatenate([grid_coords, extra_coords], axis=0)
        else:
            N_extra = 0
            all_coords = grid_coords

        self.N_grid = N_grid      # Number of grid cells.
        self.N_extra = N_extra    # Number of outside grid.
        self.N_total = N_grid + N_extra

        # 3. Compute pairwise distances for all grid.
        # all_coords shape: [N_total, 2]
        diff = all_coords[:, None, :] - all_coords[None, :, :]  # shape: [N_total, N_total, 2]
        distance_matrix_np = np.linalg.norm(diff, axis=-1)        # shape: [N_total, N_total]
        # distance_matrix_np[distance_matrix_np == 0] = float('inf')        # shape: [N_total, N_total]
        # Convert to PyTorch tensor.
        self.distance_matrix = torch.tensor(distance_matrix_np, dtype=torch.float32)
        # Set diagonal to infinity.
        self.distance_matrix.fill_diagonal_(float('inf'))
        # Also compute squared distances.
        self.distance_matrix_2 = self.distance_matrix.pow(2)

        # 4. Initialize populations.
        # For grid grid, we set initial population to 0.
        population_grid =  grid_population.flatten().astype(np.float32)
        # For outside grid, use the populations provided in the list.
        if N_extra > 0:
            population_extra = np.array(extra_pops, dtype=np.float32)
        else:
            population_extra = np.array([], dtype=np.float32)
        # Combined population vector: grid first, then extra grid.
        population_combined = np.concatenate([population_grid, population_extra], axis=0)
        self.population = torch.tensor(population_combined, dtype=torch.float32)
        self.original_population = self.population.clone()

    def reset(self):
        self.population = self.original_population.clone()

    def movein(self, i, j, popu):
        """
        Moves 'popu' people from outside grid into the grid at (x, y).

        The grid population at (x, y) is increased by popu and all outside grid (indices N_grid:]
        are decreased proportionally. That is, if total outside population is P, then each outside point's
        population is multiplied by (P - popu)/P.

        Args:
            x (int): Row index within the grid.
            y (int): Column index within the grid.
            popu (float): Number of people moving into (x, y).
        """
        if not (0 <= i < self.n and 0 <= j < self.m):
            raise ValueError("movein: (x,y) must be within the grid boundaries.")

        # Compute the index of the grid cell (using row-major order).
        grid_index = i * self.m + j

        # Increase population at the grid cell.
        self.population[grid_index] += popu

        # Adjust populations of outside grid.
        if self.N_extra > 0:
            total_outside_pop = self.population[self.N_grid:].sum()
            if total_outside_pop < popu:
                raise ValueError("Not enough population in outside grid for movein operation.")
            factor = (total_outside_pop - popu) / total_outside_pop
            self.population[self.N_grid:] *= factor
        else:
            # If no outside grid exist, raise an error.
            raise ValueError("No outside grid available for movein operation.")

    def calc_transport_time(self):
        """
        Calculates transportation times using the stored population.

        For each point i (grid or outside), we compute:
            numerator(i) = sum_j (population[j] / d(i,j))
            denominator(i) = sum_j (population[j] / d(i,j)^2)
            dis(i) = numerator(i) / denominator(i)
        Then, for grid grid (the first n*m values), dis is reshaped to (n, m) and 
        an overall weighted average transportation time is computed.

        Returns:
            dis_2d (Tensor): Transportation time for grid cells, shape (n, m).
            avg (Tensor): Overall weighted average transportation time (scalar).
        """
        # Use stored population vector (shape: [N_total]) and reshape to [1, N_total].
        pop = self.population.flatten().unsqueeze(0)
        # print(pop)

        # Calculate numerator: for each i, sum_j (pop[j] / distance_matrix[i, j])
        numerator = (pop / self.distance_matrix).sum(dim=1)  # shape: [N_total]
        # print(numerator)
        # Calculate denominator: for each i, sum_j (pop[j] / distance_matrix[i, j]^2)
        denominator = (pop / self.distance_matrix_2).sum(dim=1)  # shape: [N_total]
        # print(denominator)
        # Protect against division by zero.
        denominator[denominator == 0] = float('inf')
        dis = numerator / denominator  # shape: [N_total]

        # Compute overall population-weighted average transportation time.
        avg = (dis * self.population).sum() / self.population.sum()

        # Extract grid transportation times (first N_grid values) and reshape to (n, m).
        dis_2d = dis[:self.N_grid].view(self.n, self.m)
        return dis_2d, avg
