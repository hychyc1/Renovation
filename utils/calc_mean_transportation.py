import numpy as np
import torch
class Transportation:
    def __init__(self, n, m):
        """
        Initializes the Transportation class with a precomputed distance matrix.

        Args:
        - cfg: Configuration object containing grid size.
        """
        self.n, self.m = n, m

        # Create all grid coordinates
        coords = np.array([(x, y) for x in range(self.n) for y in range(self.m)], dtype=np.float32)
        
        # Compute pairwise distances using NumPy
        distance_matrix_np = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        
        # Convert to PyTorch tensor
        self.distance_matrix = torch.tensor(distance_matrix_np, dtype=torch.float32)
        self.distance_matrix.fill_diagonal_(float('inf'))

        self.distance_matrix_2 = self.distance_matrix.pow(2)


    def calc_transport_time(self, population):
        """
        Calculates the sum of transportation times across all grids.

        Args:
        - population (np.ndarray): Population grid of shape (n, m).

        Returns:
        - float: Sum of transportation times across all grids.
        """

        population = population.flatten()  # shape: [N]
        population = torch.tensor(population)
        population = population.unsqueeze(0)

        # 3) Broadcast population to shape [1, N], dist is [N, N]
        # => broadcasted result is [N, N]
        # numerator(i) = sum_j ( pop[j] / dist(i,j) )
        numerator = (population / self.distance_matrix).sum(dim=1)  # shape [N]

        # denominator(i) = sum_j ( pop[j] / dist(i,j)^2 )
        denominator = (population / self.distance_matrix_2).sum(dim=1)  # shape [N]

        # 4) If denominator is exactly 0, set it to infinity
        # ratio(i) => 0 if denominator is 0
        denominator[denominator == 0] = float('inf')

        dis = numerator / denominator  # shape [N]

        avg = (dis * population).sum() / population.sum()

        # 5) Reshape back to (n, m)
        dis_2d = dis.view(self.n, self.m)
        
        return dis_2d, avg
