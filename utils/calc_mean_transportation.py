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

    def calc_transport_time(self, population):
        """
        Calculates the sum of transportation times across all grids.

        Args:
        - population (np.ndarray): Population grid of shape (n, m).

        Returns:
        - float: Sum of transportation times across all grids.
        """
        # Flatten the population grid to match the distance matrix
        population = torch.tensor(population.flatten(), dtype=torch.float32)  # Shape: (n*m,)

        # Compute weighted distances
        numerator = torch.sum(population)  # Total population
        denominator = torch.matmul(population / self.distance_matrix, population)  # Weighted sum

        # print(numerator)
        # print(denominator)

        # Avoid division by zero
        denominator[denominator == 0] = float('inf')

        # Transportation time for all grids
        transport_time = numerator / denominator

        # Return the sum of transportation times
        return torch.sum(transport_time).item()
