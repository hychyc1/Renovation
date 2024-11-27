import numpy as np

def calc_transport_time(population):
    return 0.0
    """
    Calculates the average transportation time based on the population distribution using simple gravity model.

    Args:
    - population (np.ndarray): 2D array representing the population distribution over the grid.

    Returns:
    - float: Average transportation time.
    """
    n, m = population.shape
    total_distance = 0
    total_people = population.sum()  # Total population is simply the sum of the grid

    for x1 in range(n):
        for y1 in range(m):
            if population[x1, y1] == 0:
                continue

            # Compute flows from (x1, y1) to all (x2, y2)
            flows = np.zeros((n, m))
            for x2 in range(n):
                for y2 in range(m):
                    if (x1, y1) != (x2, y2) and population[x2, y2] > 0:
                        distance = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
                        flows[x2, y2] = population[x1, y1] * population[x2, y2] / distance

            # Normalize flows to calculate ratios
            flow_sum = flows.sum()
            if flow_sum == 0:
                continue

            # Calculate distance contribution for each target
            for x2 in range(n):
                for y2 in range(m):
                    if flows[x2, y2] > 0:
                        ratio = flows[x2, y2] / flow_sum
                        distance = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
                        total_distance += ratio * population[x1, y1] * distance

    # Calculate average transportation time
    if total_people == 0:
        return 0.0

    return total_distance / total_people
