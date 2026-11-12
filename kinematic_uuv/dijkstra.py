"""Three-dimensional Dijkstra cost field used for path observations."""
from heapq import heappop, heappush

import numpy as np


def build_cost_field(goal, obstacles, start, resolution=8.0,
                     obstacle_margin=5.0, goal_radius=12.0,
                     planning_bounds=((-100.0, -350.0, 0.0),
                                      (700.0, 350.0, 100.0)),
                     vertical_cost_scale=2.0):
    if resolution <= 0.0:
        raise ValueError("resolution must be positive")
    lower = np.asarray(planning_bounds[0], dtype=float)
    upper = np.asarray(planning_bounds[1], dtype=float)
    if np.any(upper <= lower):
        raise ValueError("invalid planning bounds")
    lower = np.floor(lower / resolution) * resolution
    upper = np.ceil(upper / resolution) * resolution
    shape = tuple(np.floor((upper - lower) / resolution).astype(int) + 1)
    axes = [lower[i] + np.arange(shape[i]) * resolution for i in range(3)]
    grid_x, grid_y, grid_z = np.meshgrid(*axes, indexing="ij")

    occupancy = np.zeros(shape, dtype=bool)
    for center, radius in obstacles:
        center = np.asarray(center, dtype=float)
        inflated = float(radius) + obstacle_margin
        occupancy |= ((grid_x - center[0]) ** 2
                      + (grid_y - center[1]) ** 2
                      + (grid_z - center[2]) ** 2 <= inflated ** 2)

    goal_grid = ((grid_x - goal[0]) ** 2
                 + (grid_y - goal[1]) ** 2
                 + (grid_z - goal[2]) ** 2 <= goal_radius ** 2)
    goal_grid &= ~occupancy
    if not goal_grid.any():
        index = tuple(np.clip(
            np.rint((goal - lower) / resolution).astype(int),
            0, np.asarray(shape) - 1,
        ))
        goal_grid[index] = True
        occupancy[index] = False

    costs = np.full(shape, np.inf, dtype=np.float32)
    queue = []
    for index in zip(*np.nonzero(goal_grid)):
        costs[index] = 0.0
        heappush(queue, (0.0, index))
    neighbors = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                length = np.linalg.norm((dx, dy, vertical_cost_scale * dz))
                neighbors.append(((dx, dy, dz), resolution * float(length)))

    while queue:
        current_cost, current = heappop(queue)
        if current_cost > float(costs[current]) + 1e-5:
            continue
        for (dx, dy, dz), edge_cost in neighbors:
            neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
            if not (0 <= neighbor[0] < shape[0]
                    and 0 <= neighbor[1] < shape[1]
                    and 0 <= neighbor[2] < shape[2]):
                continue
            if occupancy[neighbor]:
                continue
            candidate = current_cost + edge_cost
            if candidate < costs[neighbor]:
                costs[neighbor] = candidate
                heappush(queue, (candidate, neighbor))
    return costs, lower, float(resolution), occupancy


def interpolate_cost(costs, origin, resolution, position):
    coordinate = (np.asarray(position, dtype=float) - origin) / resolution
    if np.any(coordinate < 0.0) or np.any(coordinate > np.asarray(costs.shape) - 1):
        return float("inf")
    base = np.floor(coordinate).astype(int)
    fraction = coordinate - base
    value = 0.0
    weight_sum = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                index = base + np.array([dx, dy, dz])
                if np.any(index >= np.asarray(costs.shape)):
                    continue
                weight = float(np.prod(np.where(
                    np.array([dx, dy, dz], dtype=bool),
                    fraction, 1.0 - fraction,
                )))
                cost = float(costs[tuple(index)])
                if np.isfinite(cost):
                    value += weight * cost
                    weight_sum += weight
    return value / weight_sum if weight_sum > 0.0 else float("inf")

