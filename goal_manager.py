# goal_manager.py
"""Owns the current goal waypoint: samples reachable tiles within a curriculum
radius of the robot, and exposes the polar goal vector + achieved goal."""
import numpy as np
from goal_geometry import polar_goal, distance, GOAL_RADIUS


class GoalManager:
    def __init__(self, radius_px: float, rng: np.random.Generator | None = None):
        self.radius_px = float(radius_px)
        self.rng = rng or np.random.default_rng()
        self.goal_px = (0.0, 0.0)

    def set_radius(self, radius_px: float):
        self.radius_px = float(radius_px)

    def reset(self, base):
        """Sample a reachable tile within radius_px of the robot (rejection sample).

        Guarantees GOAL_RADIUS < dist <= radius_px so the goal is neither already
        reached nor outside the curriculum band. Falls back to the nearest
        in-band candidate if rejection fails within budget.
        """
        rx, ry = base._robot.x, base._robot.y
        tiles = base._map.valid_floor_tiles()
        best = None
        for _ in range(200):
            tx, ty = tiles[int(self.rng.integers(len(tiles)))]
            gx, gy = base._map.tile_to_pixel(tx, ty)
            d = distance(rx, ry, gx, gy)
            if GOAL_RADIUS < d <= self.radius_px:
                self.goal_px = (float(gx), float(gy))
                return
            if d > GOAL_RADIUS and (best is None or d < best[0]):
                best = (d, float(gx), float(gy))
        self.goal_px = (best[1], best[2]) if best else (float(rx), float(ry))

    def goal_vector(self, base) -> np.ndarray:
        return polar_goal(base._robot.x, base._robot.y, base._robot.angle,
                          self.goal_px[0], self.goal_px[1])

    def achieved_px(self, base):
        return (float(base._robot.x), float(base._robot.y))
