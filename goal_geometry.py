# goal_geometry.py
"""Polar goal representation + potential-based reach reward (pure functions)."""
import math
import numpy as np

GOAL_DIM = 3                 # [range_norm, sin(bearing), cos(bearing)]
GOAL_RANGE_NORM = 500.0      # px range normaliser
GOAL_RADIUS = 40.0           # px; within this the goal is "reached"
SUCCESS_REWARD = 1.0
SHAPING_SCALE = 1.0


def distance(ax, ay, bx, by):
    return math.hypot(bx - ax, by - ay)


def polar_goal(rx, ry, rtheta, gx, gy):
    """Goal in robot-centric polar coords: [range/NORM, sin(bearing), cos(bearing)].

    bearing = angle to goal MINUS robot heading, so it is egocentric: bearing 0
    means "straight ahead", matching the [linear, angular] action.
    """
    dx, dy = gx - rx, gy - ry
    rng = math.hypot(dx, dy)
    bearing = math.atan2(dy, dx) - rtheta
    return np.array([rng / GOAL_RANGE_NORM, math.sin(bearing), math.cos(bearing)],
                    dtype=np.float32)


def reach_reward(dist_s, dist_ns, gamma):
    """Potential-based shaping + sparse success.

    Phi(s) = -dist(s) / GOAL_RANGE_NORM ; F = gamma*Phi(s') - Phi(s)  (Ng 1999:
    potential-based shaping preserves the optimal policy). Plus SUCCESS_REWARD and
    termination when the next state is within GOAL_RADIUS.
    """
    phi_s = -dist_s / GOAL_RANGE_NORM
    phi_ns = -dist_ns / GOAL_RANGE_NORM
    shaping = SHAPING_SCALE * (gamma * phi_ns - phi_s)
    success = dist_ns < GOAL_RADIUS
    reward = shaping + (SUCCESS_REWARD if success else 0.0)
    return float(reward), bool(success)
