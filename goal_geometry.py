# goal_geometry.py
"""Polar goal representation + potential-based reach reward (pure functions)."""
import math
import numpy as np

GOAL_DIM = 3                 # [range_norm, sin(bearing), cos(bearing)]
GOAL_RANGE_NORM = 500.0      # px normaliser for the goal-VECTOR range only
GOAL_RADIUS = 40.0           # px; within this the goal is "reached"
# Reward scale is DECOUPLED from the goal-vector normaliser. Per-px progress must
# dominate the SAC entropy term (~0.1-0.5/step at alpha=0.1), or the actor just
# inflates entropy and thrashes. 0.1/px => ~0.4 reward for a full-speed step.
PROGRESS_SCALE = 0.03
SUCCESS_REWARD = 1.0


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
    """Potential-based progress shaping + sparse success.

    Phi(s) = -PROGRESS_SCALE * dist(s) ; F = gamma*Phi(s') - Phi(s)
           = PROGRESS_SCALE * (dist_s - gamma*dist_ns)   (Ng 1999: potential-based
    shaping preserves the optimal policy). Plus SUCCESS_REWARD and termination
    when the next state is within GOAL_RADIUS.
    """
    shaping = PROGRESS_SCALE * (dist_s - gamma * dist_ns)
    success = dist_ns < GOAL_RADIUS
    reward = shaping + (SUCCESS_REWARD if success else 0.0)
    return float(reward), bool(success)
