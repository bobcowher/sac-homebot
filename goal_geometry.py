"""Pure geometry helpers for the bearing-conditioned reacher.

All functions are stateless and unit-testable without any gym import.
"""
import math
import numpy as np

GOAL_RADIUS = 40.0       # px; within this the goal is "reached"
ROBOT_STEP_PX = 4.0      # homebot DISCRETE_SPEED
EVAL_BUDGET_MULT = 3


def bearing(rx: float, ry: float, rtheta: float, gx: float, gy: float) -> np.ndarray:
    """Egocentric bearing to (gx, gy): [sin(b), cos(b)].

    b = atan2(gy - ry, gx - rx) - rtheta  (angle to goal minus robot heading).
    Returns a 2-D unit-circle encoding — NO range information leaks through.
    """
    b = math.atan2(gy - ry, gx - rx) - rtheta
    return np.array([math.sin(b), math.cos(b)], dtype=np.float32)


def distance(ax: float, ay: float, bx: float, by: float) -> float:
    """Euclidean distance between two points in pixel space."""
    return math.hypot(bx - ax, by - ay)


def eval_step_budget(init_dist: float) -> int:
    """Step budget for greedy eval.

    Budget = EVAL_BUDGET_MULT * ceil(max(init_dist, GOAL_RADIUS) / ROBOT_STEP_PX).
    Tight enough that a circling policy cannot sweep-and-pass; grows linearly
    with spawn distance so far-away goals aren't penalised unfairly.
    """
    return EVAL_BUDGET_MULT * max(1, math.ceil(max(init_dist, GOAL_RADIUS) / ROBOT_STEP_PX))
