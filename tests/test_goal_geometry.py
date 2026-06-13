# tests/test_goal_geometry.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import math
import numpy as np
from goal_geometry import (
    polar_goal, distance, reach_reward, GOAL_DIM, GOAL_RADIUS, SUCCESS_REWARD,
)


def test_polar_goal_shape_and_facing():
    g = polar_goal(0.0, 0.0, 0.0, 100.0, 0.0)
    assert g.shape == (GOAL_DIM,)
    assert abs(g[1] - 0.0) < 1e-6
    assert abs(g[2] - 1.0) < 1e-6


def test_polar_goal_bearing_left():
    g = polar_goal(0.0, 0.0, 0.0, 0.0, 100.0)
    assert abs(g[1] - math.sin(math.pi / 2)) < 1e-6
    assert abs(g[2] - math.cos(math.pi / 2)) < 1e-6


def test_polar_goal_relative_to_heading():
    g = polar_goal(0.0, 0.0, math.pi / 2, 100.0, 0.0)
    assert abs(g[1] - math.sin(-math.pi / 2)) < 1e-6
    assert abs(g[2] - math.cos(-math.pi / 2)) < 1e-6


def test_distance():
    assert abs(distance(0, 0, 3, 4) - 5.0) < 1e-6


def test_reach_reward_success_terminal():
    r, done = reach_reward(dist_s=100.0, dist_ns=GOAL_RADIUS - 1, gamma=0.99)
    assert done is True
    assert r >= SUCCESS_REWARD


def test_reach_reward_progress_positive():
    r, done = reach_reward(dist_s=200.0, dist_ns=150.0, gamma=0.99)
    assert done is False
    assert r > 0.0


def test_reach_reward_retreat_negative():
    r, done = reach_reward(dist_s=150.0, dist_ns=200.0, gamma=0.99)
    assert done is False
    assert r < 0.0
