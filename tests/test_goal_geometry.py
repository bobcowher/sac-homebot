# tests/test_goal_geometry.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import math
import numpy as np
from goal_geometry import bearing, distance, eval_step_budget, GOAL_RADIUS, ROBOT_STEP_PX, EVAL_BUDGET_MULT


def test_bearing_straight_ahead():
    """Robot at origin, heading 0 (east), goal directly ahead (+x): b=0 -> (sin0, cos0) = (0, 1)."""
    b = bearing(rx=0.0, ry=0.0, rtheta=0.0, gx=100.0, gy=0.0)
    assert b.shape == (2,)
    assert abs(b[0]) < 1e-6, f"sin should be 0, got {b[0]}"
    assert abs(b[1] - 1.0) < 1e-6, f"cos should be 1, got {b[1]}"


def test_bearing_goal_behind():
    """Robot heading east, goal directly behind: b=pi -> (sin(pi), cos(pi)) = (0, -1)."""
    b = bearing(rx=0.0, ry=0.0, rtheta=0.0, gx=-50.0, gy=0.0)
    assert abs(b[0]) < 1e-6, f"sin should ~0, got {b[0]}"
    assert abs(b[1] - (-1.0)) < 1e-6, f"cos should be -1, got {b[1]}"


def test_bearing_goal_left():
    """Robot heading east (+x), goal due north (+y): b=pi/2 -> (1, 0)."""
    b = bearing(rx=0.0, ry=0.0, rtheta=0.0, gx=0.0, gy=100.0)
    assert abs(b[0] - 1.0) < 1e-6, f"sin should be 1 (goal to the left), got {b[0]}"
    assert abs(b[1]) < 1e-6, f"cos should be 0, got {b[1]}"


def test_bearing_independent_of_distance():
    """Near and far goals in same direction give identical bearing vectors (no range leak)."""
    near  = bearing(rx=0.0, ry=0.0, rtheta=0.5, gx=10.0,  gy=10.0)
    far   = bearing(rx=0.0, ry=0.0, rtheta=0.5, gx=1000.0, gy=1000.0)
    assert np.allclose(near, far, atol=1e-6), \
        f"bearing must not encode distance: near={near}, far={far}"


def test_bearing_is_unit_circle():
    """Output vector should always lie on the unit circle."""
    for rx, ry, rtheta, gx, gy in [
        (0, 0, 0, 1, 0),
        (100, 200, 1.2, 300, 150),
        (500, 500, -2.7, 100, 100),
    ]:
        b = bearing(rx, ry, rtheta, gx, gy)
        norm = math.hypot(float(b[0]), float(b[1]))
        assert abs(norm - 1.0) < 1e-6, f"norm should be 1.0, got {norm}"


def test_bearing_returns_float32():
    b = bearing(0, 0, 0, 1, 1)
    assert b.dtype == np.float32


def test_distance_basic():
    assert abs(distance(0, 0, 3, 4) - 5.0) < 1e-9
    assert abs(distance(0, 0, 0, 0)) < 1e-9


def test_eval_step_budget_grows_with_distance():
    """Budget should increase as initial distance increases."""
    b_small  = eval_step_budget(10.0)
    b_medium = eval_step_budget(100.0)
    b_large  = eval_step_budget(500.0)
    assert b_small <= b_medium <= b_large, \
        f"budget must grow: {b_small}, {b_medium}, {b_large}"


def test_eval_step_budget_minimum():
    """Budget never drops below EVAL_BUDGET_MULT (1 step scaled by mult)."""
    b = eval_step_budget(0.0)
    assert b >= EVAL_BUDGET_MULT, f"min budget should be >= {EVAL_BUDGET_MULT}, got {b}"


def test_eval_step_budget_formula():
    """Check the formula for a known distance."""
    d = 100.0
    expected = EVAL_BUDGET_MULT * math.ceil(max(d, GOAL_RADIUS) / ROBOT_STEP_PX)
    assert eval_step_budget(d) == expected
