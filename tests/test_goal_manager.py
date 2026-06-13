# tests/test_goal_manager.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import gymnasium as gym
import homebot  # noqa: F401
from goal_manager import GoalManager
from goal_geometry import GOAL_DIM, GOAL_RADIUS, distance


def _make_env():
    for env_id in ("HomeBot2D-v1", "HomeBot2D-V1"):
        try:
            return gym.make(env_id, render_mode="rgb_array", action_mode="continuous",
                            obs_resolution=(96, 96), n_trash=2, max_steps=100,
                            map_name="default", goals=["trash"])
        except gym.error.Error:
            continue
    raise RuntimeError("No HomeBot2D env id registered")


def test_goal_within_radius_and_reachable():
    env = _make_env(); env.reset()
    base = env.unwrapped
    gm = GoalManager(radius_px=200.0)
    gm.reset(base)
    rx, ry = base._robot.x, base._robot.y
    d = distance(rx, ry, *gm.goal_px)
    assert GOAL_RADIUS <= d <= 200.0 + 1e-3
    env.close()


def test_goal_vector_shape():
    env = _make_env(); env.reset()
    base = env.unwrapped
    gm = GoalManager(radius_px=200.0); gm.reset(base)
    v = gm.goal_vector(base)
    assert v.shape == (GOAL_DIM,)
    env.close()


def test_set_radius_grows():
    gm = GoalManager(radius_px=100.0)
    gm.set_radius(250.0)
    assert gm.radius_px == 250.0
