# tests/test_goal_labels.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import gymnasium as gym
import homebot  # noqa: F401
from goal_labels import label_rows
from models.detection_head import K_LABEL_SLOTS, OBS


def _make_env():
    for env_id in ("HomeBot2D-v1", "HomeBot2D-V1"):
        try:
            return gym.make(env_id, render_mode="rgb_array", action_mode="continuous",
                            obs_resolution=(OBS, OBS), n_trash=2, max_steps=100,
                            map_name="default", goals=["trash"])
        except gym.error.Error:
            continue
    raise RuntimeError("No HomeBot2D env id registered")


def test_label_rows_shape_and_padding():
    env = _make_env()
    env.reset()
    rows = label_rows(env.unwrapped)
    env.close()
    assert rows.shape == (K_LABEL_SLOTS, 3)
    # every non-padding row is (channel>=0, x in [0,OBS), y in [0,OBS))
    for c, x, y in rows:
        if c >= 0:
            assert 0 <= x < OBS and 0 <= y < OBS
            assert c == 0  # trash channel


def test_label_rows_padding_rows_are_all_minus_one():
    env = _make_env()
    base = env.unwrapped
    base.reset()
    rows = label_rows(base)
    env.close()
    for c, x, y in rows:
        if c < 0:
            assert (int(c), int(x), int(y)) == (-1, -1, -1)
