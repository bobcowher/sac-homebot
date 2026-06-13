# tests/test_wm_agent_unit.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import gymnasium as gym
import homebot  # noqa: F401
from agent import Agent
from models.detection_head import K_LABEL_SLOTS


def _make_env():
    for env_id in ("HomeBot2D-v1", "HomeBot2D-V1"):
        try:
            return gym.make(env_id, render_mode="rgb_array", action_mode="continuous",
                            obs_resolution=(96, 96), n_trash=2, max_steps=30,
                            map_name="default", goals=["trash"])
        except gym.error.Error:
            continue
    raise RuntimeError("No HomeBot2D env id registered")


def test_agent_builds_and_warmup_action_matches_space():
    env = _make_env()
    agent = Agent(env=env, max_buffer_size=500)
    a = agent.warmup_action()
    assert a.shape == env.action_space.shape
    assert not hasattr(agent, "decode_action")  # CarRacing mapping removed
    # buffer stores labels of the right shape
    assert agent.memory.label_memory.shape[1:] == (K_LABEL_SLOTS, 3)
    env.close()
