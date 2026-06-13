# tests/test_reacher_integration.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import gymnasium as gym
import homebot  # noqa: F401
from agent_reacher import ReacherAgent


def _make_env():
    for env_id in ("HomeBot2D-v1", "HomeBot2D-V1"):
        try:
            return gym.make(env_id, render_mode="rgb_array", action_mode="continuous",
                            obs_resolution=(96, 96), n_trash=2, max_steps=40,
                            map_name="default", goals=["trash"])
        except gym.error.Error:
            continue
    raise RuntimeError("No HomeBot2D env id registered")


def test_reacher_train_and_eval_run():
    env = _make_env()
    agent = ReacherAgent(env, max_buffer_size=3000, start_radius=150.0)
    agent.train(episodes=12, max_steps=20, batch_size=16, warmup_episodes=2,
                grad_steps=3, eval_every=5, run_tag="pytest-reacher")
    rate = agent.greedy_eval(episodes=2, max_steps=20)
    assert 0.0 <= rate <= 1.0
    env.close()
