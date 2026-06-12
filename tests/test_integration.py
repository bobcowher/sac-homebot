# tests/test_integration.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import gymnasium as gym
import homebot  # noqa: F401
from agent import Agent


def _make_env():
    # Remote homebot registers lowercase -v1, local registers -V1.
    for env_id in ("HomeBot2D-v1", "HomeBot2D-V1"):
        try:
            return gym.make(
                env_id,
                render_mode="rgb_array",
                action_mode="discrete",
                obs_resolution=(96, 96),
                n_trash=2,
                max_steps=50,
                map_name="default",
                goals=["trash"],
            )
        except gym.error.Error:
            continue
    raise RuntimeError("No HomeBot2D env id registered")


def test_three_episode_train_runs_without_error():
    env = _make_env()
    agent = Agent(env=env, max_buffer_size=2000)
    agent.train(episodes=3, batch_size=8, grad_steps=5, run_tag="pytest-smoke")
    env.close()
