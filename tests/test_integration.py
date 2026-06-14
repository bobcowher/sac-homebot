# tests/test_integration.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import gymnasium as gym
import homebot  # noqa: F401
from agent import Agent


def _make_env(max_steps=50):
    for env_id in ("HomeBot2D-Goal-v1", "HomeBot2D-Goal-V1"):
        try:
            env = gym.make(
                env_id,
                render_mode="rgb_array",
                action_mode="discrete",
                obs_resolution=(96, 96),
                n_trash=2,
                max_steps=max_steps,
                map_name="default",
                goals=["collect_trash"],
            )
            return env
        except gym.error.Error:
            continue
    raise RuntimeError("No HomeBot2D-Goal env id registered")


def test_three_episode_train_runs_without_error():
    env = _make_env(max_steps=50)
    agent = Agent(env=env, max_buffer_size=10000)
    agent.train(episodes=3, batch_size=64, eval_interval=100)  # eval_interval>3 so eval is skipped
    env.close()
