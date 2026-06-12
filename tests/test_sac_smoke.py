# tests/test_sac_smoke.py
# Smoke test: SAC train loop runs end-to-end on the continuous V1 trash env,
# exercising warmup, n-step sampling, actor-critic update, and rolling-best
# checkpoint gating without error.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import gymnasium as gym
import homebot  # noqa: F401
from agent import Agent


def _make_env():
    for env_id in ("HomeBot2D-v1", "HomeBot2D-V1"):
        try:
            return gym.make(
                env_id,
                render_mode="rgb_array",
                action_mode="continuous",
                obs_resolution=(96, 96),
                n_trash=2,
                max_steps=30,
                map_name="default",
                goals=["trash"],
            )
        except gym.error.Error:
            continue
    raise RuntimeError("No HomeBot2D env id registered")


def test_sac_three_episode_train_runs_without_error():
    env = _make_env()
    agent = Agent(env=env, max_buffer_size=2000)
    agent.train(episodes=3, offline_training_epochs=2, batch_size=4,
                warmup_episodes=1, run_tag="pytest-sac-smoke")
    env.close()
