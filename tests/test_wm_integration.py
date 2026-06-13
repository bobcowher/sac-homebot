# tests/test_wm_integration.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import gymnasium as gym
import homebot  # noqa: F401
from agent import Agent


def _make_env():
    for env_id in ("HomeBot2D-v1", "HomeBot2D-V1"):
        try:
            return gym.make(env_id, render_mode="rgb_array", action_mode="continuous",
                            obs_resolution=(96, 96), n_trash=2, max_steps=16,
                            map_name="default", goals=["trash"])
        except gym.error.Error:
            continue
    raise RuntimeError("No HomeBot2D env id registered")


def test_wm_train_exercises_full_loop():
    # 12 episodes so the WM-training path (>= min_episodes=10) actually fires:
    # real-label sequence sampling -> compute_loss_sequential, plus AC updates on
    # mixed real/imagined latent rollouts and the rolling-best checkpoint logic.
    env = _make_env()
    agent = Agent(env=env, max_buffer_size=4000, wm_sequence_length=8)
    agent.train(episodes=12, offline_training_epochs=1, batch_size=8,
                wm_batch_size=16, imagination_steps=2, real_ratio=0.5,
                warmup_episodes=1, run_tag="pytest-wm-smoke")
    env.close()
