# train_reacher.py
from agent_reacher import ReacherAgent
import gymnasium as gym
import homebot  # noqa: F401


def make_env():
    for env_id in ("HomeBot2D-v1", "HomeBot2D-V1"):
        try:
            env = gym.make(env_id, render_mode="rgb_array", action_mode="continuous",
                           obs_resolution=(96, 96), n_trash=2, max_steps=300,
                           map_name="default", goals=["trash"])
            print(f"Env: {env_id}")
            return env
        except gym.error.Error:
            continue
    raise RuntimeError("No HomeBot2D env id registered")


env = make_env()
agent = ReacherAgent(env, max_buffer_size=100000, start_radius=150.0, max_radius=600.0)
agent.train(episodes=2000, max_steps=300, batch_size=256, warmup_episodes=10,
            grad_steps=300, eval_every=25)
