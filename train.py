# train.py
from agent import Agent
import gymnasium as gym
import homebot  # noqa: F401  (side-effect env registration)


def make_env():
    for env_id in ("HomeBot2D-v1", "HomeBot2D-V1"):
        try:
            env = gym.make(
                env_id,
                render_mode="rgb_array",
                action_mode="continuous",  # world-model SAC: continuous Box actions
                obs_resolution=(96, 96),
                n_trash=2,
                max_steps=1000,
                map_name="default",
                goals=["trash"],
            )
            print(f"Env: {env_id}")
            return env
        except gym.error.Error:
            continue
    raise RuntimeError("No HomeBot2D env id registered")


env = make_env()

agent = Agent(env=env, max_buffer_size=100000, wm_sequence_length=50)

agent.train(episodes=1200, offline_training_epochs=200, batch_size=32,
            wm_batch_size=200, imagination_steps=4, real_ratio=0.5)
