"""Train the bearing-conditioned Double-DQN+HER reacher.

The policy is conditioned on an egocentric bearing [sin(b), cos(b)] to the
goal — no range/distance information leaks to the network. Random spawn and
an un-gameable budget-limited greedy eval are used throughout.
"""

import gymnasium as gym
import homebot  # noqa: F401  (side-effect env registration)

from agent import Agent


def make_env():
    # Local homebot registers -V1 (capital), remote registers -v1.
    for env_id in ("HomeBot2D-Goal-v1", "HomeBot2D-Goal-V1"):
        try:
            env = gym.make(
                env_id,
                render_mode="rgb_array",
                action_mode="discrete",
                obs_resolution=(96, 96),
                n_trash=2,
                max_steps=500,
                map_name="default",
                goals=["collect_trash"],
            )
            print(f"Env: {env_id}")
            return env
        except gym.error.Error:
            continue
    raise RuntimeError("No HomeBot2D-Goal env id registered")


env = make_env()
agent = Agent(env=env, max_buffer_size=200000)
agent.train(episodes=3000, batch_size=64, eval_interval=50, eval_episodes=20)
