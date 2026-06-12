from agent import Agent
import gymnasium as gym
import homebot  # noqa: F401  (side-effect env registration)


def make_env():
    # Remote homebot registers lowercase -v1, local registers -V1.
    for env_id in ("HomeBot2D-v1", "HomeBot2D-V1"):
        try:
            env = gym.make(
                env_id,
                render_mode="rgb_array",
                action_mode="discrete",
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

# Exp 14b: warm-start conv1-3 from the Exp 13 nav primitive (run 237).
# fc layers stay fresh — only perception transfers.
agent = Agent(env=env, max_buffer_size=200000,
              pretrained_conv="pretrained/nav_primitive.pt")

agent.train(episodes=2500, batch_size=64)
