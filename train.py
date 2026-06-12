from agent import Agent
import gymnasium as gym
import homebot  # noqa: F401  (side-effect env registration)


def make_env():
    # Remote homebot registers lowercase -v1, local registers -V1.
    for env_id in ("HomeBot2D-v1", "HomeBot2D-V1"):
        try:
            env = gym.make(
                env_id,
                render_mode="rgb_array",   # "human", "rgb_array", or None
                action_mode="continuous",  # SAC: continuous Box action space
                obs_resolution=(96, 96),   # observation image size (H, W)
                n_trash=2,                 # trash items per episode
                max_steps=1000,            # steps before truncation
                map_name="default",        # map layout
                goals=["trash"],
                # subgoals omitted: defaults to False (plain trash task). The
                # local -V1 build predates the kwarg and rejects it outright.
            )
            print(f"Env: {env_id}")
            return env
        except gym.error.Error:
            continue
    raise RuntimeError("No HomeBot2D env id registered")


env = make_env()

agent = Agent(env=env, max_buffer_size=200000)

agent.train(episodes=800, offline_training_epochs=400, batch_size=32)
