from agent import Agent
import gymnasium as gym
import homebot

env = gym.make(
    "HomeBot2D-v1",
    render_mode="rgb_array",
    action_mode="discrete",
    obs_resolution=(96, 96),
    n_trash=2,
    max_steps=1000,
    map_name="default",
    goals=["trash"],
    subgoals=True,
)

agent = Agent(env=env, max_buffer_size=100000)

agent.train(episodes=1000, batch_size=64)
