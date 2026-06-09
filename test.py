from agent import Agent
import gymnasium as gym
import homebot

env = gym.make(
    "HomeBot2D-v1",
    render_mode="human",
    action_mode="discrete",
    obs_resolution=(96, 96),
    n_trash=2,
    max_steps=1000,
    map_name="default",
    subgoals=True,
)

agent = Agent(env=env)

agent.load()

agent.test(episodes=10)
