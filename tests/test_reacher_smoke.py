# tests/test_reacher_smoke.py
"""End-to-end smoke test for the bearing-conditioned Double-DQN+HER reacher.

Tests:
  - 5-episode training run completes without error
  - greedy_eval returns a rate in [0, 1]
  - random spawn produces observations consistent with the new robot pose
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import gymnasium as gym
import homebot  # noqa: F401  (side-effect env registration)
import numpy as np

from agent import Agent
from goal_geometry import bearing as compute_bearing, distance, eval_step_budget


def _make_env(max_steps=30):
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
    raise RuntimeError("No HomeBot2D-Goal env registered")


def test_five_episode_train_runs_without_error():
    """5-episode training with tiny replay warmup — must complete without exception."""
    env = _make_env(max_steps=30)
    agent = Agent(env=env, max_buffer_size=5000)
    # Tiny batch: can_sample requires >= batch_size*10 transitions, so with 30
    # max_steps×5 episodes = 150 transitions we can hit the threshold with batch=8.
    agent.train(
        episodes=5,
        batch_size=8,
        eval_interval=5,   # eval every 5 episodes
        eval_episodes=3,   # tiny eval
    )
    env.close()


def test_greedy_eval_returns_rate_in_0_1():
    """greedy_eval must return a float in [0, 1]."""
    env = _make_env(max_steps=30)
    agent = Agent(env=env, max_buffer_size=2000)
    rate = agent.greedy_eval(n_episodes=4)
    assert isinstance(rate, float), f"expected float, got {type(rate)}"
    assert 0.0 <= rate <= 1.0, f"reach rate must be in [0,1], got {rate}"
    env.close()


def test_random_spawn_updates_achieved_goal():
    """After random spawn, achieved_goal in the fresh obs must reflect the new pose."""
    env = _make_env(max_steps=30)
    agent = Agent(env=env, max_buffer_size=1000)

    for _ in range(5):
        env.reset()
        fresh = agent._random_spawn()
        base  = env.unwrapped
        r     = base._robot

        # achieved_goal in fresh dict must match current robot position
        ag = fresh["achieved_goal"]
        assert abs(float(ag[0]) - float(r.x)) < 1.0, \
            f"achieved_goal x {ag[0]} doesn't match robot.x {r.x}"
        assert abs(float(ag[1]) - float(r.y)) < 1.0, \
            f"achieved_goal y {ag[1]} doesn't match robot.y {r.y}"

    env.close()


def test_bearing_from_agent_is_unit_vector():
    """_robot_bearing must return a 2-D unit-circle vector."""
    import math
    env = _make_env(max_steps=30)
    agent = Agent(env=env, max_buffer_size=1000)
    env.reset()
    fresh = agent._random_spawn()
    dg = fresh["desired_goal"]
    b = agent._robot_bearing(dg)
    assert b.shape == (2,), f"expected shape (2,), got {b.shape}"
    norm = math.hypot(float(b[0]), float(b[1]))
    assert abs(norm - 1.0) < 1e-5, f"bearing not on unit circle: norm={norm}"
    env.close()
