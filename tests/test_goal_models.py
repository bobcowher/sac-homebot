# tests/test_goal_models.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
import numpy as np
from gymnasium.spaces import Box
from models.goal_actor import GoalActor
from models.goal_critic import GoalCritic
from goal_geometry import GOAL_DIM


def _space():
    return Box(low=-np.ones(2, np.float32), high=np.ones(2, np.float32))


def test_actor_shapes():
    actor = GoalActor(input_shape=(3, 96, 96), goal_dim=GOAL_DIM, n_actions=2,
                      hidden_dim=256, action_space=_space())
    img = torch.rand(5, 3, 96, 96)
    goal = torch.rand(5, GOAL_DIM)
    a, logp, mean = actor.sample(img, goal)
    assert a.shape == (5, 2) and logp.shape == (5, 1) and mean.shape == (5, 2)


def test_critic_shapes():
    critic = GoalCritic(input_shape=(3, 96, 96), goal_dim=GOAL_DIM, n_actions=2,
                        hidden_dim=256)
    img = torch.rand(4, 3, 96, 96)
    goal = torch.rand(4, GOAL_DIM)
    action = torch.rand(4, 2)
    q1, q2 = critic(img, goal, action)
    assert q1.shape == (4, 1) and q2.shape == (4, 1)
