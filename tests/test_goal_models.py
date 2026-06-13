# tests/test_goal_models.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
import numpy as np
from gymnasium.spaces import Box
from models.encoder import Encoder, Decoder
from models.goal_actor import GoalActor
from models.goal_critic import GoalCritic
from goal_geometry import GOAL_DIM

EMBED = 256


def _space():
    return Box(low=-np.ones(2, np.float32), high=np.ones(2, np.float32))


def test_encoder_decoder_roundtrip():
    enc = Encoder(observation_shape=(3, 96, 96), embed_dim=EMBED)
    dec = Decoder(observation_shape=(3, 96, 96), embed_dim=EMBED,
                  conv_output_shape=enc.get_output_shape(),
                  conv_channels=enc.get_conv_channels())
    x = torch.rand(4, 3, 96, 96)
    z = enc(x)
    assert z.shape == (4, EMBED)
    assert dec(z).shape == (4, 3, 96, 96)


def test_actor_head_shapes():
    actor = GoalActor(embed_dim=EMBED, goal_dim=GOAL_DIM, n_actions=2,
                      hidden_dim=256, action_space=_space())
    embed = torch.rand(5, EMBED)
    goal = torch.rand(5, GOAL_DIM)
    a, logp, mean = actor.sample(embed, goal)
    assert a.shape == (5, 2) and logp.shape == (5, 1) and mean.shape == (5, 2)


def test_critic_head_shapes():
    critic = GoalCritic(embed_dim=EMBED, goal_dim=GOAL_DIM, n_actions=2, hidden_dim=256)
    embed = torch.rand(4, EMBED)
    goal = torch.rand(4, GOAL_DIM)
    action = torch.rand(4, 2)
    q1, q2 = critic(embed, goal, action)
    assert q1.shape == (4, 1) and q2.shape == (4, 1)
