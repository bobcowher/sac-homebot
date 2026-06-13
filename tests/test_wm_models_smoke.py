# tests/test_wm_models_smoke.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
from models.encoder import Encoder, Decoder
from models.dynamics_model import DynamicsModel
from models.actor import Actor
from models.critic import Critic
from gymnasium.spaces import Box
import numpy as np


def test_encoder_decoder_roundtrip_shape():
    enc = Encoder(observation_shape=(3, 96, 96), embed_dim=1024)
    dec = Decoder(observation_shape=(3, 96, 96), embed_dim=1024,
                  conv_output_shape=enc.get_output_shape(),
                  conv_channels=enc.get_conv_channels())
    x = torch.rand(2, 3, 96, 96)
    z = enc(x)
    assert z.shape == (2, 1024)
    assert dec(z).shape == (2, 3, 96, 96)


def test_dynamics_residual_shape():
    dyn = DynamicsModel(embed_dim=1024, n_actions=2, hidden_dim=2048)
    z = torch.rand(5, 1024)
    a = torch.rand(5, 2)
    assert dyn(z, a).shape == (5, 1024)


def test_actor_critic_latent_shapes():
    space = Box(low=-np.ones(2, np.float32), high=np.ones(2, np.float32))
    actor = Actor(num_inputs=1024, num_actions=2, hidden_dim=256, action_space=space)
    critic = Critic(num_inputs=1024, num_actions=2, hidden_dim=256)
    z = torch.rand(4, 1024)
    a, logp, mean = actor.sample(z)
    assert a.shape == (4, 2) and logp.shape == (4, 1)
    q1, q2 = critic(z, a)
    assert q1.shape == (4, 1) and q2.shape == (4, 1)
