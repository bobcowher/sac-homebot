# models/goal_actor.py
# MLP actor head operating on a precomputed (shared-encoder) embedding + polar goal.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from models.base import BaseModel, weights_init_

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPS = 1e-6


class GoalActor(BaseModel):
    def __init__(self, embed_dim, goal_dim, n_actions, hidden_dim,
                 action_space=None, checkpoint_dir='checkpoints', name='goal_actor'):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim + goal_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, n_actions)
        self.log_std_linear = nn.Linear(hidden_dim, n_actions)

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = f"{checkpoint_dir}/{name}"
        self.apply(weights_init_)

        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.0)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.0)

    def forward(self, embed, goal):
        x = torch.cat([embed, goal], dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = torch.clamp(self.log_std_linear(x), LOG_SIG_MIN, LOG_SIG_MAX)
        return mean, log_std

    def sample(self, embed, goal):
        mean, log_std = self.forward(embed, goal)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + EPS)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)
