# models/goal_actor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from models.base import BaseModel, weights_init_

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPS = 1e-6


class GoalActor(BaseModel):
    def __init__(self, input_shape, goal_dim, n_actions, hidden_dim,
                 action_space=None, checkpoint_dir='checkpoints', name='goal_actor'):
        super().__init__()
        c, h, w = input_shape
        self.conv1 = nn.Conv2d(c, 32, 3, stride=2, padding=1)    # 96->48
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)   # 48->24
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # 24->12
        self.flatten = nn.Flatten()
        conv_dim = 128 * (h // 8) * (w // 8)

        self.linear1 = nn.Linear(conv_dim + goal_dim, hidden_dim)
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

    def _features(self, img, goal):
        x = F.relu(self.conv1(img))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = torch.cat([x, goal], dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return x

    def forward(self, img, goal):
        x = self._features(img, goal)
        mean = self.mean_linear(x)
        log_std = torch.clamp(self.log_std_linear(x), LOG_SIG_MIN, LOG_SIG_MAX)
        return mean, log_std

    def sample(self, img, goal):
        mean, log_std = self.forward(img, goal)
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
