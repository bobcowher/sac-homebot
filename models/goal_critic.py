# models/goal_critic.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseModel, weights_init_


class GoalCritic(BaseModel):
    def __init__(self, input_shape, goal_dim, n_actions, hidden_dim,
                 checkpoint_dir='checkpoints', name='goal_critic'):
        super().__init__()
        c, h, w = input_shape
        conv_dim = 128 * (h // 8) * (w // 8)

        def conv_stack():
            return nn.ModuleList([
                nn.Conv2d(c, 32, 3, stride=2, padding=1),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
            ])

        # Two independent encoders + heads (twin Q).
        self.conv_a = conv_stack()
        self.conv_b = conv_stack()
        self.flatten = nn.Flatten()
        in_dim = conv_dim + goal_dim + n_actions
        self.a1 = nn.Linear(in_dim, hidden_dim); self.a2 = nn.Linear(hidden_dim, hidden_dim); self.a_out = nn.Linear(hidden_dim, 1)
        self.b1 = nn.Linear(in_dim, hidden_dim); self.b2 = nn.Linear(hidden_dim, hidden_dim); self.b_out = nn.Linear(hidden_dim, 1)

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = f"{checkpoint_dir}/{name}"
        self.apply(weights_init_)

    def _encode(self, convs, img):
        x = img
        for layer in convs:
            x = F.relu(layer(x))
        return self.flatten(x)

    def forward(self, img, goal, action):
        fa = torch.cat([self._encode(self.conv_a, img), goal, action], dim=1)
        fb = torch.cat([self._encode(self.conv_b, img), goal, action], dim=1)
        q1 = self.a_out(F.relu(self.a2(F.relu(self.a1(fa)))))
        q2 = self.b_out(F.relu(self.b2(F.relu(self.b1(fb)))))
        return q1, q2
