# models/goal_critic.py
# Twin-Q MLP critic head on a precomputed (shared-encoder) embedding + polar goal + action.
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseModel, weights_init_


class GoalCritic(BaseModel):
    def __init__(self, embed_dim, goal_dim, n_actions, hidden_dim,
                 checkpoint_dir='checkpoints', name='goal_critic'):
        super().__init__()
        in_dim = embed_dim + goal_dim + n_actions
        # LayerNorm after each linear — standard SAC value stabiliser.
        self.a1 = nn.Linear(in_dim, hidden_dim); self.a_ln1 = nn.LayerNorm(hidden_dim)
        self.a2 = nn.Linear(hidden_dim, hidden_dim); self.a_ln2 = nn.LayerNorm(hidden_dim)
        self.a_out = nn.Linear(hidden_dim, 1)
        self.b1 = nn.Linear(in_dim, hidden_dim); self.b_ln1 = nn.LayerNorm(hidden_dim)
        self.b2 = nn.Linear(hidden_dim, hidden_dim); self.b_ln2 = nn.LayerNorm(hidden_dim)
        self.b_out = nn.Linear(hidden_dim, 1)

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = f"{checkpoint_dir}/{name}"
        self.apply(weights_init_)

    def forward(self, embed, goal, action):
        x = torch.cat([embed, goal, action], dim=1)
        x1 = F.relu(self.a_ln1(self.a1(x)))
        x1 = F.relu(self.a_ln2(self.a2(x1)))
        q1 = self.a_out(x1)
        x2 = F.relu(self.b_ln1(self.b1(x)))
        x2 = F.relu(self.b_ln2(self.b2(x2)))
        q2 = self.b_out(x2)
        return q1, q2
