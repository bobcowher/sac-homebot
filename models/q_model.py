import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseModel


class QModel(BaseModel):
    def __init__(self, action_dim, input_shape=(3, 96, 96), goal_dim=2,
                 goal_scale=(864.0, 576.0)):
        super(QModel, self).__init__()

        # Goal coords arrive as raw map pixels (default map: 864x576). Scale to
        # [0, 1] so the goal encoder sees the same input range as the obs branch.
        self.register_buffer("goal_scale", torch.tensor(goal_scale, dtype=torch.float32))

        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            flat_size = self._conv_forward(dummy).shape[1]

        self.goal_encoder = nn.Linear(goal_dim, 128)
        self.fc1    = nn.Linear(flat_size + 128, 512)
        self.output = nn.Linear(512, action_dim)

        self.apply(self._weights_init)

        print(f"QModel: input={input_shape}, conv_flat={flat_size}, goal_dim={goal_dim}, actions={action_dim}")

    def _conv_forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.flatten(1)

    def encode_goal(self, goal):
        """Scale raw pixel goal coords to [0, 1] and encode. All goal encoding
        must go through here so the scaling can't be bypassed."""
        return self.goal_encoder(goal / self.goal_scale)

    def forward(self, obs, goal):
        x = self._conv_forward(obs)
        g = self.encode_goal(goal)
        x = torch.cat([x, g], dim=1)
        x = F.relu(self.fc1(x))
        return self.output(x)

    def _weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
