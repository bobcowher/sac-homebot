# models/detection_head.py
import torch
import torch.nn as nn

OBS = 96               # observation H/W
GRID = 24              # detection heatmap resolution (each cell ~4px)
DETECT_CHANNELS = 1    # trash only for now; multi-channel-ready
K_LABEL_SLOTS = 4      # max labelled objects stored per frame


class DetectionHead(nn.Module):
    """Shallow goal-object detection head off the per-frame latent.

    One linear layer embed -> (DETECT_CHANNELS, GRID, GRID) logits. Deliberately
    shallow so success means the LATENT encodes object location, not that the head
    re-detects from a rich feature map. Reads the encoder embedding (not the GRU
    state) so it shapes per-frame perception.
    """

    def __init__(self, embed_dim, grid=GRID, channels=DETECT_CHANNELS):
        super().__init__()
        self.grid = grid
        self.channels = channels
        self.fc = nn.Linear(embed_dim, channels * grid * grid)

    def forward(self, embed):
        return self.fc(embed).view(-1, self.channels, self.grid, self.grid)


def build_detection_targets(labels, device, grid=GRID, channels=DETECT_CHANNELS):
    """labels: (B, K_LABEL_SLOTS, 3) int tensor of (channel, x, y); padding = -1.

    Returns (B, channels, grid, grid) float occupancy targets. Each object paints
    a 3x3 block (localization to ~12px; gives the shallow head a learnable gradient
    instead of one needle cell in grid*grid).
    """
    b = labels.shape[0]
    tgt = torch.zeros(b, channels, grid, grid, device=device)
    labels = labels.to(torch.int64)
    for i in range(b):
        for k in range(labels.shape[1]):
            c, x, y = labels[i, k].tolist()
            if c < 0:
                continue
            gx = min(grid - 1, x * grid // OBS)
            gy = min(grid - 1, y * grid // OBS)
            tgt[i, c, max(0, gy - 1):gy + 2, max(0, gx - 1):gx + 2] = 1.0
    return tgt
