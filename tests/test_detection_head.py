# tests/test_detection_head.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
from models.detection_head import (
    DetectionHead, build_detection_targets, OBS, GRID, DETECT_CHANNELS, K_LABEL_SLOTS,
)


def test_head_forward_shape():
    head = DetectionHead(embed_dim=1024)
    z = torch.rand(7, 1024)
    out = head(z)
    assert out.shape == (7, DETECT_CHANNELS, GRID, GRID)


def test_build_targets_places_blob_at_cell():
    # one frame, one trash object (channel 0) at obs pixel (48, 24)
    labels = torch.full((1, K_LABEL_SLOTS, 3), -1, dtype=torch.int16)
    labels[0, 0] = torch.tensor([0, 48, 24], dtype=torch.int16)  # (channel, x, y)
    tgt = build_detection_targets(labels, device="cpu")
    assert tgt.shape == (1, DETECT_CHANNELS, GRID, GRID)
    gx, gy = 48 * GRID // OBS, 24 * GRID // OBS
    assert tgt[0, 0, gy, gx] == 1.0          # center cell set
    assert tgt[0, 0].sum() >= 1.0            # at least the center
    assert tgt[0, 0, 0, 0] == 0.0            # far corner empty


def test_build_targets_ignores_padding():
    labels = torch.full((2, K_LABEL_SLOTS, 3), -1, dtype=torch.int16)  # all padding
    tgt = build_detection_targets(labels, device="cpu")
    assert tgt.sum() == 0.0
