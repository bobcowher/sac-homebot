# tests/test_world_model_wm.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
from models.world_model import WorldModel
from models.detection_head import GRID, DETECT_CHANNELS, K_LABEL_SLOTS


def _batch(N=2, T=6):
    return {
        "obs":     torch.randint(0, 255, (N, T, 3, 96, 96), dtype=torch.uint8),
        "actions": torch.rand(N, T, 2),
        "rewards": torch.zeros(N, T),
        "dones":   torch.zeros(N, T),
        "labels":  torch.full((N, T, K_LABEL_SLOTS, 3), -1, dtype=torch.int16),
    }


def test_compute_loss_includes_detection():
    wm = WorldModel(observation_shape=(3, 96, 96), embed_dim=1024, n_actions=2)
    batch = _batch()
    # put a trash object in one frame so detection loss is non-trivial
    batch["labels"][0, 0, 0] = torch.tensor([0, 48, 24], dtype=torch.int16)
    loss, d = wm.compute_loss_sequential(batch)
    assert loss.dim() == 0
    assert "detect" in d
    assert d["detect"] >= 0.0


def test_detection_head_shape():
    wm = WorldModel(observation_shape=(3, 96, 96), embed_dim=1024, n_actions=2)
    z = torch.rand(3, 1024)
    out = wm.detection_head(z)
    assert out.shape == (3, DETECT_CHANNELS, GRID, GRID)


def test_imagine_step_shapes():
    wm = WorldModel(observation_shape=(3, 96, 96), embed_dim=1024, n_actions=2)
    embed = torch.rand(4, 1024)
    h_t = torch.rand(4, wm.gru_dim)
    action = torch.rand(4, 2)
    next_embed, next_h_t, _, reward, done = wm.imagine_step(embed, h_t, action)
    assert next_embed.shape == (4, 1024)
    assert reward.shape == (4, 1) and done.shape == (4, 1)
