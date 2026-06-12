# tests/test_q_model.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
from models.q_model import QModel


def test_forward_returns_correct_shape():
    model = QModel(action_dim=8, input_shape=(3, 96, 96))
    obs = torch.rand(4, 3, 96, 96)
    q   = model(obs)
    assert q.shape == (4, 8), f"expected (4,8), got {q.shape}"


def test_forward_single_sample():
    model = QModel(action_dim=8, input_shape=(3, 96, 96))
    obs = torch.rand(1, 3, 96, 96)
    q   = model(obs)
    assert q.shape == (1, 8)


def test_load_conv_trunk_copies_only_conv(tmp_path):
    """Warm-start must copy conv1-3 and leave fc layers untouched, even when
    the source checkpoint has extra keys (e.g. the goal-conditioned net)."""
    source = QModel(action_dim=8)
    state = source.state_dict()
    state["goal_encoder.weight"] = torch.rand(128, 2)  # foreign key, must be ignored
    path = tmp_path / "src.pt"
    torch.save(state, path)

    target = QModel(action_dim=8)
    fc1_before = target.fc1.weight.clone()
    loaded = target.load_conv_trunk(str(path))

    assert sorted(loaded) == [
        "conv1.bias", "conv1.weight",
        "conv2.bias", "conv2.weight",
        "conv3.bias", "conv3.weight",
    ]
    assert torch.equal(target.conv1.weight, source.conv1.weight)
    assert torch.equal(target.conv3.bias, source.conv3.bias)
    assert torch.equal(target.fc1.weight, fc1_before), "fc1 must not be touched"
