# tests/test_goal_buffer.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
from goal_buffer import GoalHERBuffer
from goal_geometry import GOAL_DIM


def _buf():
    return GoalHERBuffer(max_size=500, input_shape=(3, 96, 96), device="cpu",
                         action_dim=2, her_prob=1.0)


def _store_episode(buf, n, goal_px=(300.0, 300.0)):
    img = torch.zeros(3, 96, 96, dtype=torch.uint8)
    for t in range(n):
        rx, ry = float(t * 10), 0.0
        nrx, nry = float((t + 1) * 10), 0.0
        done = (t == n - 1)
        buf.store(img, [0.1, 0.0], img, rx, ry, 0.0, nrx, nry, 0.0, goal_px, done)


def test_sample_shapes_and_goal_dim():
    buf = _buf()
    _store_episode(buf, 30)
    img_s, goal_s, action, reward, img_ns, goal_ns, done = buf.sample(8, gamma=0.99)
    assert img_s.shape == (8, 3, 96, 96)
    assert goal_s.shape == (8, GOAL_DIM) and goal_ns.shape == (8, GOAL_DIM)
    assert action.shape == (8, 2)
    assert reward.shape == (8,) and done.shape == (8,)


def test_her_relabel_can_produce_success():
    buf = _buf()
    _store_episode(buf, 40)
    any_done = False
    for _ in range(20):
        *_, done = buf.sample(32, gamma=0.99)
        if done.any():
            any_done = True
            break
    assert any_done, "HER relabeling should yield reached (terminal) transitions"
