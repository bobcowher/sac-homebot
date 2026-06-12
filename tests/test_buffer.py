# tests/test_buffer.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
from buffer import ReplayBuffer


def _make_buf():
    return ReplayBuffer(
        max_size=100,
        input_shape=(3, 96, 96),
        input_device='cpu',
        output_device='cpu',
    )


def test_round_trip_shapes():
    buf = _make_buf()
    obs = torch.zeros(3, 96, 96, dtype=torch.uint8)
    for _ in range(20):
        buf.store_transition(obs, 3, 1.0, obs, False)
    states, actions, rewards, next_states, dones = buf.sample_buffer(10)
    assert states.shape == (10, 3, 96, 96)
    assert actions.shape == (10,)
    assert rewards.shape == (10,)
    assert next_states.shape == (10, 3, 96, 96)
    assert dones.shape == (10,)
    assert (actions == 3).all() and (rewards == 1.0).all()


def test_term_flag_round_trip():
    buf = _make_buf()
    obs = torch.zeros(3, 96, 96, dtype=torch.uint8)
    buf.store_transition(obs, 0, 1.0, obs, False)   # mid-episode pickup
    buf.store_transition(obs, 0, 1.0, obs, True)    # final pickup terminates
    assert not bool(buf.terminal_memory[0])
    assert bool(buf.terminal_memory[1])


def test_wraparound():
    buf = _make_buf()
    obs = torch.zeros(3, 96, 96, dtype=torch.uint8)
    for i in range(150):
        buf.store_transition(obs, i % 8, 0.0, obs, False)
    assert buf.mem_ctr == 150
    assert buf.can_sample(10)
