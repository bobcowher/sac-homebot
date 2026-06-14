# tests/test_episode_buffer.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import math
import torch
import numpy as np
from episode_buffer import EpisodeBuffer
from buffer import ReplayBuffer
from goal_geometry import bearing as compute_bearing


def _make_replay():
    return ReplayBuffer(
        max_size=10000,
        input_shape=(3, 96, 96),
        input_device='cpu',
        output_device='cpu',
    )


def _dummy_compute_reward(ag, dg, info):
    return np.zeros(len(ag), dtype=np.float32)


def _pos(x, y):
    return np.array([float(x), float(y)], dtype=np.float32)


def _store_walk(ep, obs, n, heading=0.0):
    """n-step straight-line walk: step i moves (i*10, i*10) -> ((i+1)*10, (i+1)*10).
    Heading is constant for simplicity.
    """
    for i in range(n):
        ep.store(obs, 0, 0.0, obs, False,
                 achieved_prev=_pos(i * 10, i * 10),
                 achieved_next=_pos((i + 1) * 10, (i + 1) * 10),
                 heading_prev=heading,
                 heading_next=heading)


def test_send_to_transition_count():
    """10-step episode, future strategy.

    Original: 10 transitions.
    Hindsight per step i: min(K, steps remaining after i) — last step has no
    future, skipped. Expected count is derived from EpisodeBuffer.K so the
    test tracks K tuning.
    """
    n = 10
    expected = n + sum(min(EpisodeBuffer.K, n - 1 - i) for i in range(n))

    ep  = EpisodeBuffer()
    rep = _make_replay()
    obs = torch.zeros(3, 96, 96, dtype=torch.uint8)

    _store_walk(ep, obs, n)
    ep.send_to(rep, _pos(300, 400), _dummy_compute_reward)
    assert rep.mem_ctr == expected, f"expected {expected}, got {rep.mem_ctr}"


def test_send_to_clears_nothing_on_its_own():
    ep  = EpisodeBuffer()
    rep = _make_replay()
    obs = torch.zeros(3, 96, 96, dtype=torch.uint8)

    _store_walk(ep, obs, 1)
    ep.send_to(rep, _pos(100, 100), _dummy_compute_reward)
    assert len(ep) == 1, "send_to must not clear the buffer — caller does that"


def test_bearing_goals_are_unit_vectors():
    """Stored goals must be 2-D unit-circle vectors (bearings), not raw displacements."""
    ep  = EpisodeBuffer()
    rep = _make_replay()
    obs = torch.zeros(3, 96, 96, dtype=torch.uint8)
    desired = _pos(300, 400)

    n = 5
    _store_walk(ep, obs, n, heading=0.5)
    ep.send_to(rep, desired, _dummy_compute_reward)

    for i in range(n):
        goal_vec      = rep.goal_memory[i].numpy()
        next_goal_vec = rep.next_goal_memory[i].numpy()
        # Each must lie on the unit circle
        norm_g  = math.hypot(float(goal_vec[0]), float(goal_vec[1]))
        norm_ng = math.hypot(float(next_goal_vec[0]), float(next_goal_vec[1]))
        assert abs(norm_g  - 1.0) < 1e-5, f"transition {i}: goal not on unit circle, norm={norm_g}"
        assert abs(norm_ng - 1.0) < 1e-5, f"transition {i}: next_goal not on unit circle, norm={norm_ng}"


def test_bearing_goals_match_geometry():
    """Stored goals must exactly match compute_bearing() for the robot's pose."""
    ep  = EpisodeBuffer()
    rep = _make_replay()
    obs = torch.zeros(3, 96, 96, dtype=torch.uint8)
    desired = _pos(300.0, 400.0)
    heading = 0.7

    n = 3
    _store_walk(ep, obs, n, heading=heading)
    ep.send_to(rep, desired, _dummy_compute_reward)

    # Pass 1 originals are the first n stored transitions.
    for i in range(n):
        expected_goal      = compute_bearing(i * 10.0, i * 10.0, heading,
                                             desired[0], desired[1])
        expected_next_goal = compute_bearing((i + 1) * 10.0, (i + 1) * 10.0, heading,
                                             desired[0], desired[1])
        stored_goal      = rep.goal_memory[i].numpy()
        stored_next_goal = rep.next_goal_memory[i].numpy()
        assert np.allclose(stored_goal, expected_goal, atol=1e-6), \
            f"transition {i}: goal mismatch: {stored_goal} vs {expected_goal}"
        assert np.allclose(stored_next_goal, expected_next_goal, atol=1e-6), \
            f"transition {i}: next_goal mismatch: {stored_next_goal} vs {expected_next_goal}"


def test_hindsight_bearing_goals_are_unit_vectors():
    """Hindsight goals are future achieved_next positions; stored bearings must lie on unit circle."""
    ep  = EpisodeBuffer()
    rep = _make_replay()
    obs = torch.zeros(3, 96, 96, dtype=torch.uint8)

    _store_walk(ep, obs, 2)  # step 0: (0,0)->(10,10), step 1: (10,10)->(20,20)
    ep.send_to(rep, _pos(300, 400), _dummy_compute_reward)

    # Layout: 2 originals, then step 0's hindsight (step 1 has no future).
    assert rep.mem_ctr == 3

    # Hindsight goal is step 1's achieved_next = (20, 20)
    # From step 0's perspective: prev=(0,0), next=(10,10), heading=0.0
    hs_goal  = rep.goal_memory[2].numpy()
    hs_ngoal = rep.next_goal_memory[2].numpy()

    expected_hs_goal  = compute_bearing(0.0, 0.0,  0.0, 20.0, 20.0)
    expected_hs_ngoal = compute_bearing(10.0, 10.0, 0.0, 20.0, 20.0)

    assert np.allclose(hs_goal,  expected_hs_goal,  atol=1e-6), \
        f"hindsight goal mismatch: {hs_goal} vs {expected_hs_goal}"
    assert np.allclose(hs_ngoal, expected_hs_ngoal, atol=1e-6), \
        f"hindsight next_goal mismatch: {hs_ngoal} vs {expected_hs_ngoal}"


def test_hindsight_success_is_terminal():
    """Relabeled success (reward 1) must store done=True; reward 0 stores done=False.

    The env terminates on success, so hindsight transitions must match — otherwise
    targets bootstrap past the goal and inflate Q in hindsight data.
    """
    def _success_compute_reward(ag, dg, info):
        return np.ones(len(ag), dtype=np.float32)

    ep  = EpisodeBuffer()
    rep = _make_replay()
    obs = torch.zeros(3, 96, 96, dtype=torch.uint8)

    n = 3
    _store_walk(ep, obs, n)
    ep.send_to(rep, _pos(100, 100), _success_compute_reward)
    # Layout: 3 original (reward 0 via stored value, done False) then hindsight.
    # All hindsight rewards are 1.0 here -> all hindsight dones must be True.
    cnt = rep.mem_ctr
    assert cnt > n, "expected hindsight transitions beyond the originals"
    assert not rep.terminal_memory[:n].any(), "original transitions must keep done=False"
    assert rep.terminal_memory[n:cnt].all(), "hindsight successes must be terminal"

    # And reward-0 relabels stay non-terminal.
    ep2  = EpisodeBuffer()
    rep2 = _make_replay()
    _store_walk(ep2, obs, n)
    ep2.send_to(rep2, _pos(100, 100), _dummy_compute_reward)
    assert not rep2.terminal_memory[:rep2.mem_ctr].any(), "reward-0 relabels must stay done=False"


def test_send_to_original_reward_preserved():
    ep  = EpisodeBuffer()
    rep = _make_replay()
    obs = torch.zeros(3, 96, 96, dtype=torch.uint8)

    ep.store(obs, 0, 7.0, obs, False,
             achieved_prev=_pos(0, 0), achieved_next=_pos(10, 10),
             heading_prev=0.0, heading_next=0.0)
    ep.send_to(rep, _pos(100, 100), _dummy_compute_reward)

    # First stored transition is the original — reward must be 7.0
    assert float(rep.reward_memory[0]) == 7.0


def test_heading_defaults():
    """heading_prev/heading_next default to 0.0 if omitted (backward compat)."""
    ep  = EpisodeBuffer()
    rep = _make_replay()
    obs = torch.zeros(3, 96, 96, dtype=torch.uint8)

    # Call store without heading args — should not raise
    ep.store(obs, 0, 0.0, obs, False,
             achieved_prev=_pos(0, 0), achieved_next=_pos(10, 10))
    ep.send_to(rep, _pos(100, 100), _dummy_compute_reward)
    assert rep.mem_ctr >= 1
