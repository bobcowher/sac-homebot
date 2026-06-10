# HER Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Hindsight Experience Replay (HER) with a goal-conditioned DQN on the `HomeBot2D-Goal-v1` environment.

**Architecture:** The Q-network gains a `goal_encoder` branch (Linear 2→128) whose output is concatenated with CNN flat features before the FC head. The replay buffer stores goals alongside transitions. `EpisodeBuffer.send_to` runs two passes — original transitions and K=4 hindsight-relabeled copies using the future strategy — before writing to the global buffer.

**Tech Stack:** PyTorch, Gymnasium, HomeBot2D-Goal-v1, pytest

---

## File Map

| File | What changes |
|------|-------------|
| `models/q_model.py` | Add `goal_dim`, `goal_encoder`, update `forward(obs, goal)` |
| `buffer.py` | Add `goal_memory`, `goal_dim` param, update `store_transition` / `sample_buffer` |
| `episode_buffer.py` | Rename `flush_to` → `send_to`, implement HER relabeling (K=4, future) |
| `agent.py` | Thread `goal` through `select_action`, `train_step`, `train` loop, `test` loop |
| `tests/test_q_model.py` | New — unit tests for goal-conditioned forward pass |
| `tests/test_buffer.py` | New — unit tests for goal storage and sampling |
| `tests/test_episode_buffer.py` | New — unit tests for `send_to` transition counts and relabeling |
| `tests/test_integration.py` | New — 3-episode smoke test confirming end-to-end shape correctness |

---

## Task 1: Goal-conditioned QModel

**Files:**
- Modify: `models/q_model.py`
- Create: `tests/test_q_model.py`

- [ ] **Step 1: Create the failing test**

```python
# tests/test_q_model.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
from models.q_model import QModel


def test_forward_returns_correct_shape():
    model = QModel(action_dim=8, input_shape=(3, 96, 96), goal_dim=2)
    obs  = torch.rand(4, 3, 96, 96)
    goal = torch.rand(4, 2)
    q    = model(obs, goal)
    assert q.shape == (4, 8), f"expected (4,8), got {q.shape}"


def test_forward_single_sample():
    model = QModel(action_dim=8, input_shape=(3, 96, 96), goal_dim=2)
    obs  = torch.rand(1, 3, 96, 96)
    goal = torch.rand(1, 2)
    q    = model(obs, goal)
    assert q.shape == (1, 8)
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
conda run -n sac-homebot pytest tests/test_q_model.py -v
```

Expected: `TypeError` — `forward()` takes 2 positional arguments but 3 were given.

- [ ] **Step 3: Update `models/q_model.py`**

Replace the entire file:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseModel


class QModel(BaseModel):
    def __init__(self, action_dim, input_shape=(3, 96, 96), goal_dim=2):
        super(QModel, self).__init__()

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

    def forward(self, obs, goal):
        x = self._conv_forward(obs)
        g = F.relu(self.goal_encoder(goal))
        x = torch.cat([x, g], dim=1)
        x = F.relu(self.fc1(x))
        return self.output(x)

    def _weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
conda run -n sac-homebot pytest tests/test_q_model.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add models/q_model.py tests/test_q_model.py
git commit -m "Add goal-conditioned QModel with goal_encoder branch"
```

---

## Task 2: Goal-aware ReplayBuffer

**Files:**
- Modify: `buffer.py`
- Create: `tests/test_buffer.py`

- [ ] **Step 1: Create the failing test**

```python
# tests/test_buffer.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
import numpy as np
from buffer import ReplayBuffer


def _make_buf():
    return ReplayBuffer(
        max_size=100,
        input_shape=(3, 96, 96),
        input_device='cpu',
        output_device='cpu',
    )


def _dummy_transition(buf, goal):
    obs = torch.zeros(3, 96, 96, dtype=torch.uint8)
    buf.store_transition(obs, 3, 1.0, obs, False, goal)


def test_goals_stored_and_sampled():
    buf  = _make_buf()
    goal = np.array([100.0, 200.0], dtype=np.float32)
    for _ in range(20):
        _dummy_transition(buf, goal)
    _, _, _, _, _, goals = buf.sample_buffer(10)
    assert goals.shape == (10, 2), f"expected (10,2), got {goals.shape}"
    assert torch.allclose(goals, torch.tensor([100.0, 200.0]).expand(10, 2))


def test_different_goals_round_trip():
    buf   = _make_buf()
    goal_a = np.array([10.0, 20.0], dtype=np.float32)
    goal_b = np.array([30.0, 40.0], dtype=np.float32)
    obs = torch.zeros(3, 96, 96, dtype=torch.uint8)
    for _ in range(10):
        buf.store_transition(obs, 0, 0.0, obs, False, goal_a)
    for _ in range(10):
        buf.store_transition(obs, 0, 0.0, obs, False, goal_b)
    _, _, _, _, _, goals = buf.sample_buffer(20)
    assert goals.shape == (20, 2)
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
conda run -n sac-homebot pytest tests/test_buffer.py -v
```

Expected: `TypeError` — `store_transition()` missing `goal` argument or unexpected keyword.

- [ ] **Step 3: Update `buffer.py`**

Replace the entire file:

```python
import torch
import os


class ReplayBuffer:
    def __init__(self, max_size, input_shape, input_device, output_device='cpu', goal_dim=2):
        self.mem_size = max_size
        self.mem_ctr  = 0

        override = os.getenv("REPLAY_BUFFER_MEMORY")
        if override in ["cpu", "cuda:0", "cuda:1"]:
            print("Received replay buffer memory override.")
            self.input_device = override
        else:
            self.input_device = input_device

        print(f"Replay buffer memory on: {self.input_device}")
        self.output_device = output_device

        self.state_memory      = torch.zeros((max_size, *input_shape), dtype=torch.uint8,   device=self.input_device)
        self.next_state_memory = torch.zeros((max_size, *input_shape), dtype=torch.uint8,   device=self.input_device)
        self.action_memory     = torch.zeros(max_size,                 dtype=torch.int64,   device=self.input_device)
        self.reward_memory     = torch.zeros(max_size,                 dtype=torch.float32, device=self.input_device)
        self.terminal_memory   = torch.zeros(max_size,                 dtype=torch.bool,    device=self.input_device)
        self.goal_memory       = torch.zeros((max_size, goal_dim),     dtype=torch.float32, device=self.input_device)

    def can_sample(self, batch_size: int) -> bool:
        return self.mem_ctr >= batch_size * 10

    def store_transition(self, state, action, reward, next_state, done, goal):
        idx = self.mem_ctr % self.mem_size
        self.state_memory[idx]      = torch.as_tensor(state,      dtype=torch.uint8,   device=self.input_device)
        self.next_state_memory[idx] = torch.as_tensor(next_state, dtype=torch.uint8,   device=self.input_device)
        self.action_memory[idx]     = int(action)
        self.reward_memory[idx]     = float(reward)
        self.terminal_memory[idx]   = bool(done)
        self.goal_memory[idx]       = torch.as_tensor(goal, dtype=torch.float32, device=self.input_device)
        self.mem_ctr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_ctr, self.mem_size)
        batch   = torch.randint(0, max_mem, (batch_size,), device=self.input_device, dtype=torch.int64)

        states      = self.state_memory[batch].to(self.output_device,      dtype=torch.float32)
        next_states = self.next_state_memory[batch].to(self.output_device, dtype=torch.float32)
        actions     = self.action_memory[batch].to(self.output_device)
        rewards     = self.reward_memory[batch].to(self.output_device)
        dones       = self.terminal_memory[batch].to(self.output_device)
        goals       = self.goal_memory[batch].to(self.output_device)

        return states, actions, rewards, next_states, dones, goals
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
conda run -n sac-homebot pytest tests/test_buffer.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add buffer.py tests/test_buffer.py
git commit -m "Add goal storage to ReplayBuffer"
```

---

## Task 3: HER relabeling in EpisodeBuffer (`send_to`)

**Files:**
- Modify: `episode_buffer.py`
- Create: `tests/test_episode_buffer.py`

- [ ] **Step 1: Create the failing test**

```python
# tests/test_episode_buffer.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
import numpy as np
from episode_buffer import EpisodeBuffer
from buffer import ReplayBuffer


def _make_replay():
    return ReplayBuffer(
        max_size=10000,
        input_shape=(3, 96, 96),
        input_device='cpu',
        output_device='cpu',
    )


def _dummy_compute_reward(ag, dg, info):
    return np.zeros(len(ag), dtype=np.float32)


def test_send_to_transition_count():
    """10-step episode with K=4 future strategy.

    Original: 10 transitions.
    Hindsight per step: min(4, len(future)) — step 9 has no future, skipped.
      steps 0-5: 4 each  =  24
      step 6: 3, step 7: 2, step 8: 1, step 9: 0  = 6
    Total: 10 + 30 = 40
    """
    ep  = EpisodeBuffer()
    rep = _make_replay()
    obs = torch.zeros(3, 96, 96, dtype=torch.uint8)
    desired_goal = np.array([300.0, 400.0], dtype=np.float32)

    for i in range(10):
        ep.store(obs, 0, 0.0, obs, False,
                 achieved_goal=np.array([float(i * 10), float(i * 10)], dtype=np.float32))

    ep.send_to(rep, desired_goal, _dummy_compute_reward)
    assert rep.mem_ctr == 40, f"expected 40, got {rep.mem_ctr}"


def test_send_to_clears_nothing_on_its_own():
    ep  = EpisodeBuffer()
    rep = _make_replay()
    obs = torch.zeros(3, 96, 96, dtype=torch.uint8)
    ag  = np.array([0.0, 0.0], dtype=np.float32)
    dg  = np.array([100.0, 100.0], dtype=np.float32)

    ep.store(obs, 0, 0.0, obs, False, achieved_goal=ag)
    ep.send_to(rep, dg, _dummy_compute_reward)
    assert len(ep) == 1, "send_to must not clear the buffer — caller does that"


def test_send_to_original_reward_preserved():
    ep  = EpisodeBuffer()
    rep = _make_replay()
    obs = torch.zeros(3, 96, 96, dtype=torch.uint8)
    dg  = np.array([100.0, 100.0], dtype=np.float32)
    ag  = np.array([0.0, 0.0], dtype=np.float32)

    ep.store(obs, 0, 7.0, obs, False, achieved_goal=ag)
    ep.send_to(rep, dg, _dummy_compute_reward)

    # First stored transition is the original — reward must be 7.0
    assert float(rep.reward_memory[0]) == 7.0
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
conda run -n sac-homebot pytest tests/test_episode_buffer.py -v
```

Expected: `AttributeError: 'EpisodeBuffer' object has no attribute 'send_to'`

- [ ] **Step 3: Update `episode_buffer.py`**

Replace the entire file:

```python
from dataclasses import dataclass
from typing import Callable
import random
import numpy as np
import torch


@dataclass
class Transition:
    obs:           torch.Tensor
    action:        int
    reward:        float
    next_obs:      torch.Tensor
    done:          bool
    achieved_goal: np.ndarray  # robot pixel (x, y) at this step


class EpisodeBuffer:
    """Caches one episode's transitions for HER relabeling.

    Usage:
        # each step:
        episode_buffer.store(obs, action, reward, next_obs, done, achieved_goal)

        # end of episode:
        episode_buffer.send_to(replay_buffer, desired_goal, compute_reward)
        episode_buffer.clear()
    """

    K = 4  # hindsight goals per transition (future strategy)

    def __init__(self):
        self._transitions: list[Transition] = []

    def store(self, obs, action, reward, next_obs, done, achieved_goal):
        self._transitions.append(Transition(
            obs=obs,
            action=action,
            reward=float(reward),
            next_obs=next_obs,
            done=bool(done),
            achieved_goal=achieved_goal,
        ))

    def __len__(self):
        return len(self._transitions)

    def clear(self):
        self._transitions.clear()

    def send_to(
        self,
        replay_buffer,
        desired_goal: np.ndarray,
        compute_reward: Callable,
    ) -> None:
        """Write original transitions then K hindsight-relabeled copies to replay_buffer.

        Strategy: future — hindsight goals are sampled from achieved_goals strictly
        after the current step. Last step is skipped (no future states).
        """
        # Pass 1: original transitions (env reward, episode desired_goal)
        for t in self._transitions:
            replay_buffer.store_transition(
                t.obs, t.action, t.reward, t.next_obs, t.done, desired_goal
            )

        # Pass 2: hindsight transitions
        for i, t in enumerate(self._transitions):
            future = self._transitions[i + 1:]
            if not future:
                continue
            k = min(self.K, len(future))
            for hg_t in random.sample(future, k):
                hindsight_goal   = hg_t.achieved_goal
                hindsight_reward = float(compute_reward(
                    t.achieved_goal[np.newaxis],
                    hindsight_goal[np.newaxis],
                    {},
                ))
                replay_buffer.store_transition(
                    t.obs, t.action, hindsight_reward, t.next_obs, t.done, hindsight_goal
                )
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
conda run -n sac-homebot pytest tests/test_episode_buffer.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add episode_buffer.py tests/test_episode_buffer.py
git commit -m "Implement HER relabeling in EpisodeBuffer.send_to (K=4, future strategy)"
```

---

## Task 4: Thread goal through Agent

**Files:**
- Modify: `agent.py`

There's no isolated unit test here — the agent is wired to the full env. Integration test is Task 5.

- [ ] **Step 1: Update `select_action`**

Replace lines 65–70:

```python
    def select_action(self, obs, goal):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            obs_t  = obs.unsqueeze(0).float().to(self.device) / 255.0
            goal_t = torch.as_tensor(goal, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.q_model(obs_t, goal_t).argmax(dim=1).item()
```

- [ ] **Step 2: Update `train_step`**

Replace lines 72–101:

```python
    def train_step(self, batch_size):
        obs, actions, rewards, next_obs, dones, goals = self.memory.sample_buffer(batch_size)

        obs      = obs      / 255.0
        next_obs = next_obs / 255.0

        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones   = dones.unsqueeze(1).float()

        q_sa = self.q_model(obs, goals).gather(1, actions)

        with torch.no_grad():
            next_actions = self.q_model(next_obs, goals).argmax(dim=1, keepdim=True)
            next_q       = self.target_q_model(next_obs, goals).gather(1, next_actions)
            targets      = rewards + (1 - dones) * self.gamma * next_q

        loss = F.mse_loss(q_sa, targets)

        self.q_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_model.parameters(), max_norm=1.0)
        self.q_optimizer.step()

        if self.total_steps % self.target_update_interval == 0:
            self.target_q_model.load_state_dict(self.q_model.state_dict())

        self.total_steps += 1
        return loss.item()
```

- [ ] **Step 3: Update the `train` episode loop**

Replace lines 137–164 (from `for episode in range(episodes):` through `self.epsilon = max(...)`):

```python
        for episode in range(episodes):
            raw_obs, _ = self.env.reset()
            obs          = self.process_observation(raw_obs["observation"])
            desired_goal = raw_obs["desired_goal"]

            done = False
            episode_reward = 0.0
            episode_loss   = 0.0
            episode_steps  = 0

            while not done:
                action = self.select_action(obs, desired_goal)
                raw_next, reward, term, trunc, _ = self.env.step(action)
                next_obs = self.process_observation(raw_next["observation"])
                done = term or trunc

                self.episode_buffer.store(
                    obs, action, reward, next_obs, done,
                    achieved_goal=raw_next["achieved_goal"],
                )
                episode_reward += float(reward)
                episode_steps  += 1
                obs = next_obs

            self.episode_buffer.send_to(
                self.memory,
                desired_goal=desired_goal,
                compute_reward=self.env.unwrapped.compute_reward,  # type: ignore[attr-defined]
            )
            self.episode_buffer.clear()

            for _ in range(episode_steps):
                if self.memory.can_sample(batch_size):
                    episode_loss += self.train_step(batch_size)

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
```

- [ ] **Step 4: Update the `test` loop**

Replace lines 183–208 (full `test` method):

```python
    def test(self, episodes=10):
        self.q_model.eval()
        total_rewards = []

        for episode in range(episodes):
            raw_obs, _ = self.env.reset()
            obs          = self.process_observation(raw_obs["observation"])
            desired_goal = raw_obs["desired_goal"]
            done = False
            episode_reward = 0.0

            while not done:
                with torch.no_grad():
                    obs_t  = obs.unsqueeze(0).float().to(self.device) / 255.0
                    goal_t = torch.as_tensor(desired_goal, dtype=torch.float32,
                                             device=self.device).unsqueeze(0)
                    action = self.q_model(obs_t, goal_t).argmax(dim=1).item()
                raw_next, reward, term, trunc, _ = self.env.step(action)
                next_obs = self.process_observation(raw_next["observation"])
                done = term or trunc
                episode_reward += float(reward)
                obs = next_obs

            total_rewards.append(episode_reward)
            print(f"Test episode {episode} | reward: {episode_reward:.1f}")

        avg = sum(total_rewards) / len(total_rewards)
        print(f"Average reward over {episodes} episodes: {avg:.1f}")
        self.q_model.train()
        return total_rewards
```

- [ ] **Step 5: Commit**

```bash
git add agent.py
git commit -m "Thread goal through Agent: select_action, train_step, train loop, test loop"
```

---

## Task 5: Integration smoke test + push

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Create the integration test**

```python
# tests/test_integration.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import gymnasium as gym
import homebot  # noqa: F401
from agent import Agent


def test_three_episode_train_runs_without_error():
    env = gym.make(
        "HomeBot2D-Goal-v1",
        render_mode="rgb_array",
        action_mode="discrete",
        obs_resolution=(96, 96),
        n_trash=2,
        max_steps=50,
        map_name="default",
        goals=["collect_trash"],
    )
    agent = Agent(env=env, max_buffer_size=10000)
    agent.train(episodes=3, batch_size=64)
    env.close()
```

- [ ] **Step 2: Run the integration test**

```bash
conda run -n sac-homebot pytest tests/test_integration.py -v
```

Expected: 1 passed (takes ~10–20 seconds).

- [ ] **Step 3: Run the full test suite**

```bash
conda run -n sac-homebot pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit and push**

```bash
git add tests/test_integration.py
git commit -m "Add integration smoke test for HER end-to-end training"
git push origin her
```
