# Bearing-Conditioned DQN+HER Reacher (Tier C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A discrete Double-DQN + HER reacher that drives toward a **bearing** to a (perceivable) target, the Tier-C primitive of the A/B/C hierarchy contract.

**Architecture:** Reuse the goal/HER/env plumbing from `goal-reacher` (`goal_geometry`, `goal_manager`, `goal_buffer`). Replace the SAC actor/critic with a goal-conditioned discrete **Double-DQN** Q-network (8-direction actions). The policy is conditioned on a **bearing** `[sin θ, cos θ]` (not precise range), produced by a visibility-gated `bearing_to` stub (the honest stand-in for a real detector) with noise for robustness. Random robot spawn each episode (our-side teleport, no homebot change). HER relabels the target *position* and recomputes the bearing. Greedy-eval-first, with a step budget that makes circling un-gameable.

**Tech Stack:** Python, PyTorch, Gymnasium, local `homebot` (`HomeBot2D-v1`, `action_mode="discrete"`), conda env `sac-homebot`, Beekeeper project `Q-Homebot`.

**Spec:** `docs/superpowers/specs/2026-06-13-reacher-hierarchy-contract-design.md`.
**Guiding rule:** no env hack a real planning agent couldn't do (random spawn ✓, visibility-gated bearing ✓, exact out-of-view range ✗, HER/curriculum ✓).

---

## Conventions

- Tests: `conda run -n sac-homebot python3 -m pytest <path> -v`.
- Env-id fallback: `"HomeBot2D-v1"` then `"HomeBot2D-V1"`; `action_mode="discrete"` (8 dirs).
- Constants in `goal_geometry.py`: `BEARING_DIM = 2`, `GOAL_RADIUS = 40.0`, `PROGRESS_SCALE = 0.1`, `SUCCESS_REWARD = 5.0`, `ROBOT_STEP_PX = 4.0` (homebot `DISCRETE_SPEED`), `EVAL_BUDGET_MULT = 3`.
- Robot pose: `base._robot.x/.y/.angle`; map: `base._map.tile_to_pixel`, `valid_floor_tiles`, `wall_solid`, `tile_size`, `fixtures`.

---

## File Structure

**Reuse as-is from `goal-reacher`:** `models/base.py`.

**Modify (from `goal-reacher`):**
- `goal_geometry.py` — add `bearing_vec()` (bearing-only), `eval_step_budget()`; keep `distance`, `reach_reward`.
- `goal_manager.py` — add `random_spawn()`, `bearing_to(target)` (visibility-gated stub), `add_noise()`; keep waypoint sampling.
- `goal_buffer.py` — store **discrete int actions**; return **bearing** goals (recompute on HER relabel); HER **on** (`her_prob=0.8`).

**New:**
- `models/q_model.py` — goal-conditioned discrete Double-DQN Q-network.
- `agent_dqn.py` — the DQN+HER agent (collection, Double-DQN update, random spawn, un-gameable greedy eval).
- `train.py` — entry point (discrete env).

**Remove on this branch** (SAC leftovers, not used): `agent_reacher.py`, `models/goal_actor.py`, `models/goal_critic.py`, `models/encoder.py`, `models/ssim_loss.py` — delete to avoid confusion (they live on `reacher-encoder`).

**Tests:** `tests/test_bearing_geometry.py`, `tests/test_goal_manager_dqn.py`, `tests/test_goal_buffer_dqn.py`, `tests/test_q_model.py`, `tests/test_dqn_integration.py`.

---

## Task 1: Branch + bearing-primary geometry

**Files:** create branch `bearing-reacher`; modify `goal_geometry.py`; test `tests/test_bearing_geometry.py`.

- [ ] **Step 1: Branch and clean SAC leftovers**

```bash
cd /home/robertcowher/pythonprojects/sac-homebot
git checkout goal-reacher
git checkout -b bearing-reacher
git rm agent_reacher.py models/goal_actor.py models/goal_critic.py models/encoder.py models/ssim_loss.py 2>/dev/null || true
git rm tests/test_goal_models.py tests/test_reacher_integration.py probe_goal.py 2>/dev/null || true
git commit -m "bearing-reacher: drop SAC-AE leftovers (kept on reacher-encoder)"
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_bearing_geometry.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import math, numpy as np
from goal_geometry import bearing_vec, eval_step_budget, BEARING_DIM


def test_bearing_vec_straight_ahead():
    v = bearing_vec(0, 0, 0.0, 100, 0)        # goal +x, heading 0
    assert v.shape == (BEARING_DIM,)
    assert abs(v[0] - 0.0) < 1e-6 and abs(v[1] - 1.0) < 1e-6   # sin0, cos0


def test_bearing_vec_relative_to_heading():
    v = bearing_vec(0, 0, math.pi / 2, 100, 0)  # goal +x, heading +90 -> bearing -90
    assert abs(v[0] - (-1.0)) < 1e-6 and abs(v[1] - 0.0) < 1e-6


def test_bearing_vec_has_no_range():
    near = bearing_vec(0, 0, 0.0, 50, 0)
    far  = bearing_vec(0, 0, 0.0, 500, 0)
    assert np.allclose(near, far)   # bearing-only: distance must not appear


def test_eval_budget_scales_with_distance():
    # budget = EVAL_BUDGET_MULT * ceil(dist / ROBOT_STEP_PX)
    assert eval_step_budget(40.0) < eval_step_budget(400.0)
    assert eval_step_budget(400.0) == 3 * math.ceil(400.0 / 4.0)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `conda run -n sac-homebot python3 -m pytest tests/test_bearing_geometry.py -v`
Expected: FAIL — `bearing_vec` / `eval_step_budget` not defined.

- [ ] **Step 4: Edit `goal_geometry.py`**

Add constants `BEARING_DIM = 2`, `ROBOT_STEP_PX = 4.0`, `EVAL_BUDGET_MULT = 3` (alongside existing `GOAL_RADIUS`, `PROGRESS_SCALE`, `SUCCESS_REWARD`). Add:

```python
def bearing_vec(rx, ry, rtheta, gx, gy):
    """Bearing-only egocentric goal: [sin(bearing), cos(bearing)]. NO range
    (deploy-honest: a detector gives direction cheaply, precise range does not)."""
    bearing = math.atan2(gy - ry, gx - rx) - rtheta
    return np.array([math.sin(bearing), math.cos(bearing)], dtype=np.float32)


def eval_step_budget(init_dist):
    """Step budget for an eval episode: a multiple of the straight-line steps.
    Tight enough that a circling/sweeping policy cannot 'pass' by luck."""
    return EVAL_BUDGET_MULT * math.ceil(max(init_dist, GOAL_RADIUS) / ROBOT_STEP_PX)
```

Keep `distance` and `reach_reward` as-is (reward is a training-time signal computed from positions — allowed).

- [ ] **Step 5: Run test to verify it passes**

Run: `conda run -n sac-homebot python3 -m pytest tests/test_bearing_geometry.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add goal_geometry.py tests/test_bearing_geometry.py
git commit -m "bearing-reacher: bearing-only goal vector + un-gameable eval step budget"
```

---

## Task 2: GoalManager — random spawn, bearing_to stub, noise

**Files:** modify `goal_manager.py`; test `tests/test_goal_manager_dqn.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_goal_manager_dqn.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np, gymnasium as gym, homebot  # noqa: F401
from goal_manager import GoalManager
from goal_geometry import BEARING_DIM, distance, GOAL_RADIUS


def _env():
    for env_id in ("HomeBot2D-v1", "HomeBot2D-V1"):
        try:
            return gym.make(env_id, render_mode="rgb_array", action_mode="discrete",
                            obs_resolution=(96, 96), n_trash=2, max_steps=100,
                            map_name="default", goals=["trash"])
        except gym.error.Error:
            continue
    raise RuntimeError("no env")


def test_random_spawn_varies_and_is_valid():
    env = _env(); base = env.unwrapped
    gm = GoalManager(radius_px=200.0, rng=np.random.default_rng(0))
    env.reset(); gm.random_spawn(base); p1 = (base._robot.x, base._robot.y)
    env.reset(); gm.random_spawn(base); p2 = (base._robot.x, base._robot.y)
    assert p1 != p2                                   # randomized
    # spawned on a valid (non-wall) tile
    ts = base._map.tile_size
    assert not base._map.wall_solid[int(p1[1] // ts), int(p1[0] // ts)]
    env.close()


def test_bearing_to_returns_bearing_or_none():
    env = _env(); base = env.unwrapped
    env.reset()
    gm = GoalManager(radius_px=200.0, rng=np.random.default_rng(0))
    b = gm.bearing_to(base, "fridge")
    assert b is None or b.shape == (BEARING_DIM,)
    env.close()


def test_noise_dropout_returns_none_sometimes():
    gm = GoalManager(radius_px=200.0, rng=np.random.default_rng(1))
    v = np.array([0.0, 1.0], np.float32)
    outs = [gm.add_noise(v, dropout=1.0) for _ in range(5)]   # dropout=1 -> always None
    assert all(o is None for o in outs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n sac-homebot python3 -m pytest tests/test_goal_manager_dqn.py -v`
Expected: FAIL — `random_spawn` / `bearing_to` / `add_noise` not defined.

- [ ] **Step 3: Edit `goal_manager.py`**

Add imports `from goal_geometry import bearing_vec, distance, GOAL_RADIUS` (keep existing). Add methods to `GoalManager`:

```python
    def random_spawn(self, base):
        """Teleport the robot to a random valid floor tile + random heading.
        Our-side (no homebot change); a real robot starts anywhere."""
        tiles = base._map.valid_floor_tiles()
        tx, ty = tiles[int(self.rng.integers(len(tiles)))]
        base._robot.x, base._robot.y = base._map.tile_to_pixel(tx, ty)
        base._robot.angle = float(self.rng.uniform(-np.pi, np.pi))

    def bearing_to(self, base, target):
        """Visibility-gated bearing to a named fixture: ground-truth position,
        gated by in-view + line-of-sight (stand-in for a real detector).
        Returns [sin,cos] or None if not perceivable."""
        gx, gy = base._map.tile_to_pixel(*base._map.fixtures[target])
        if not self._in_view(base, gx, gy):
            return None
        if not self._line_of_sight(base, gx, gy):
            return None
        return bearing_vec(base._robot.x, base._robot.y, base._robot.angle, gx, gy)

    def add_noise(self, bearing, dropout=0.1, jitter_rad=0.1):
        """Degrade a bearing like a real detector: random dropout + angular jitter."""
        if bearing is None or self.rng.random() < dropout:
            return None
        ang = np.arctan2(bearing[0], bearing[1]) + self.rng.normal(0, jitter_rad)
        return np.array([np.sin(ang), np.cos(ang)], np.float32)

    def _in_view(self, base, gx, gy):
        r, rend = base._robot, base._renderer
        vw, vh = rend._viewport_w, rend._viewport_h
        mw, mh = base._map.pixel_width, base._map.pixel_height
        vx = max(0, min(int(r.x - vw / 2), mw - vw))
        vy = max(0, min(int(r.y - vh / 2), mh - vh))
        return vx <= gx <= vx + vw and vy <= gy <= vy + vh

    def _line_of_sight(self, base, gx, gy):
        rx, ry = base._robot.x, base._robot.y
        solid, ts = base._map.wall_solid, base._map.tile_size
        steps = max(2, int(distance(rx, ry, gx, gy) / (ts / 2)))
        for i in range(steps + 1):
            f = i / steps
            x, y = rx + (gx - rx) * f, ry + (gy - ry) * f
            col, row = int(x // ts), int(y // ts)
            if 0 <= row < solid.shape[0] and 0 <= col < solid.shape[1] and solid[row, col]:
                return False
        return True
```

Ensure `import numpy as np` is present.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n sac-homebot python3 -m pytest tests/test_goal_manager_dqn.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add goal_manager.py tests/test_goal_manager_dqn.py
git commit -m "bearing-reacher: random spawn + visibility-gated bearing_to stub + detector noise"
```

---

## Task 3: goal_buffer — discrete actions + bearing-primary HER

**Files:** modify `goal_buffer.py`; test `tests/test_goal_buffer_dqn.py`.

The existing `GoalHERBuffer` stores float actions and recomputes a polar `[range,sin,cos]` goal. Change: store **discrete int** actions, and recompute a **bearing** `[sin,cos]` goal (drop range). HER stays (future relabel of the target position, recompute bearing).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_goal_buffer_dqn.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
from goal_buffer import GoalHERBuffer
from goal_geometry import BEARING_DIM


def _buf():
    return GoalHERBuffer(max_size=500, input_shape=(3, 96, 96), device="cpu",
                         her_prob=1.0)


def _ep(buf, n, goal=(300.0, 300.0)):
    img = torch.zeros(3, 96, 96, dtype=torch.uint8)
    for t in range(n):
        rx, ry = float(t * 10), 0.0
        nrx, nry = float((t + 1) * 10), 0.0
        buf.store(img, 3, img, rx, ry, 0.0, nrx, nry, 0.0, goal, t == n - 1)  # action=3 (int)


def test_sample_returns_int_actions_and_bearing_goals():
    buf = _buf(); _ep(buf, 30)
    img_s, goal_s, action, reward, img_ns, goal_ns, done = buf.sample(8, gamma=0.99)
    assert goal_s.shape == (8, BEARING_DIM) and goal_ns.shape == (8, BEARING_DIM)
    assert action.dtype == torch.int64 and action.shape == (8,)
    assert reward.shape == (8,) and done.shape == (8,)


def test_her_relabel_yields_reached():
    buf = _buf(); _ep(buf, 40)
    assert any(buf.sample(32, 0.99)[-1].any() for _ in range(20))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n sac-homebot python3 -m pytest tests/test_goal_buffer_dqn.py -v`
Expected: FAIL — `store` signature (int action) / goal shape (2 vs 3) mismatch.

- [ ] **Step 3: Edit `goal_buffer.py`**

Change the action storage to int64 and the goal recompute to bearing-only:
- In `__init__`: `self.action = torch.zeros(max_size, dtype=torch.int64, device=device)` (was float `(max_size, action_dim)`); drop the `action_dim` arg (default ignore).
- In `store(self, img_s, action, img_ns, rx, ry, rth, nrx, nry, nrth, goal_px, done)`: `self.action[i] = int(action)`.
- In `sample`, replace the polar block (`range/NORM, sin, cos`) with bearing-only:

```python
        def bearing(px, py, th):
            ang = np.arctan2(gy - py, gx - px) - th
            return np.stack([np.sin(ang), np.cos(ang)], axis=1).astype(np.float32)

        goal_s = bearing(rx, ry, rth)
        goal_ns = bearing(nrx, nry, nrth)
```

Keep the geometry storage, HER future-relabel (`gx,gy` swap to a future achieved position), and the reward recompute (`reach_reward` via `dist_s/dist_ns`). Update imports: `from goal_geometry import GOAL_RADIUS, SUCCESS_REWARD, PROGRESS_SCALE` (drop `GOAL_RANGE_NORM`). Return `self.action[t_slots]` (already int64).

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n sac-homebot python3 -m pytest tests/test_goal_buffer_dqn.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add goal_buffer.py tests/test_goal_buffer_dqn.py
git commit -m "bearing-reacher: HER buffer stores discrete actions, recomputes bearing goals"
```

---

## Task 4: Goal-conditioned discrete Q-network

**Files:** create `models/q_model.py`; test `tests/test_q_model.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_q_model.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
from models.q_model import GoalQNet
from goal_geometry import BEARING_DIM


def test_q_shape():
    net = GoalQNet(input_shape=(3, 96, 96), goal_dim=BEARING_DIM, n_actions=8, hidden_dim=256)
    img = torch.rand(5, 3, 96, 96)
    goal = torch.rand(5, BEARING_DIM)
    q = net(img, goal)
    assert q.shape == (5, 8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n sac-homebot python3 -m pytest tests/test_q_model.py -v`
Expected: FAIL — no module `models.q_model`.

- [ ] **Step 3: Implement `models/q_model.py`**

```python
# models/q_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseModel, weights_init_


class GoalQNet(BaseModel):
    """Goal-conditioned discrete Q-network: conv(image) -> embed, concat bearing,
    MLP -> Q-value per discrete action."""

    def __init__(self, input_shape, goal_dim, n_actions, hidden_dim,
                 checkpoint_dir='checkpoints', name='q_model'):
        super().__init__()
        c, h, w = input_shape
        self.conv1 = nn.Conv2d(c, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        conv_dim = 128 * (h // 8) * (w // 8)
        self.l1 = nn.Linear(conv_dim + goal_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, n_actions)
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = f"{checkpoint_dir}/{name}"
        self.apply(weights_init_)

    def forward(self, img, goal):
        x = F.relu(self.conv1(img))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = torch.cat([x, goal], dim=1)
        x = F.relu(self.ln1(self.l1(x)))
        x = F.relu(self.ln2(self.l2(x)))
        return self.out(x)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n sac-homebot python3 -m pytest tests/test_q_model.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add models/q_model.py tests/test_q_model.py
git commit -m "bearing-reacher: goal-conditioned discrete Q-network"
```

---

## Task 5: DQN+HER agent

**Files:** create `agent_dqn.py`; exercised by Task 6's integration test.

Double-DQN: online + target `GoalQNet`; epsilon-greedy collection on the discrete env; random spawn each episode; bearing goal to a sampled waypoint via `bearing_to`-style geometry (waypoint training target) with detector noise; HER buffer; un-gameable greedy eval.

- [ ] **Step 1: Implement `agent_dqn.py`**

```python
# agent_dqn.py
import os, subprocess, datetime
from collections import deque
import cv2, numpy as np, torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard.writer import SummaryWriter

from models.q_model import GoalQNet
from goal_buffer import GoalHERBuffer
from goal_manager import GoalManager
from goal_geometry import BEARING_DIM, GOAL_RADIUS, distance, bearing_vec, eval_step_budget


def _hard(t, s):
    for tp, sp in zip(t.parameters(), s.parameters()):
        tp.data.copy_(sp.data)


class ReacherDQN:
    def __init__(self, env, max_buffer_size=100000, gamma=0.99, lr=1e-4, tau=0.005,
                 start_radius=200.0):
        self.env = env
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.gamma, self.tau = gamma, tau
        os.makedirs("checkpoints", exist_ok=True); os.makedirs("runs", exist_ok=True)

        obs, _ = env.reset()
        self.input_shape = tuple(self.process(obs).shape)
        self.n_actions = env.action_space.n
        self.q = GoalQNet(self.input_shape, BEARING_DIM, self.n_actions, 256).to(self.device)
        self.qt = GoalQNet(self.input_shape, BEARING_DIM, self.n_actions, 256).to(self.device)
        _hard(self.qt, self.q)
        self.opt = Adam(self.q.parameters(), lr=lr)
        self.memory = GoalHERBuffer(max_buffer_size, self.input_shape, self.device, her_prob=0.8)
        self.goals = GoalManager(radius_px=start_radius)

    def process(self, obs):
        obs = cv2.resize(obs, (96, 96), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(obs).permute(2, 0, 1)

    def _goal_vec(self, base, noisy):
        v = bearing_vec(base._robot.x, base._robot.y, base._robot.angle, *self.goals.goal_px)
        return self.goals.add_noise(v) if noisy else v

    def _act(self, obs, goal_v, epsilon):
        if goal_v is None or np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            img = (obs.unsqueeze(0).float() / 255.0).to(self.device)
            g = torch.as_tensor(goal_v).unsqueeze(0).to(self.device)
            return int(self.q(img, g).argmax(1).item())

    def train_step(self, batch_size):
        img_s, goal_s, action, reward, img_ns, goal_ns, done = self.memory.sample(batch_size, self.gamma)
        img_s = (img_s / 255.0).to(self.device); img_ns = (img_ns / 255.0).to(self.device)
        reward = reward.to(self.device); done = done.to(self.device); action = action.to(self.device)
        with torch.no_grad():
            next_online = self.q(img_ns, goal_ns).argmax(1, keepdim=True)        # Double-DQN
            next_q = self.qt(img_ns, goal_ns).gather(1, next_online).squeeze(1)
            target = reward + (1.0 - done) * self.gamma * next_q
        q = self.q(img_s, goal_s).gather(1, action.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q, target)
        self.opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0); self.opt.step()
        for tp, sp in zip(self.qt.parameters(), self.q.parameters()):
            tp.data.copy_(tp.data * (1 - self.tau) + sp.data * self.tau)
        return loss.item()

    def greedy_eval(self, episodes=50):
        self.q.eval(); reached = 0
        for _ in range(episodes):
            obs, _ = self.env.reset(); base = self.env.unwrapped
            self.goals.random_spawn(base); self.goals.reset(base)
            obs = self.process(obs)
            budget = eval_step_budget(distance(base._robot.x, base._robot.y, *self.goals.goal_px))
            for _ in range(budget):
                a = self._act(obs, self._goal_vec(base, noisy=False), epsilon=0.0)
                nobs, _, _, _, _ = self.env.step(a)
                obs = self.process(nobs)
                if distance(base._robot.x, base._robot.y, *self.goals.goal_px) < GOAL_RADIUS:
                    reached += 1; break
        self.q.train()
        return reached / episodes

    def train(self, episodes=3000, max_steps=200, batch_size=128, warmup_episodes=10,
              grad_steps=100, eps_start=1.0, eps_end=0.1, eps_decay=500, eval_every=25, run_tag=None):
        if run_tag is None:
            try:
                refs = subprocess.check_output(['git', 'for-each-ref', '--format=%(refname:short)',
                       '--points-at', 'HEAD', 'refs/remotes/origin/'], stderr=subprocess.DEVNULL).decode().strip()
                run_tag = (refs.splitlines()[0].replace('origin/', '') if refs else
                           subprocess.check_output(['git', 'branch', '--show-current']).decode().strip()) or 'unknown'
            except Exception:
                run_tag = 'unknown'
        writer = SummaryWriter(f'runs/{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}_{run_tag}')
        best = -1.0
        for ep in range(episodes):
            obs, _ = self.env.reset(); base = self.env.unwrapped
            self.goals.random_spawn(base); self.goals.reset(base)
            obs = self.process(obs)
            eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / eps_decay)
            for _ in range(max_steps):
                rx, ry, rth = base._robot.x, base._robot.y, base._robot.angle
                a = (np.random.randint(self.n_actions) if ep < warmup_episodes
                     else self._act(obs, self._goal_vec(base, noisy=True), eps))
                nobs, _, _, trunc, _ = self.env.step(a)
                nobs_t = self.process(nobs)
                nrx, nry, nrth = base._robot.x, base._robot.y, base._robot.angle
                reached = distance(nrx, nry, *self.goals.goal_px) < GOAL_RADIUS
                self.memory.store(obs, a, nobs_t, rx, ry, rth, nrx, nry, nrth, self.goals.goal_px, reached or trunc)
                obs = nobs_t
                if reached or trunc:
                    break
            if ep >= warmup_episodes and self.memory.can_sample(batch_size):
                loss = 0.0
                for _ in range(grad_steps):
                    loss = self.train_step(batch_size)
                writer.add_scalar("DQN/loss", loss, ep)
            if ep % eval_every == 0 and ep >= warmup_episodes:
                gr = self.greedy_eval()
                writer.add_scalar("Eval/greedy_reach_rate", gr, ep)
                writer.add_scalar("Train/epsilon", eps, ep)
                print(f"Episode {ep} | greedy reach-rate: {gr:.2f} | eps {eps:.2f}", flush=True)
                if gr > best:
                    best = gr; self.q.save_the_model("q_model_best", verbose=True)
            if ep % 50 == 0:
                self.q.save_the_model("q_model")
```

- [ ] **Step 2: Commit (exercised by Task 6)**

```bash
git add agent_dqn.py
git commit -m "bearing-reacher: Double-DQN + HER agent (bearing goal, random spawn, un-gameable eval)"
```

---

## Task 6: train.py + integration smoke + full suite

**Files:** create `train.py`; test `tests/test_dqn_integration.py`.

- [ ] **Step 1: Write the failing integration test**

```python
# tests/test_dqn_integration.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import gymnasium as gym, homebot  # noqa: F401
from agent_dqn import ReacherDQN


def _env():
    for env_id in ("HomeBot2D-v1", "HomeBot2D-V1"):
        try:
            return gym.make(env_id, render_mode="rgb_array", action_mode="discrete",
                            obs_resolution=(96, 96), n_trash=2, max_steps=30,
                            map_name="default", goals=["trash"])
        except gym.error.Error:
            continue
    raise RuntimeError("no env")


def test_dqn_train_and_eval_run():
    env = _env()
    agent = ReacherDQN(env, max_buffer_size=2000, start_radius=150.0)
    agent.train(episodes=14, max_steps=20, batch_size=16, warmup_episodes=2,
                grad_steps=3, eval_every=5, run_tag="pytest-dqn")
    assert 0.0 <= agent.greedy_eval(episodes=2) <= 1.0
    env.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n sac-homebot python3 -m pytest tests/test_dqn_integration.py -v`
Expected: FAIL until `train.py`/agent wiring validated (or passes — acceptable).

- [ ] **Step 3: Implement `train.py`**

```python
# train.py
from agent_dqn import ReacherDQN
import gymnasium as gym
import homebot  # noqa: F401


def make_env():
    for env_id in ("HomeBot2D-v1", "HomeBot2D-V1"):
        try:
            env = gym.make(env_id, render_mode="rgb_array", action_mode="discrete",
                           obs_resolution=(96, 96), n_trash=2, max_steps=200,
                           map_name="default", goals=["trash"])
            print(f"Env: {env_id}")
            return env
        except gym.error.Error:
            continue
    raise RuntimeError("No HomeBot2D env id registered")


env = make_env()
agent = ReacherDQN(env, max_buffer_size=100000, start_radius=200.0)
agent.train(episodes=3000, max_steps=200, batch_size=128, warmup_episodes=10, grad_steps=100)
```

- [ ] **Step 4: Run integration + full suite**

Run: `conda run -n sac-homebot python3 -m pytest tests/test_dqn_integration.py -v` → PASS.
Run: `conda run -n sac-homebot python3 -m pytest tests/ -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add train.py tests/test_dqn_integration.py
git commit -m "bearing-reacher: train entry point + end-to-end DQN integration smoke"
```

---

## Task 7: Push + Beekeeper run

- [ ] **Step 1:** `git push -u origin bearing-reacher`
- [ ] **Step 2:** `get_capacity` (confirm `Q-Homebot` free) → `start_training(project_name="Q-Homebot", branch="bearing-reacher")` → `training_status` confirms running.
- [ ] **Step 3: Health check (analyze_run, not raw tails).** Watch `Eval/greedy_reach_rate` climb (target ≥0.8); `DQN/loss` bounded; `Train/epsilon` decaying. The eval is un-gameable (budget-limited), so the number is real reach, not a sweep.

---

## Self-Review Notes

- **Spec coverage:** bearing-primary input (Task 1,3,5) ✓; visibility-gated `bearing_to` stub + noise (Task 2) ✓; HER on, bearing recompute (Task 3) ✓; discrete Double-DQN (Task 4,5) ✓; random spawn (Task 2,5) ✓; un-gameable eval (Task 1 budget + Task 5 use) ✓; greedy-eval-first checkpoint (Task 5) ✓; realism rule (no exact range, no out-of-view oracle, random spawn our-side) ✓.
- **Training reward** uses `reach_reward` (potential shaping + success) computed from positions in the buffer — a training-time signal, allowed by the realism rule; the *policy* only ever sees the bearing.
- **`bearing_to` vs training goal:** in the first build, training goals are sampled waypoints (GoalManager) with the bearing computed + noised — `bearing_to(fixture)` is wired and tested for when B/landmark grounding comes online, but waypoints are the training distribution for the C primitive.
- **Env-improvement log:** running list seeded in the spec's "Recommended environment tweaks"; append anything new found during build.
- **Do not** re-introduce continuous actions, SAC, or precise range — those are explicitly out of scope.
```
