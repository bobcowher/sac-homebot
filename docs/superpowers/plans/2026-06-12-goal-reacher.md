# Polar Goal-Conditioned SAC Reacher (Tier C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A model-free, goal-conditioned continuous SAC policy that reaches arbitrary nearby waypoints on HomeBot2D — the low-level "reacher" rung of the eventual LLM→grounding→policy hierarchy.

**Architecture:** Plain continuous SAC (no world model). The policy is conditioned on a **polar goal** `[range_norm, sin(bearing), cos(bearing)]` (bearing relative to robot heading — aligns with the `[linear, angular]` action and with directional commands like "go left"). Dense **potential-based distance shaping** + **HER** (future relabeling) make the sparse reach reward learnable. A **near-goal spawn curriculum** keeps the reacher in its strong regime (auto-grows the goal radius as greedy success climbs). The deployable metric is **greedy reach-rate**, evaluated periodically; checkpointing is on greedy success, never training EMA.

**Tech Stack:** Python, PyTorch, Gymnasium, the local `homebot` env (`HomeBot2D-v1`, `action_mode="continuous"`), conda env `sac-homebot`, Beekeeper project `Q-Homebot`.

**Why this design (validated context):** The prior relative-goal HER run peaked at 0.9 *training EMA* but only ~45% *greedy* (~65% noisy), and faded. Causes were identified: DQN argmax-looping (→ use stochastic SAC), goals out-of-view ~half the time (→ near-goal curriculum keeps it in-regime; out-of-view "search" belongs in a higher Tier-B layer, out of scope here), and we never ran dense shaping or a stochastic policy. This plan addresses each and measures greedy from the start.

**Substrate decision:** Use `HomeBot2D-v1` (continuous), NOT the goal env. Goals are arbitrary reachable waypoints (random free tiles) sampled by our own `GoalManager` from `base._map.valid_floor_tiles()`. Reward, goal vector, and curriculum live in our code using robot pose (`base._robot.x/.y/.angle`) — the homebot package is untouched.

**Branch:** create `goal-reacher` off `main`.

---

## Conventions

- Tests run with `conda run -n sac-homebot python3 -m pytest <path> -v`.
- Env-id fallback: try `"HomeBot2D-v1"` then `"HomeBot2D-V1"`.
- Action is 2-D continuous `[linear, angular]`, both in `[-1, 1]`. `linear>0` = forward along heading; `angular` = turn rate.
- Constants live in `goal_geometry.py` and are imported everywhere:
  - `GOAL_DIM = 3` (range_norm, sin bearing, cos bearing)
  - `GOAL_RANGE_NORM = 500.0` (px, range normaliser)
  - `GOAL_RADIUS = 40.0` (px; within this = reached)
  - `SUCCESS_REWARD = 1.0`
  - `SHAPING_SCALE = 1.0`
- Robot heading `base._robot.angle` is radians, updated by `move_continuous`. Reset angle is 0.

---

## File Structure

**New:**
- `goal_geometry.py` — polar-goal + distance + reward math (pure functions + constants).
- `goal_manager.py` — waypoint sampling within a curriculum radius, per-step goal vector, achieved-goal, radius control.
- `models/goal_actor.py` — conv encoder + polar-goal-conditioned Gaussian-tanh actor.
- `models/goal_critic.py` — conv encoder + polar-goal-conditioned twin-Q critic.
- `goal_buffer.py` — goal-conditioned HER replay buffer (stores image + geometry; relabels + recomputes reward/polar at sample time).
- `agent_reacher.py` — the goal-SAC agent: collection, SAC update, greedy eval, curriculum controller, checkpoint-on-greedy.
- `train_reacher.py` — entry point.
- `evaluate_reacher.py` — standalone greedy reach-rate eval.

**Tests:**
- `tests/test_goal_geometry.py`, `tests/test_goal_manager.py`, `tests/test_goal_models.py`,
  `tests/test_goal_buffer.py`, `tests/test_reacher_integration.py`.

**Note:** `models/base.py` (with `BaseModel`, `weights_init_`) already exists on `main` and is reused.

---

## Task 1: Branch + goal geometry (polar + reward math)

**Files:**
- Create branch `goal-reacher`
- Create: `goal_geometry.py`
- Test: `tests/test_goal_geometry.py`

- [ ] **Step 1: Create the branch**

```bash
cd /home/robertcowher/pythonprojects/sac-homebot
git checkout main
git checkout -b goal-reacher
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_goal_geometry.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import math
import numpy as np
from goal_geometry import (
    polar_goal, distance, reach_reward, GOAL_DIM, GOAL_RADIUS, SUCCESS_REWARD,
)


def test_polar_goal_shape_and_facing():
    # Robot at origin, heading 0 (facing +x). Goal straight ahead on +x.
    g = polar_goal(0.0, 0.0, 0.0, 100.0, 0.0)
    assert g.shape == (GOAL_DIM,)
    # bearing 0 -> sin 0, cos 1
    assert abs(g[1] - 0.0) < 1e-6
    assert abs(g[2] - 1.0) < 1e-6


def test_polar_goal_bearing_left():
    # Goal directly above (+y is "down" in image coords; atan2(dy,dx) with dy>0).
    # Heading 0, goal at (0, 100): world_angle = atan2(100,0) = +pi/2.
    g = polar_goal(0.0, 0.0, 0.0, 0.0, 100.0)
    assert abs(g[1] - math.sin(math.pi / 2)) < 1e-6   # sin = 1
    assert abs(g[2] - math.cos(math.pi / 2)) < 1e-6   # cos = 0


def test_polar_goal_relative_to_heading():
    # Goal at +x, but robot already heading +pi/2 -> goal is at bearing -pi/2.
    g = polar_goal(0.0, 0.0, math.pi / 2, 100.0, 0.0)
    assert abs(g[1] - math.sin(-math.pi / 2)) < 1e-6  # sin = -1
    assert abs(g[2] - math.cos(-math.pi / 2)) < 1e-6  # cos = 0


def test_distance():
    assert abs(distance(0, 0, 3, 4) - 5.0) < 1e-6


def test_reach_reward_success_terminal():
    # next state inside the radius -> success reward + terminal
    r, done = reach_reward(dist_s=100.0, dist_ns=GOAL_RADIUS - 1, gamma=0.99)
    assert done is True
    assert r >= SUCCESS_REWARD  # success bonus present


def test_reach_reward_progress_positive():
    # moving closer (dist decreases) without reaching -> positive shaping, not done
    r, done = reach_reward(dist_s=200.0, dist_ns=150.0, gamma=0.99)
    assert done is False
    assert r > 0.0


def test_reach_reward_retreat_negative():
    # moving away -> negative shaping
    r, done = reach_reward(dist_s=150.0, dist_ns=200.0, gamma=0.99)
    assert done is False
    assert r < 0.0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `conda run -n sac-homebot python3 -m pytest tests/test_goal_geometry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'goal_geometry'`.

- [ ] **Step 4: Implement `goal_geometry.py`**

```python
# goal_geometry.py
"""Polar goal representation + potential-based reach reward (pure functions)."""
import math
import numpy as np

GOAL_DIM = 3                 # [range_norm, sin(bearing), cos(bearing)]
GOAL_RANGE_NORM = 500.0      # px range normaliser
GOAL_RADIUS = 40.0           # px; within this the goal is "reached"
SUCCESS_REWARD = 1.0
SHAPING_SCALE = 1.0


def distance(ax, ay, bx, by):
    return math.hypot(bx - ax, by - ay)


def polar_goal(rx, ry, rtheta, gx, gy):
    """Goal in robot-centric polar coords: [range/NORM, sin(bearing), cos(bearing)].

    bearing = angle to goal MINUS robot heading, so it is egocentric: bearing 0
    means "straight ahead", matching the [linear, angular] action.
    """
    dx, dy = gx - rx, gy - ry
    rng = math.hypot(dx, dy)
    bearing = math.atan2(dy, dx) - rtheta
    return np.array([rng / GOAL_RANGE_NORM, math.sin(bearing), math.cos(bearing)],
                    dtype=np.float32)


def reach_reward(dist_s, dist_ns, gamma):
    """Potential-based shaping + sparse success.

    Phi(s) = -dist(s) / GOAL_RANGE_NORM ; F = gamma*Phi(s') - Phi(s)  (Ng 1999:
    potential-based shaping preserves the optimal policy). Plus SUCCESS_REWARD and
    termination when the next state is within GOAL_RADIUS.
    """
    phi_s = -dist_s / GOAL_RANGE_NORM
    phi_ns = -dist_ns / GOAL_RANGE_NORM
    shaping = SHAPING_SCALE * (gamma * phi_ns - phi_s)
    success = dist_ns < GOAL_RADIUS
    reward = shaping + (SUCCESS_REWARD if success else 0.0)
    return float(reward), bool(success)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `conda run -n sac-homebot python3 -m pytest tests/test_goal_geometry.py -v`
Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add tests/test_goal_geometry.py goal_geometry.py
git commit -m "Reacher: polar goal representation + potential-based reach reward"
```

---

## Task 2: GoalManager — waypoint sampling + curriculum

**Files:**
- Create: `goal_manager.py`
- Test: `tests/test_goal_manager.py`

The manager owns the current goal waypoint. At reset it samples a random reachable tile within `radius_px` of the robot (the curriculum knob). It exposes the polar goal vector and the achieved goal (robot position). `valid_floor_tiles()` returns tile coords; `tile_to_pixel` converts to px.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_goal_manager.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import gymnasium as gym
import homebot  # noqa: F401
import numpy as np
from goal_manager import GoalManager
from goal_geometry import GOAL_DIM, GOAL_RADIUS, distance


def _make_env():
    for env_id in ("HomeBot2D-v1", "HomeBot2D-V1"):
        try:
            return gym.make(env_id, render_mode="rgb_array", action_mode="continuous",
                            obs_resolution=(96, 96), n_trash=2, max_steps=100,
                            map_name="default", goals=["trash"])
        except gym.error.Error:
            continue
    raise RuntimeError("No HomeBot2D env id registered")


def test_goal_within_radius_and_reachable():
    env = _make_env(); env.reset()
    base = env.unwrapped
    gm = GoalManager(radius_px=200.0)
    gm.reset(base)
    rx, ry = base._robot.x, base._robot.y
    d = distance(rx, ry, *gm.goal_px)
    # within curriculum radius and not trivially on top of the robot
    assert GOAL_RADIUS <= d <= 200.0 + 1e-3
    env.close()


def test_goal_vector_shape():
    env = _make_env(); env.reset()
    base = env.unwrapped
    gm = GoalManager(radius_px=200.0); gm.reset(base)
    v = gm.goal_vector(base)
    assert v.shape == (GOAL_DIM,)
    env.close()


def test_set_radius_grows():
    gm = GoalManager(radius_px=100.0)
    gm.set_radius(250.0)
    assert gm.radius_px == 250.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n sac-homebot python3 -m pytest tests/test_goal_manager.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'goal_manager'`.

- [ ] **Step 3: Implement `goal_manager.py`**

```python
# goal_manager.py
"""Owns the current goal waypoint: samples reachable tiles within a curriculum
radius of the robot, and exposes the polar goal vector + achieved goal."""
import numpy as np
from goal_geometry import polar_goal, distance, GOAL_RADIUS


class GoalManager:
    def __init__(self, radius_px: float, rng: np.random.Generator | None = None):
        self.radius_px = float(radius_px)
        self.rng = rng or np.random.default_rng()
        self.goal_px = (0.0, 0.0)

    def set_radius(self, radius_px: float):
        self.radius_px = float(radius_px)

    def reset(self, base):
        """Sample a reachable tile within radius_px of the robot (rejection sample).

        Guarantees GOAL_RADIUS < dist <= radius_px so the goal is neither already
        reached nor outside the curriculum band. Falls back to the farthest
        in-band candidate if rejection fails to find one in budget.
        """
        rx, ry = base._robot.x, base._robot.y
        tiles = base._map.valid_floor_tiles()
        best = None
        for _ in range(200):
            tx, ty = tiles[int(self.rng.integers(len(tiles)))]
            gx, gy = base._map.tile_to_pixel(tx, ty)
            d = distance(rx, ry, gx, gy)
            if GOAL_RADIUS < d <= self.radius_px:
                self.goal_px = (float(gx), float(gy))
                return
            if d > GOAL_RADIUS and (best is None or d < best[0]):
                best = (d, float(gx), float(gy))
        # fallback: nearest valid tile beyond GOAL_RADIUS
        self.goal_px = (best[1], best[2]) if best else (rx, ry)

    def goal_vector(self, base) -> np.ndarray:
        return polar_goal(base._robot.x, base._robot.y, base._robot.angle,
                          self.goal_px[0], self.goal_px[1])

    def achieved_px(self, base):
        return (float(base._robot.x), float(base._robot.y))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n sac-homebot python3 -m pytest tests/test_goal_manager.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_goal_manager.py goal_manager.py
git commit -m "Reacher: GoalManager waypoint sampling within curriculum radius"
```

---

## Task 3: Goal-conditioned actor + critic

**Files:**
- Create: `models/goal_actor.py`, `models/goal_critic.py`
- Test: `tests/test_goal_models.py`

Each net has its own conv encoder (image → features), concatenates the 3-D polar goal, and runs MLP heads. Actor is a Gaussian-tanh SAC policy; critic is twin-Q. Reuses `models/base.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_goal_models.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
import numpy as np
from gymnasium.spaces import Box
from models.goal_actor import GoalActor
from models.goal_critic import GoalCritic
from goal_geometry import GOAL_DIM


def _space():
    return Box(low=-np.ones(2, np.float32), high=np.ones(2, np.float32))


def test_actor_shapes():
    actor = GoalActor(input_shape=(3, 96, 96), goal_dim=GOAL_DIM, n_actions=2,
                      hidden_dim=256, action_space=_space())
    img = torch.rand(5, 3, 96, 96)
    goal = torch.rand(5, GOAL_DIM)
    a, logp, mean = actor.sample(img, goal)
    assert a.shape == (5, 2) and logp.shape == (5, 1) and mean.shape == (5, 2)


def test_critic_shapes():
    critic = GoalCritic(input_shape=(3, 96, 96), goal_dim=GOAL_DIM, n_actions=2,
                        hidden_dim=256)
    img = torch.rand(4, 3, 96, 96)
    goal = torch.rand(4, GOAL_DIM)
    action = torch.rand(4, 2)
    q1, q2 = critic(img, goal, action)
    assert q1.shape == (4, 1) and q2.shape == (4, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n sac-homebot python3 -m pytest tests/test_goal_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'models.goal_actor'`.

- [ ] **Step 3: Implement `models/goal_actor.py`**

```python
# models/goal_actor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from models.base import BaseModel, weights_init_

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPS = 1e-6


class GoalActor(BaseModel):
    def __init__(self, input_shape, goal_dim, n_actions, hidden_dim,
                 action_space=None, checkpoint_dir='checkpoints', name='goal_actor'):
        super().__init__()
        c, h, w = input_shape
        self.conv1 = nn.Conv2d(c, 32, 3, stride=2, padding=1)   # 96->48
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # 48->24
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # 24->12
        self.flatten = nn.Flatten()
        conv_dim = 128 * (h // 8) * (w // 8)

        self.linear1 = nn.Linear(conv_dim + goal_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, n_actions)
        self.log_std_linear = nn.Linear(hidden_dim, n_actions)

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = f"{checkpoint_dir}/{name}"
        self.apply(weights_init_)

        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.0)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.0)

    def _features(self, img, goal):
        x = F.relu(self.conv1(img))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = torch.cat([x, goal], dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return x

    def forward(self, img, goal):
        x = self._features(img, goal)
        mean = self.mean_linear(x)
        log_std = torch.clamp(self.log_std_linear(x), LOG_SIG_MIN, LOG_SIG_MAX)
        return mean, log_std

    def sample(self, img, goal):
        mean, log_std = self.forward(img, goal)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + EPS)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)
```

- [ ] **Step 4: Implement `models/goal_critic.py`**

```python
# models/goal_critic.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseModel, weights_init_


class GoalCritic(BaseModel):
    def __init__(self, input_shape, goal_dim, n_actions, hidden_dim,
                 checkpoint_dir='checkpoints', name='goal_critic'):
        super().__init__()
        c, h, w = input_shape
        conv_dim = 128 * (h // 8) * (w // 8)

        def conv_stack():
            return nn.ModuleList([
                nn.Conv2d(c, 32, 3, stride=2, padding=1),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
            ])

        # Two independent encoders + heads (twin Q).
        self.conv_a = conv_stack()
        self.conv_b = conv_stack()
        self.flatten = nn.Flatten()
        in_dim = conv_dim + goal_dim + n_actions
        self.a1 = nn.Linear(in_dim, hidden_dim); self.a2 = nn.Linear(hidden_dim, hidden_dim); self.a_out = nn.Linear(hidden_dim, 1)
        self.b1 = nn.Linear(in_dim, hidden_dim); self.b2 = nn.Linear(hidden_dim, hidden_dim); self.b_out = nn.Linear(hidden_dim, 1)

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = f"{checkpoint_dir}/{name}"
        self.apply(weights_init_)

    def _encode(self, convs, img):
        x = img
        for layer in convs:
            x = F.relu(layer(x))
        return self.flatten(x)

    def forward(self, img, goal, action):
        fa = torch.cat([self._encode(self.conv_a, img), goal, action], dim=1)
        fb = torch.cat([self._encode(self.conv_b, img), goal, action], dim=1)
        q1 = self.a_out(F.relu(self.a2(F.relu(self.a1(fa)))))
        q2 = self.b_out(F.relu(self.b2(F.relu(self.b1(fb)))))
        return q1, q2
```

- [ ] **Step 5: Run test to verify it passes**

Run: `conda run -n sac-homebot python3 -m pytest tests/test_goal_models.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add tests/test_goal_models.py models/goal_actor.py models/goal_critic.py
git commit -m "Reacher: goal-conditioned SAC actor + twin-Q critic (image + polar goal)"
```

---

## Task 4: Goal-conditioned HER replay buffer

**Files:**
- Create: `goal_buffer.py`
- Test: `tests/test_goal_buffer.py`

Stores image(s/s'), action, and **geometry** (robot pose at s and s', plus the desired goal px) per transition, with episode boundaries. At sample time it optionally relabels the desired goal with a future achieved position (HER, future strategy) and **recomputes the polar goal vectors and the reward** from `goal_geometry`. This is why geometry is stored rather than precomputed goal vectors — relabeling needs to recompute both.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_goal_buffer.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
import numpy as np
from goal_buffer import GoalHERBuffer
from goal_geometry import GOAL_DIM, GOAL_RADIUS


def _buf():
    return GoalHERBuffer(max_size=500, input_shape=(3, 96, 96), device="cpu",
                         action_dim=2, her_prob=1.0)


def _store_episode(buf, n, goal_px=(300.0, 300.0)):
    img = torch.zeros(3, 96, 96, dtype=torch.uint8)
    for t in range(n):
        # robot walks +x from 0..n along y=0; pose angle 0
        rx, ry = float(t * 10), 0.0
        nrx, nry = float((t + 1) * 10), 0.0
        done = (t == n - 1)
        buf.store(img, [0.1, 0.0], img,
                  rx, ry, 0.0, nrx, nry, 0.0, goal_px, done)


def test_sample_shapes_and_goal_dim():
    buf = _buf()
    _store_episode(buf, 30)
    img_s, goal_s, action, reward, img_ns, goal_ns, done = buf.sample(8, gamma=0.99)
    assert img_s.shape == (8, 3, 96, 96)
    assert goal_s.shape == (8, GOAL_DIM) and goal_ns.shape == (8, GOAL_DIM)
    assert action.shape == (8, 2)
    assert reward.shape == (8,) and done.shape == (8,)


def test_her_relabel_can_produce_success():
    # With her_prob=1, every transition's goal is relabeled to a FUTURE achieved
    # position in the same episode. The last few steps before that future point
    # land within GOAL_RADIUS of it, so some sampled transitions must be terminal.
    buf = _buf()
    _store_episode(buf, 40)
    any_done = False
    for _ in range(20):
        *_, done = buf.sample(32, gamma=0.99)
        if done.any():
            any_done = True
            break
    assert any_done, "HER relabeling should yield reached (terminal) transitions"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n sac-homebot python3 -m pytest tests/test_goal_buffer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'goal_buffer'`.

- [ ] **Step 3: Implement `goal_buffer.py`**

```python
# goal_buffer.py
"""Goal-conditioned HER replay buffer.

Stores image(s/s') + geometry (robot pose at s and s', desired goal px) and
episode boundaries. At sample time, with prob her_prob, relabels the desired
goal to a future achieved position within the same episode (future strategy),
then recomputes the polar goal vectors and the reach reward via goal_geometry.
"""
import numpy as np
import torch
from collections import deque
from goal_geometry import polar_goal, distance, reach_reward, GOAL_DIM


class GoalHERBuffer:
    def __init__(self, max_size, input_shape, device, action_dim=2, her_prob=0.8):
        self.max_size = max_size
        self.device = device
        self.her_prob = her_prob
        self.ctr = 0
        self.rng = np.random.default_rng()

        self.img_s = torch.zeros((max_size, *input_shape), dtype=torch.uint8, device=device)
        self.img_ns = torch.zeros((max_size, *input_shape), dtype=torch.uint8, device=device)
        self.action = torch.zeros((max_size, action_dim), dtype=torch.float32, device=device)
        # geometry: rx, ry, rtheta, nrx, nry, nrtheta, gx, gy
        self.geom = torch.zeros((max_size, 8), dtype=torch.float32, device=device)
        self.terminal = torch.zeros(max_size, dtype=torch.bool, device=device)

        self._episodes: deque[tuple[int, int]] = deque()
        self._ep_start = 0

    def store(self, img_s, action, img_ns, rx, ry, rth, nrx, nry, nrth, goal_px, done):
        i = self.ctr % self.max_size
        self.img_s[i] = torch.as_tensor(img_s, dtype=torch.uint8, device=self.device)
        self.img_ns[i] = torch.as_tensor(img_ns, dtype=torch.uint8, device=self.device)
        self.action[i] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.geom[i] = torch.tensor([rx, ry, rth, nrx, nry, nrth, goal_px[0], goal_px[1]],
                                    dtype=torch.float32, device=self.device)
        self.terminal[i] = bool(done)
        self.ctr += 1
        if done:
            self._episodes.append((self._ep_start, self.ctr))
            self._ep_start = self.ctr
            oldest = self.ctr - self.max_size
            while self._episodes and self._episodes[0][0] < oldest:
                self._episodes.popleft()

    def can_sample(self, batch_size):
        return self.ctr >= batch_size * 10 and len(self._episodes) > 0

    def _episode_of(self, abs_idx):
        for s, e in self._episodes:
            if s <= abs_idx < e:
                return s, e
        return None

    def sample(self, batch_size, gamma):
        filled = min(self.ctr, self.max_size)
        abs_min = self.ctr - filled
        idxs = self.rng.integers(abs_min, self.ctr, size=batch_size)

        goal_s = np.zeros((batch_size, GOAL_DIM), dtype=np.float32)
        goal_ns = np.zeros((batch_size, GOAL_DIM), dtype=np.float32)
        rewards = np.zeros(batch_size, dtype=np.float32)
        dones = np.zeros(batch_size, dtype=np.float32)
        slots = (idxs % self.max_size)

        geom = self.geom[torch.as_tensor(slots, device=self.device)].cpu().numpy()
        for b in range(batch_size):
            rx, ry, rth, nrx, nry, nrth, gx, gy = geom[b]
            ep = self._episode_of(int(idxs[b]))
            if ep is not None and self.rng.random() < self.her_prob:
                s, e = ep
                fut = int(self.rng.integers(int(idxs[b]), e))  # >= current, within episode
                fg = self.geom[fut % self.max_size, 3:5].cpu().numpy()  # achieved (nrx,nry) at fut
                gx, gy = float(fg[0]), float(fg[1])
            goal_s[b] = polar_goal(rx, ry, rth, gx, gy)
            goal_ns[b] = polar_goal(nrx, nry, nrth, gx, gy)
            d_s = distance(rx, ry, gx, gy)
            d_ns = distance(nrx, nry, gx, gy)
            r, success = reach_reward(d_s, d_ns, gamma)
            rewards[b] = r
            dones[b] = 1.0 if success else 0.0

        t_slots = torch.as_tensor(slots, device=self.device)
        return (
            self.img_s[t_slots].float(),
            torch.as_tensor(goal_s, device=self.device),
            self.action[t_slots],
            torch.as_tensor(rewards, device=self.device),
            self.img_ns[t_slots].float(),
            torch.as_tensor(goal_ns, device=self.device),
            torch.as_tensor(dones, device=self.device),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n sac-homebot python3 -m pytest tests/test_goal_buffer.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_goal_buffer.py goal_buffer.py
git commit -m "Reacher: goal-conditioned HER buffer (geometry storage, future relabel, reward recompute)"
```

---

## Task 5: Reacher agent — collection, SAC update, greedy eval, curriculum

**Files:**
- Create: `agent_reacher.py`
- Test: `tests/test_reacher_integration.py` (added in Task 6; this task is exercised by it)

The agent ties everything together. `process_observation` resizes to 96×96 and converts to `(C,H,W)` uint8 tensor. Collection drives the env with the goal-conditioned actor, resets the goal on reach or episode end, and stores transitions with geometry. The SAC update is standard twin-Q with **fixed alpha** (no auto-entropy). `greedy_eval` runs the deterministic policy and reports reach-rate. The curriculum controller grows the goal radius when greedy reach-rate clears a threshold.

- [ ] **Step 1: Implement `agent_reacher.py`**

```python
# agent_reacher.py
import os
import subprocess
import datetime
from collections import deque
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard.writer import SummaryWriter

from models.goal_actor import GoalActor
from models.goal_critic import GoalCritic
from goal_buffer import GoalHERBuffer
from goal_manager import GoalManager
from goal_geometry import GOAL_DIM, GOAL_RADIUS, distance


def _hard_update(target, source):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(s.data)


def _soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)


class ReacherAgent:
    def __init__(self, env, max_buffer_size=100000, alpha=0.1, tau=0.005,
                 gamma=0.99, start_radius=150.0, max_radius=600.0):
        self.env = env
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.alpha, self.tau, self.gamma = alpha, tau, gamma
        os.makedirs("checkpoints", exist_ok=True)

        obs, _ = env.reset()
        self.input_shape = self.process_observation(obs).shape  # (3,96,96)
        self.action_space = env.action_space
        self.n_actions = int(self.action_space.shape[0])
        hid = 256

        self.actor = GoalActor(self.input_shape, GOAL_DIM, self.n_actions, hid, self.action_space).to(self.device)
        self.critic = GoalCritic(self.input_shape, GOAL_DIM, self.n_actions, hid).to(self.device)
        self.critic_target = GoalCritic(self.input_shape, GOAL_DIM, self.n_actions, hid).to(self.device)
        _hard_update(self.critic_target, self.critic)

        self.actor_optim = Adam(self.actor.parameters(), lr=3e-5)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-4)

        self.memory = GoalHERBuffer(max_buffer_size, self.input_shape, self.device, self.n_actions)
        self.goals = GoalManager(radius_px=start_radius)
        self.max_radius = max_radius

    def process_observation(self, obs):
        obs = cv2.resize(obs, (96, 96), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(obs).permute(2, 0, 1)

    def _act(self, img_t, goal_t, evaluate):
        with torch.no_grad():
            a, _, mean = self.actor.sample(img_t, goal_t)
        out = mean if evaluate else a
        return out.detach().cpu().numpy()[0]

    def warmup_action(self):
        # Heavy forward bias on linear; zero-mean turning (see analysis: zero-mean
        # linear spins in place). linear always >= 0.
        if np.random.random() < 0.5:
            return np.array([np.random.uniform(0.6, 1.0), np.random.uniform(-0.3, 0.3)], np.float32)
        return np.array([np.random.uniform(0.3, 0.8), np.random.uniform(-1.0, 1.0)], np.float32)

    def train_step(self, batch_size):
        img_s, goal_s, action, reward, img_ns, goal_ns, done = self.memory.sample(batch_size, self.gamma)
        img_s = (img_s / 255.0).to(self.device)
        img_ns = (img_ns / 255.0).to(self.device)
        reward = reward.unsqueeze(1).to(self.device)
        done = done.unsqueeze(1).to(self.device)

        with torch.no_grad():
            na, nlogp, _ = self.actor.sample(img_ns, goal_ns)
            q1t, q2t = self.critic_target(img_ns, goal_ns, na)
            min_q = torch.min(q1t, q2t) - self.alpha * nlogp
            target_q = reward + (1.0 - done) * self.gamma * min_q

        q1, q2 = self.critic(img_s, goal_s, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_optim.zero_grad(); critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0); self.critic_optim.step()

        pi, logp, _ = self.actor.sample(img_s, goal_s)
        q1pi, q2pi = self.critic(img_s, goal_s, pi)
        actor_loss = (self.alpha * logp - torch.min(q1pi, q2pi)).mean()
        self.actor_optim.zero_grad(); actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0); self.actor_optim.step()

        _soft_update(self.critic_target, self.critic, self.tau)
        return critic_loss.item(), actor_loss.item()

    def greedy_eval(self, episodes=50, max_steps=300):
        self.actor.eval()
        reached = 0
        for _ in range(episodes):
            obs, _ = self.env.reset()
            base = self.env.unwrapped
            self.goals.reset(base)
            obs = self.process_observation(obs)
            for _ in range(max_steps):
                img_t = (obs.unsqueeze(0).float() / 255.0).to(self.device)
                goal_t = torch.as_tensor(self.goals.goal_vector(base)).unsqueeze(0).to(self.device)
                action = self._act(img_t, goal_t, evaluate=True)
                nobs, _, _, trunc, _ = self.env.step(action)
                obs = self.process_observation(nobs)
                if distance(base._robot.x, base._robot.y, *self.goals.goal_px) < GOAL_RADIUS:
                    reached += 1
                    break
                if trunc:
                    break
        self.actor.train()
        return reached / episodes

    def train(self, episodes=2000, max_steps=300, batch_size=256, warmup_episodes=10,
              grad_steps=300, eval_every=25, run_tag=None):
        if run_tag is None:
            try:
                refs = subprocess.check_output(
                    ['git', 'for-each-ref', '--format=%(refname:short)', '--points-at', 'HEAD',
                     'refs/remotes/origin/'], stderr=subprocess.DEVNULL).decode().strip()
                run_tag = (refs.splitlines()[0].replace('origin/', '') if refs else
                           subprocess.check_output(['git', 'branch', '--show-current']).decode().strip()) or 'unknown'
            except Exception:
                run_tag = 'unknown'
        writer = SummaryWriter(f'runs/{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}_{run_tag}')
        best_greedy = -1.0

        for episode in range(episodes):
            obs, _ = self.env.reset()
            base = self.env.unwrapped
            self.goals.reset(base)
            obs = self.process_observation(obs)

            for _ in range(max_steps):
                rx, ry, rth = base._robot.x, base._robot.y, base._robot.angle
                if episode < warmup_episodes:
                    action = self.warmup_action()
                else:
                    img_t = (obs.unsqueeze(0).float() / 255.0).to(self.device)
                    goal_t = torch.as_tensor(self.goals.goal_vector(base)).unsqueeze(0).to(self.device)
                    action = self._act(img_t, goal_t, evaluate=False)

                nobs, _, _, trunc, _ = self.env.step(action)
                nobs_t = self.process_observation(nobs)
                nrx, nry, nrth = base._robot.x, base._robot.y, base._robot.angle
                reached = distance(nrx, nry, *self.goals.goal_px) < GOAL_RADIUS

                self.memory.store(obs, action, nobs_t, rx, ry, rth, nrx, nry, nrth,
                                  self.goals.goal_px, reached or trunc)
                obs = nobs_t
                if reached:
                    self.goals.reset(base)   # new waypoint, keep episode going
                if trunc:
                    break

            if episode >= warmup_episodes and self.memory.can_sample(batch_size):
                for _ in range(grad_steps):
                    closs, aloss = self.train_step(batch_size)
                writer.add_scalar("SAC/critic_loss", closs, episode)
                writer.add_scalar("SAC/actor_loss", aloss, episode)

            if episode % eval_every == 0 and episode >= warmup_episodes:
                gr = self.greedy_eval()
                writer.add_scalar("Eval/greedy_reach_rate", gr, episode)
                writer.add_scalar("Curriculum/radius_px", self.goals.radius_px, episode)
                print(f"Episode {episode} | greedy reach-rate: {gr:.2f} | radius: {self.goals.radius_px:.0f}")
                if gr > best_greedy:
                    best_greedy = gr
                    self.actor.save_the_model("goal_actor_best", verbose=True)
                    self.critic.save_the_model("goal_critic_best", verbose=True)
                # Curriculum: grow the goal band once the reacher is strong in-regime.
                if gr >= 0.8 and self.goals.radius_px < self.max_radius:
                    self.goals.set_radius(min(self.max_radius, self.goals.radius_px + 75.0))

            if episode % 50 == 0:
                self.actor.save_the_model("goal_actor")
                self.critic.save_the_model("goal_critic")
```

- [ ] **Step 2: Commit (exercised by the Task 6 integration test)**

```bash
git add agent_reacher.py
git commit -m "Reacher: goal-SAC agent — collection, twin-Q update, greedy eval, radius curriculum"
```

---

## Task 6: train_reacher.py + evaluate_reacher.py + integration smoke

**Files:**
- Create: `train_reacher.py`, `evaluate_reacher.py`
- Test: `tests/test_reacher_integration.py`

- [ ] **Step 1: Write the failing integration smoke test**

```python
# tests/test_reacher_integration.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import gymnasium as gym
import homebot  # noqa: F401
from agent_reacher import ReacherAgent


def _make_env():
    for env_id in ("HomeBot2D-v1", "HomeBot2D-V1"):
        try:
            return gym.make(env_id, render_mode="rgb_array", action_mode="continuous",
                            obs_resolution=(96, 96), n_trash=2, max_steps=40,
                            map_name="default", goals=["trash"])
        except gym.error.Error:
            continue
    raise RuntimeError("No HomeBot2D env id registered")


def test_reacher_train_and_eval_run():
    env = _make_env()
    agent = ReacherAgent(env, max_buffer_size=3000, start_radius=150.0)
    # tiny: a few episodes past warmup so a train_step + a greedy_eval both fire
    agent.train(episodes=12, max_steps=20, batch_size=16, warmup_episodes=2,
                grad_steps=3, eval_every=5, run_tag="pytest-reacher")
    rate = agent.greedy_eval(episodes=2, max_steps=20)
    assert 0.0 <= rate <= 1.0
    env.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n sac-homebot python3 -m pytest tests/test_reacher_integration.py -v`
Expected: FAIL — collection/eval wiring not yet validated, or passes if `agent_reacher.py` is already complete (acceptable — proceed).

- [ ] **Step 3: Implement `train_reacher.py`**

```python
# train_reacher.py
from agent_reacher import ReacherAgent
import gymnasium as gym
import homebot  # noqa: F401


def make_env():
    for env_id in ("HomeBot2D-v1", "HomeBot2D-V1"):
        try:
            env = gym.make(env_id, render_mode="rgb_array", action_mode="continuous",
                           obs_resolution=(96, 96), n_trash=2, max_steps=300,
                           map_name="default", goals=["trash"])
            print(f"Env: {env_id}")
            return env
        except gym.error.Error:
            continue
    raise RuntimeError("No HomeBot2D env id registered")


env = make_env()
agent = ReacherAgent(env, max_buffer_size=100000, start_radius=150.0, max_radius=600.0)
agent.train(episodes=2000, max_steps=300, batch_size=256, warmup_episodes=10,
            grad_steps=300, eval_every=25)
```

- [ ] **Step 4: Implement `evaluate_reacher.py`**

```python
# evaluate_reacher.py
"""Standalone greedy reach-rate eval for a trained reacher checkpoint.

Usage: python3 evaluate_reacher.py --episodes 100 --radius 300
"""
import argparse
import gymnasium as gym
import homebot  # noqa: F401
from agent_reacher import ReacherAgent


def make_env():
    for env_id in ("HomeBot2D-v1", "HomeBot2D-V1"):
        try:
            return gym.make(env_id, render_mode="rgb_array", action_mode="continuous",
                            obs_resolution=(96, 96), n_trash=2, max_steps=300,
                            map_name="default", goals=["trash"])
        except gym.error.Error:
            continue
    raise RuntimeError("No HomeBot2D env id registered")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--radius", type=float, default=300.0)
    args = ap.parse_args()
    env = make_env()
    agent = ReacherAgent(env, max_buffer_size=1000, start_radius=args.radius)
    agent.actor.load_the_model("goal_actor_best", device=agent.device)
    rate = agent.greedy_eval(episodes=args.episodes)
    print(f"greedy reach-rate @ radius {args.radius:.0f}: {rate:.2f} ({100*rate:.0f}%)")
    env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the integration smoke + full suite**

Run: `conda run -n sac-homebot python3 -m pytest tests/test_reacher_integration.py -v`
Expected: PASS (train loop + greedy eval run end-to-end).

Run: `conda run -n sac-homebot python3 -m pytest tests/ -q`
Expected: all reacher tests pass.

- [ ] **Step 6: Commit**

```bash
git add tests/test_reacher_integration.py train_reacher.py evaluate_reacher.py
git commit -m "Reacher: train + greedy-eval entry points + end-to-end integration smoke"
```

---

## Task 7: Push + first Beekeeper run

**Files:** none (ops)

- [ ] **Step 1: Push**

```bash
git push -u origin goal-reacher
```

- [ ] **Step 2: Launch on Q-Homebot / branch `goal-reacher`**

Confirm `get_capacity` shows `Q-Homebot` free, then `start_training(project_name="Q-Homebot", branch="goal-reacher")`, then `training_status` to confirm `running`. One run at a time.

- [ ] **Step 3: Health check (use analyze_run, not raw log tails)**

The metric that matters is **`Eval/greedy_reach_rate`** — it should climb past 0.8 at the starting radius within the first few hundred episodes, then `Curriculum/radius_px` should start stepping up. `SAC/critic_loss` should be non-trivial (NOT collapsing to ~0 like the starved runs — dense shaping keeps it alive). If greedy reach-rate is stuck near 0 with critic loss ~0, the shaping/HER wiring is suspect.

---

## Self-Review Notes (for the executor)

- **Greedy-first:** judge ONLY by `Eval/greedy_reach_rate`. Training-reward EMA is deliberately not even the headline metric this time.
- **Curriculum gate:** radius grows only when greedy ≥ 0.8 at the current radius — so success rate stays high as difficulty rises. If it stalls at a radius, that radius is the reacher's current ceiling — a data point, not a bug.
- **Fixed alpha (0.1):** do NOT add automatic entropy tuning (it has consistently failed here).
- **Forward-biased warmup** is kept (linear ≥ 0) so warmup actually traverses; the dense shaping then drives directed movement once learning starts.
- **Multi-goal episodes:** reaching a waypoint resamples a new one and continues the episode (more reach events per episode), but truncation still bounds episode length.
- **Polar bearing** is encoded as `(sin, cos)` to avoid the ±π wrap; range is normalised by `GOAL_RANGE_NORM`.
- **HER recompute:** the buffer stores geometry, not precomputed goal vectors, precisely so relabeling can recompute both polar goal and reward. Keep that invariant if extending.
- **Scope boundary:** out-of-view / "search for an unseen target" is explicitly NOT handled here — that's the higher Tier-B layer. This rung only reaches goals whose direction is given.
```
