# HomeBot World-Model (Dreamer-style) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the car-racing Dreamer-style world model to HomeBot2D-V1 (trash task), adding a supervised goal-detection aux loss so the reconstruction-grounded latent actually encodes small movable goal objects (which pure reconstruction drops).

**Architecture:** A recurrent latent world model (encoder + GRU + residual dynamics + reward/done heads) trained on episode sequences; an SAC actor-critic trained on a mix of real and imagined latent rollouts; and a NEW multi-channel goal-detection head off the per-frame embedding, supervised by env object positions, that forces small goal objects into the latent. The actor/critic are MLPs operating on the 1024-d latent (the encoder does all perception).

**Tech Stack:** Python, PyTorch, Gymnasium, the local `homebot` env package, conda env `sac-homebot`, Beekeeper for remote training (project `Q-Homebot`).

**Why this exists (validated):** A probe (`probe_encoder.py` on branch `wm-encoder-probe`) showed pure reconstruction is blind to trash (trash reconstructs at 16–24× the error of a random patch), but a shallow detection head off the latent localizes trash 92% vs a 5% prior floor once it is supervised. So the world model is viable **only** with the detection aux loss baked in.

**Source to port from:** `/home/robertcowher/pythonprojects/car-racing/car-racing-world-models-continuous/` (referred to below as `$CR`). Several files are copied verbatim; CarRacing-specific code (`decode_action`, hardcoded `Box[-1,1]²`, `warmup_action` gas bias, `CarRacing-v3` env) is replaced with HomeBot equivalents.

**Branch:** create `world-model` off `main` (the current SAC line).

---

## Conventions used throughout this plan

- `conda run -n sac-homebot python3 -m pytest <path> -v` runs tests (never `python3 -c`).
- Constants (define once, in `models/detection_head.py`, imported elsewhere):
  - `OBS = 96` (obs H/W)
  - `GRID = 24` (detection heatmap resolution; each cell ≈ 4px)
  - `DETECT_CHANNELS = 1` (trash; multi-channel-ready)
  - `K_LABEL_SLOTS = 4` (max labelled objects stored per frame)
- Label row format (compact, multi-channel-ready): each labelled object is `(channel, x, y)` in obs-pixel coords; unused slots are `(-1, -1, -1)`. Stored as `int16`.
- HomeBot env id fallback: try `"HomeBot2D-v1"` then `"HomeBot2D-V1"` (remote registers lowercase, local capital).

---

## File Structure

**Copied verbatim from `$CR` (no logic change):**
- `models/encoder.py` — `Encoder` (conv→1024 embed) + `Decoder`. Already present on `wm-encoder-probe`.
- `models/ssim_loss.py` — SSIM reconstruction loss. Already present on `wm-encoder-probe`.
- `models/dynamics_model.py` — residual next-embed predictor; already continuous-action-ready (concatenates the action vector).

**Copied then modified:**
- `models/actor.py` — MLP actor on the latent (REPLACES main's conv SAC actor on this branch).
- `models/critic.py` — twin-Q MLP critic on the latent (REPLACES main's conv SAC critic on this branch).
- `models/world_model.py` — port + ADD detection head wiring + detection aux loss term.
- `buffer.py` — port `EpisodeReplayBuffer` + ADD compact label storage + return labels from `sample_sequences`.
- `agent.py` — port WM `Agent` + strip CarRacing bits + HomeBot action space/warmup + thread detection labels through collection + rolling-window-best checkpoint.
- `train.py` — HomeBot `make_env` (continuous + id fallback) + WM `Agent` + params.

**New:**
- `models/detection_head.py` — `DetectionHead` + the shared constants + heatmap target builder.
- `goal_labels.py` — extract object label rows from the live env (viewport projection of `trash_positions`).

**Tests (new):**
- `tests/test_episode_buffer_wm.py` — sequence sampling + label round-trip.
- `tests/test_goal_labels.py` — viewport projection correctness + heatmap builder.
- `tests/test_detection_head.py` — head forward shape.
- `tests/test_world_model_wm.py` — `compute_loss_sequential` runs incl. detection loss; `imagine_step` shapes.
- `tests/test_wm_integration.py` — 2-episode end-to-end smoke.

---

## Task 1: Branch + bring over the reusable models

**Files:**
- Create branch `world-model`
- Copy: `models/encoder.py`, `models/ssim_loss.py`, `models/dynamics_model.py`, `models/actor.py`, `models/critic.py`
- Test: `tests/test_wm_models_smoke.py`

- [ ] **Step 1: Create the branch off main**

```bash
cd /home/robertcowher/pythonprojects/sac-homebot
git checkout main
git checkout -b world-model
```

- [ ] **Step 2: Copy the reusable model files from car-racing**

```bash
CR=/home/robertcowher/pythonprojects/car-racing/car-racing-world-models-continuous
cp "$CR/models/encoder.py"        models/encoder.py
cp "$CR/models/ssim_loss.py"      models/ssim_loss.py
cp "$CR/models/dynamics_model.py" models/dynamics_model.py
cp "$CR/models/actor.py"          models/actor.py
cp "$CR/models/critic.py"         models/critic.py
```

Note: this REPLACES `models/actor.py` and `models/critic.py` (the SAC conv versions). On this branch the actor/critic are MLPs over the latent — the encoder does perception. `models/base.py` is byte-identical to car-racing's, so the `from models.base import ...` imports resolve unchanged.

- [ ] **Step 3: Write the model smoke test**

```python
# tests/test_wm_models_smoke.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
from models.encoder import Encoder, Decoder
from models.dynamics_model import DynamicsModel
from models.actor import Actor
from models.critic import Critic
from gymnasium.spaces import Box
import numpy as np


def test_encoder_decoder_roundtrip_shape():
    enc = Encoder(observation_shape=(3, 96, 96), embed_dim=1024)
    dec = Decoder(observation_shape=(3, 96, 96), embed_dim=1024,
                  conv_output_shape=enc.get_output_shape(),
                  conv_channels=enc.get_conv_channels())
    x = torch.rand(2, 3, 96, 96)
    z = enc(x)
    assert z.shape == (2, 1024)
    assert dec(z).shape == (2, 3, 96, 96)


def test_dynamics_residual_shape():
    dyn = DynamicsModel(embed_dim=1024, n_actions=2, hidden_dim=2048)
    z = torch.rand(5, 1024)
    a = torch.rand(5, 2)
    assert dyn(z, a).shape == (5, 1024)


def test_actor_critic_latent_shapes():
    space = Box(low=-np.ones(2, np.float32), high=np.ones(2, np.float32))
    actor = Actor(num_inputs=1024, num_actions=2, hidden_dim=256, action_space=space)
    critic = Critic(num_inputs=1024, num_actions=2, hidden_dim=256)
    z = torch.rand(4, 1024)
    a, logp, mean = actor.sample(z)
    assert a.shape == (4, 2) and logp.shape == (4, 1)
    q1, q2 = critic(z, a)
    assert q1.shape == (4, 1) and q2.shape == (4, 1)
```

- [ ] **Step 4: Run the smoke test**

Run: `conda run -n sac-homebot python3 -m pytest tests/test_wm_models_smoke.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add -f tests/test_wm_models_smoke.py
git add models/encoder.py models/ssim_loss.py models/dynamics_model.py models/actor.py models/critic.py
git commit -m "WM port: bring over encoder/decoder/dynamics/actor/critic from car-racing"
```

---

## Task 2: Detection head + shared constants + heatmap target builder

**Files:**
- Create: `models/detection_head.py`
- Test: `tests/test_detection_head.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n sac-homebot python3 -m pytest tests/test_detection_head.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.detection_head'`.

- [ ] **Step 3: Implement `models/detection_head.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n sac-homebot python3 -m pytest tests/test_detection_head.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add -f tests/test_detection_head.py
git add models/detection_head.py
git commit -m "WM: add multi-channel-ready goal-detection head + heatmap target builder"
```

---

## Task 3: Goal-label extraction from the live env

**Files:**
- Create: `goal_labels.py`
- Test: `tests/test_goal_labels.py`

Reference: the projection math is the validated `trash_pixels_in_view` from `probe_encoder.py` (branch `wm-encoder-probe`). The env exposes `base._robot.x/.y`, `base._renderer._viewport_w/_h`, `base._map.pixel_width/pixel_height`, `base._map.tile_to_pixel(tx,ty)`, and `base._task_manager.trash_positions` (tile coords).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_goal_labels.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import gymnasium as gym
import homebot  # noqa: F401
from goal_labels import label_rows
from models.detection_head import K_LABEL_SLOTS, OBS


def _make_env():
    for env_id in ("HomeBot2D-v1", "HomeBot2D-V1"):
        try:
            return gym.make(env_id, render_mode="rgb_array", action_mode="continuous",
                            obs_resolution=(OBS, OBS), n_trash=2, max_steps=100,
                            map_name="default", goals=["trash"])
        except gym.error.Error:
            continue
    raise RuntimeError("No HomeBot2D env id registered")


def test_label_rows_shape_and_padding():
    env = _make_env()
    env.reset()
    rows = label_rows(env.unwrapped)
    env.close()
    assert rows.shape == (K_LABEL_SLOTS, 3)
    # every non-padding row is (channel>=0, x in [0,OBS), y in [0,OBS))
    for c, x, y in rows:
        if c >= 0:
            assert 0 <= x < OBS and 0 <= y < OBS
            assert c == 0  # trash channel


def test_label_rows_all_padding_when_no_trash_in_view():
    # Teleport robot far from all trash so none is in view; expect all -1 rows.
    env = _make_env()
    base = env.unwrapped
    base.reset()
    # Move robot to a corner tile; some configs will still see trash, so only
    # assert the structure holds and padding rows are exactly (-1,-1,-1).
    rows = label_rows(base)
    env.close()
    for c, x, y in rows:
        if c < 0:
            assert (c, x, y).count(-1) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n sac-homebot python3 -m pytest tests/test_goal_labels.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goal_labels'`.

- [ ] **Step 3: Implement `goal_labels.py`**

```python
# goal_labels.py
"""Extract goal-object label rows from the live HomeBot env for the detection head.

Each row is (channel, x, y) in obs-pixel coords; padding rows are (-1, -1, -1).
Channel 0 = trash. Add channels here when a future goal introduces a small
movable object (the fixtures drink/package deliver to are large and already
captured by reconstruction, so they get no channel).
"""
import numpy as np
from models.detection_head import OBS, K_LABEL_SLOTS

TRASH_CHANNEL = 0


def _trash_pixels_in_view(base):
    """Trash positions projected into the OBSxOBS frame (only those in view).

    Mirrors the renderer's clamped, robot-centered viewport extraction.
    """
    r = base._robot
    rend = base._renderer
    vw, vh = rend._viewport_w, rend._viewport_h
    mw, mh = base._map.pixel_width, base._map.pixel_height
    vx = max(0, min(int(r.x - vw / 2), mw - vw))
    vy = max(0, min(int(r.y - vh / 2), mh - vh))
    pts = []
    for pos in base._task_manager.trash_positions:
        px, py = base._map.tile_to_pixel(*pos)
        if vx <= px <= vx + vw and vy <= py <= vy + vh:
            ox = int((px - vx) / vw * OBS)
            oy = int((py - vy) / vh * OBS)
            if 0 <= ox < OBS and 0 <= oy < OBS:
                pts.append((ox, oy))
    return pts


def label_rows(base):
    """Return (K_LABEL_SLOTS, 3) int16 array of (channel, x, y); padding = -1."""
    rows = np.full((K_LABEL_SLOTS, 3), -1, dtype=np.int16)
    pts = _trash_pixels_in_view(base)[:K_LABEL_SLOTS]
    for i, (ox, oy) in enumerate(pts):
        rows[i] = (TRASH_CHANNEL, ox, oy)
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n sac-homebot python3 -m pytest tests/test_goal_labels.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add -f tests/test_goal_labels.py
git add goal_labels.py
git commit -m "WM: extract goal-object label rows from live env (viewport projection)"
```

---

## Task 4: EpisodeReplayBuffer with compact label storage

**Files:**
- Create: `buffer.py` (replace main's flat ReplayBuffer with the car-racing buffer + label storage)
- Test: `tests/test_episode_buffer_wm.py`

Reference: copy the `ReplayBuffer` and `EpisodeReplayBuffer` classes verbatim from `$CR/buffer.py`, then apply the additions below. The verbatim classes provide `store_transition`, `sample_buffer`, `sample_nstep`, `_episodes` tracking, `can_sample_sequences`, and `sample_sequences`.

- [ ] **Step 1: Copy the car-racing buffer verbatim**

```bash
CR=/home/robertcowher/pythonprojects/car-racing/car-racing-world-models-continuous
cp "$CR/buffer.py" buffer.py
```

- [ ] **Step 2: Write the failing test (label round-trip through sequences)**

```python
# tests/test_episode_buffer_wm.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
from buffer import EpisodeReplayBuffer
from models.detection_head import K_LABEL_SLOTS


def _buf():
    return EpisodeReplayBuffer(max_size=200, input_shape=(3, 96, 96),
                               input_device="cpu", output_device="cpu", action_dim=2)


def _labels(x):
    rows = torch.full((K_LABEL_SLOTS, 3), -1, dtype=torch.int16)
    rows[0] = torch.tensor([0, x, x], dtype=torch.int16)
    return rows


def test_sequences_return_labels_aligned():
    buf = _buf()
    obs = torch.zeros(3, 96, 96, dtype=torch.uint8)
    # one episode of 10 steps; label x == step index
    for t in range(10):
        done = (t == 9)
        buf.store_transition(obs, [0.0, 0.0], 0.0, obs, done, done, _labels(t))
    batch = buf.sample_sequences(batch_size=5, sequence_length=5)
    assert batch["labels"].shape == (1, 5, K_LABEL_SLOTS, 3)
    # within the sampled contiguous window, label x increments by 1 each step
    xs = batch["labels"][0, :, 0, 1]
    assert torch.all(xs[1:] - xs[:-1] == 1)


def test_can_sample_sequences_gate():
    buf = _buf()
    obs = torch.zeros(3, 96, 96, dtype=torch.uint8)
    for t in range(5):
        buf.store_transition(obs, [0.0, 0.0], 0.0, obs, t == 4, t == 4, _labels(t))
    # only one short episode -> cannot sample 10-long sequences
    assert not buf.can_sample_sequences(batch_size=10, sequence_length=10)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `conda run -n sac-homebot python3 -m pytest tests/test_episode_buffer_wm.py -v`
Expected: FAIL — `store_transition()` takes no `labels` arg / `batch["labels"]` KeyError.

- [ ] **Step 4: Add label storage to `buffer.py`**

In `ReplayBuffer.__init__`, after `self.episode_done_memory = ...`, add the label tensor:

```python
        # Detection labels: up to K objects per frame, each (channel, x, y); pad -1.
        from models.detection_head import K_LABEL_SLOTS
        self.label_memory = torch.full(
            (max_size, K_LABEL_SLOTS, 3), -1, dtype=torch.int16, device=self.input_device
        )
```

Change `ReplayBuffer.store_transition` signature and body to accept and store labels:

```python
    def store_transition(self, state, action, reward, next_state, terminal, episode_done, labels):
        idx = self.mem_ctr % self.mem_size
        self.state_memory[idx]        = torch.as_tensor(state, dtype=torch.uint8, device=self.input_device)
        self.next_state_memory[idx]   = torch.as_tensor(next_state, dtype=torch.uint8, device=self.input_device)
        self.action_memory[idx]       = torch.as_tensor(action, dtype=torch.float32, device=self.input_device)
        self.reward_memory[idx]       = float(reward)
        self.terminal_memory[idx]     = bool(terminal)
        self.episode_done_memory[idx] = bool(episode_done)
        self.label_memory[idx]        = torch.as_tensor(labels, dtype=torch.int16, device=self.input_device)
        self.mem_ctr += 1
```

In `EpisodeReplayBuffer.store_transition`, forward the new arg:

```python
    def store_transition(self, state, action, reward, next_state, terminal, episode_done, labels):
        super().store_transition(state, action, reward, next_state, terminal, episode_done, labels)
        if episode_done:
            self._episodes.append((self._current_episode_start, self.mem_ctr))
            self._current_episode_start = self.mem_ctr
            oldest_valid_index = self.mem_ctr - self.mem_size
            while self._episodes and self._episodes[0][0] < oldest_valid_index:
                self._episodes.popleft()
```

In `EpisodeReplayBuffer.sample_sequences`, add a labels accumulator. After `dones_out = []` add `labels_out = []`; inside the per-sequence loop after `dones_out.append(...)` add:

```python
            labels_out.append(self.label_memory[indices])
```

and in the returned dict add:

```python
            "labels":   torch.stack(labels_out).to(output_device),    # (num_sequences, sequence_length, K_LABEL_SLOTS, 3)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `conda run -n sac-homebot python3 -m pytest tests/test_episode_buffer_wm.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add -f tests/test_episode_buffer_wm.py
git add buffer.py
git commit -m "WM: EpisodeReplayBuffer with compact per-frame detection-label storage"
```

---

## Task 5: WorldModel + detection aux loss

**Files:**
- Create: `models/world_model.py` (copy from car-racing, then add detection head + loss)
- Test: `tests/test_world_model_wm.py`

Reference: copy `$CR/models/world_model.py` verbatim, then apply the additions below. The verbatim file provides `Encoder`/`Decoder`/`DynamicsModel` wiring, the GRU, `encode`, `decode`, `imagine_step`, and `compute_loss_sequential` (recon + dynamics + overshoot + reward + done).

- [ ] **Step 1: Copy the car-racing world model verbatim**

```bash
CR=/home/robertcowher/pythonprojects/car-racing/car-racing-world-models-continuous
cp "$CR/models/world_model.py" models/world_model.py
```

- [ ] **Step 2: Write the failing test**

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `conda run -n sac-homebot python3 -m pytest tests/test_world_model_wm.py -v`
Expected: FAIL — `wm.detection_head` AttributeError / `"detect"` not in loss dict.

- [ ] **Step 4: Wire the detection head into `models/world_model.py`**

Add imports near the top:

```python
import torch.nn as nn
from models.detection_head import DetectionHead, build_detection_targets
```

In `WorldModel.__init__`, after `self.gru = nn.GRU(...)`, add:

```python
        self.detection_head = DetectionHead(embed_dim=embed_dim)
        # ~9 positive cells (3x3) of GRID*GRID within a frame that has an object.
        self.detect_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(60.0))
        self.detect_weight = 5.0
```

In `compute_loss_sequential`, after the reconstruction block computes `embeds` (shape `(N, T, embed_dim)`), add the detection loss. Insert this block before `combined_loss = (...)`:

```python
        # === Detection loss: shallow head off the per-frame embedding ===
        # Forces small goal objects into the latent (pure reconstruction drops
        # 2-3px trash). Trained ONLY on frames that contain a labelled object so
        # the head cannot collapse to predicting "nothing" everywhere.
        embeds_det = embeds.reshape(num_sequences * sequence_length, -1)
        labels_flat = batch["labels"].reshape(num_sequences * sequence_length,
                                               batch["labels"].shape[-2], 3)
        det_tgt = build_detection_targets(labels_flat, device=embeds.device)
        has_obj = (labels_flat[:, :, 0] >= 0).any(dim=1)
        if has_obj.any():
            det_logits = self.detection_head(embeds_det[has_obj])
            detect_loss = self.detect_bce.to(embeds.device)(det_logits, det_tgt[has_obj])
        else:
            detect_loss = torch.zeros((), device=embeds.device)
```

Change the `combined_loss` assignment to include detection:

```python
        combined_loss = (1.0 * recon_loss
                         + 1.0 * dynamics_loss
                         + self.overshoot_weight * overshoot_loss
                         + 2.0 * reward_loss
                         + 0.5 * done_loss
                         + self.detect_weight * detect_loss)
```

Add `"detect"` to the returned `loss_dict`:

```python
            "detect":    detect_loss.item(),
```

- [ ] **Step 5: Run test to verify it passes**

Run: `conda run -n sac-homebot python3 -m pytest tests/test_world_model_wm.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add -f tests/test_world_model_wm.py
git add models/world_model.py
git commit -m "WM: detection aux loss in world model (masked BCE off per-frame embed)"
```

---

## Task 6: Agent — collection with labels, WM/AC training, HomeBot adaptation

**Files:**
- Create: `agent.py` (copy from car-racing, then apply HomeBot + label diffs)
- Test: `tests/test_wm_agent_unit.py`

Reference: copy `$CR/agent.py` verbatim, then apply the diffs below. The verbatim file provides `MixedSampler`, `imagine_trajectory`, `train_world_model`, `train_actor_critic`, `evaluate_reconstruction`, `select_action`, `test`, and the `train` loop.

- [ ] **Step 1: Copy the car-racing agent verbatim**

```bash
CR=/home/robertcowher/pythonprojects/car-racing/car-racing-world-models-continuous
cp "$CR/agent.py" agent.py
```

- [ ] **Step 2: Strip CarRacing specifics + use HomeBot action space**

In `Agent.__init__`, REPLACE the hardcoded `self.actor_action_space = Box(...)` block (the 2D `[steering, throttle_brake]` Box) with the env's own continuous action space:

```python
        self.actor_action_space = env.action_space
        self.n_actions = int(self.actor_action_space.shape[0])
```

DELETE the entire `decode_action` method (HomeBot consumes the continuous action directly — no 3D CarRacing mapping).

REPLACE `warmup_action` with a HomeBot version (uniform over the env action space):

```python
    def warmup_action(self) -> np.ndarray:
        return self.actor_action_space.sample()
```

- [ ] **Step 3: Replace every `decode_action(...)` call site with the raw action**

There are call sites in `test()` and in `train()` (`self.env.step(self.decode_action(actor_action))` and `car_action = self.decode_action(actor_action)`). Change them to step the action directly:

```python
        next_obs, reward, term, trunc, _ = self.env.step(actor_action)
```

(in `test()` the line becomes `next_obs, reward, term, trunc, _ = self.env.step(actor_action)`; in `train()` delete the `car_action = ...` line and use `actor_action` in `self.env.step(...)`.)

- [ ] **Step 4: Thread detection labels through collection**

Add the import near the top of `agent.py`:

```python
from goal_labels import label_rows
```

In `train()`, the collection loop currently calls
`self.memory.store_transition(obs, actor_action, reward, next_obs, term, episode_done)`.
Capture the label rows from the live env at each step and pass them through. Replace that call with:

```python
                labels = label_rows(self.env.unwrapped)
                self.memory.store_transition(obs, actor_action, reward, next_obs,
                                             term, episode_done, labels)
```

- [ ] **Step 5: Swap single-episode best for rolling-window-best checkpoint**

HomeBot trash reward is sparse-binary; single-episode `best_score` fires once. Replace the `best_score` logic with the rolling-100-window-best pattern (same as the SAC line on `main`). Add near the top of `train()` (after `mixed_sampler = ...`):

```python
        from collections import deque
        success_window = deque(maxlen=100)
        best_window_rate = -1.0
        success_reward = 2.0  # full clear of n_trash=2
```

REPLACE the block:

```python
            if episode_reward > best_score:
                best_score = episode_reward
                self.save_best(best_score, episode)
```

with:

```python
            success_window.append(1.0 if episode_reward >= success_reward else 0.0)
            window_rate = sum(success_window) / len(success_window)
            if len(success_window) == success_window.maxlen and window_rate > best_window_rate:
                best_window_rate = window_rate
                self.save_best(window_rate, episode)
            writer.add_scalar("Train/success_rate_100", window_rate, episode)
```

(`best_score` may remain for the TB `Train/best_score` scalar; keep updating `if episode_reward > best_score: best_score = episode_reward` without the save.)

- [ ] **Step 6: Write the agent unit test (label-threaded smoke of one collected step)**

```python
# tests/test_wm_agent_unit.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import gymnasium as gym
import homebot  # noqa: F401
from agent import Agent
from models.detection_head import K_LABEL_SLOTS


def _make_env():
    for env_id in ("HomeBot2D-v1", "HomeBot2D-V1"):
        try:
            return gym.make(env_id, render_mode="rgb_array", action_mode="continuous",
                            obs_resolution=(96, 96), n_trash=2, max_steps=30,
                            map_name="default", goals=["trash"])
        except gym.error.Error:
            continue
    raise RuntimeError("No HomeBot2D env id registered")


def test_agent_builds_and_warmup_action_matches_space():
    env = _make_env()
    agent = Agent(env=env, max_buffer_size=500)
    a = agent.warmup_action()
    assert a.shape == env.action_space.shape
    assert not hasattr(agent, "decode_action")  # CarRacing mapping removed
    # buffer stores labels of the right shape
    assert agent.memory.label_memory.shape[1:] == (K_LABEL_SLOTS, 3)
    env.close()
```

- [ ] **Step 7: Run the agent unit test**

Run: `conda run -n sac-homebot python3 -m pytest tests/test_wm_agent_unit.py -v`
Expected: 1 passed.

- [ ] **Step 8: Commit**

```bash
git add -f tests/test_wm_agent_unit.py
git add agent.py
git commit -m "WM agent: HomeBot action space, drop decode_action, label-threaded collection, rolling-best ckpt"
```

---

## Task 7: HomeBot train.py + end-to-end integration smoke

**Files:**
- Create: `train.py` (HomeBot world-model entry point)
- Test: `tests/test_wm_integration.py`

- [ ] **Step 1: Write the failing integration smoke test**

```python
# tests/test_wm_integration.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import gymnasium as gym
import homebot  # noqa: F401
from agent import Agent


def _make_env():
    for env_id in ("HomeBot2D-v1", "HomeBot2D-V1"):
        try:
            return gym.make(env_id, render_mode="rgb_array", action_mode="continuous",
                            obs_resolution=(96, 96), n_trash=2, max_steps=40,
                            map_name="default", goals=["trash"])
        except gym.error.Error:
            continue
    raise RuntimeError("No HomeBot2D env id registered")


def test_two_episode_wm_train_runs():
    env = _make_env()
    agent = Agent(env=env, max_buffer_size=2000, wm_sequence_length=10)
    # warmup=1 so AC + WM updates both fire; tiny sizes to keep it fast.
    agent.train(episodes=2, offline_training_epochs=2, batch_size=8,
                wm_batch_size=20, imagination_steps=2, real_ratio=0.5,
                warmup_episodes=1, run_tag="pytest-wm-smoke")
    env.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n sac-homebot python3 -m pytest tests/test_wm_integration.py -v`
Expected: FAIL — `train.py`/env wiring not yet validated; or `can_sample_sequences` gating needs the small sizes. (If it passes immediately because agent.py is complete, that's acceptable — proceed.)

- [ ] **Step 3: Implement `train.py`**

```python
# train.py
from agent import Agent
import gymnasium as gym
import homebot  # noqa: F401  (side-effect env registration)


def make_env():
    for env_id in ("HomeBot2D-v1", "HomeBot2D-V1"):
        try:
            env = gym.make(
                env_id,
                render_mode="rgb_array",
                action_mode="continuous",  # world-model SAC: continuous Box actions
                obs_resolution=(96, 96),
                n_trash=2,
                max_steps=1000,
                map_name="default",
                goals=["trash"],
            )
            print(f"Env: {env_id}")
            return env
        except gym.error.Error:
            continue
    raise RuntimeError("No HomeBot2D env id registered")


env = make_env()

agent = Agent(env=env, max_buffer_size=200000, wm_sequence_length=50)

agent.train(episodes=1200, offline_training_epochs=200, batch_size=32,
            wm_batch_size=200, imagination_steps=4, real_ratio=0.5)
```

- [ ] **Step 4: Run the integration smoke test**

Run: `conda run -n sac-homebot python3 -m pytest tests/test_wm_integration.py -v`
Expected: 1 passed (2 episodes train end-to-end: collection with labels, WM sequence updates, AC updates on mixed real/imagined latent rollouts).

- [ ] **Step 5: Run the full test suite**

Run: `conda run -n sac-homebot python3 -m pytest tests/ -v`
Expected: all tests pass (models smoke, detection head, goal labels, episode buffer, world model, agent unit, integration).

- [ ] **Step 6: Commit**

```bash
git add -f tests/test_wm_integration.py
git add train.py
git commit -m "WM: HomeBot world-model train entry point + end-to-end integration smoke"
```

---

## Task 8: Validation harness — latent trash-visibility + policy success

**Files:**
- Create: `validate_wm.py`
- Test: manual run (this is an analysis script, not a unit-tested module)

Purpose: two checks on a trained WM checkpoint before trusting a Beekeeper run — (1) does the learned latent actually encode trash (reuse the probe's localization metric, but on the WM's frozen encoder + detection head), and (2) does the WM policy collect trash (success rate via greedy rollout).

- [ ] **Step 1: Implement `validate_wm.py`**

```python
# validate_wm.py
"""Validate a trained HomeBot world-model checkpoint.

(1) Latent trash-visibility: collect diversified frames (teleport robot to random
    floor tiles), run the frozen WM encoder + detection head, and report the
    detection localization hit-rate vs the prior floor (reuses the probe's logic).
(2) Policy success: greedy rollouts, report full-clear rate on the trash task.

Usage:
    python3 validate_wm.py --episodes 100
"""
import argparse
import numpy as np
import torch
import cv2
import gymnasium as gym
import homebot  # noqa: F401
from agent import Agent
from goal_labels import label_rows
from models.detection_head import OBS, GRID


def make_env(max_steps=1000):
    for env_id in ("HomeBot2D-v1", "HomeBot2D-V1"):
        try:
            return gym.make(env_id, render_mode="rgb_array", action_mode="continuous",
                            obs_resolution=(OBS, OBS), n_trash=2, max_steps=max_steps,
                            map_name="default", goals=["trash"])
        except gym.error.Error:
            continue
    raise RuntimeError("No HomeBot2D env id registered")


@torch.no_grad()
def latent_trash_visibility(agent, env, n_frames=4000):
    base = env.unwrapped
    rng = np.random.default_rng(0)
    floor = base._map.valid_floor_tiles()
    env.reset()
    hits = total = 0
    for i in range(n_frames):
        tx, ty = floor[int(rng.integers(len(floor)))]
        base._robot.x, base._robot.y = base._map.tile_to_pixel(tx, ty)
        if i % 40 == 0:
            env.reset()
            base._robot.x, base._robot.y = base._map.tile_to_pixel(tx, ty)
        frame = cv2.resize(base._get_obs(), (OBS, OBS), interpolation=cv2.INTER_NEAREST)
        rows = label_rows(base)
        true = [(int(r[2]) * GRID // OBS, int(r[1]) * GRID // OBS) for r in rows if r[0] >= 0]
        if not true:
            continue
        total += 1
        obs_t = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(agent.device) / 255.0
        embed, _, _ = agent.world_model.encode(obs_t)
        hm = torch.sigmoid(agent.world_model.detection_head(embed.squeeze(1)))[0, 0].cpu().numpy()
        py, px = np.unravel_index(int(hm.argmax()), hm.shape)
        if any(abs(py - gy) <= 1 and abs(px - gx) <= 1 for gy, gx in true):
            hits += 1
    print(f"latent trash-visibility hit-rate: {hits}/{total} = {100 * hits / max(total, 1):.0f}%")


def policy_success(agent, env, episodes=100):
    full = 0
    for ep in range(episodes):
        obs, _ = env.reset()
        obs = agent.process_observation(obs)
        done, ep_r = False, 0.0
        while not done:
            with torch.no_grad():
                obs_t = obs.unsqueeze(0).float().to(agent.device) / 255.0
                embed, _, _ = agent.world_model.encode(obs_t)
                action = agent.select_action(embed.squeeze(1), evaluate=True)
            nxt, r, term, trunc, _ = env.step(action)
            obs = agent.process_observation(nxt)
            done = term or trunc
            ep_r += float(r)
        full += int(ep_r >= 2)
    print(f"policy full-clear rate: {full}/{episodes} = {100 * full / episodes:.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    args = ap.parse_args()
    env = make_env()
    agent = Agent(env=env, max_buffer_size=1000)
    agent.load()  # loads world_model/actor/critic from checkpoints/
    agent.actor.eval()
    latent_trash_visibility(agent, env)
    policy_success(agent, env, args.episodes)
    env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Sanity-run on the smoke checkpoint**

After Task 7 the integration smoke leaves `checkpoints/world_model.pt` etc. Run:

Run: `conda run -n sac-homebot python3 validate_wm.py --episodes 5`
Expected: prints a (poor, untrained) latent hit-rate and a full-clear rate without error. This validates the harness wiring, not performance.

- [ ] **Step 3: Commit**

```bash
git add validate_wm.py
git commit -m "WM: validation harness — latent trash-visibility + policy success rate"
```

---

## Task 9: Push + first Beekeeper run

**Files:** none (ops)

- [ ] **Step 1: Push the branch**

```bash
git push -u origin world-model
```

- [ ] **Step 2: Confirm Beekeeper capacity, then launch on `Q-Homebot` / branch `world-model`**

Use the Beekeeper MCP tools: `get_capacity` (confirm `Q-Homebot` has a free slot), then `start_training(project_name="Q-Homebot", branch="world-model")`, then `training_status` to confirm it is `running`. One run at a time per the campaign protocol; do not stack a second.

- [ ] **Step 3: Early health check (do NOT judge from raw log tails)**

After ~30–60 min use `analyze_run`. Watch `World Model/reconstruction_loss` (should fall), `World Model/reward_loss` and `World Model/done_loss` (should fall — the model is learning the sparse reward), and `World Model/detect` if surfaced. `Train/episode_reward` and `Train/success_rate_100` are the task metrics. The detection-loss falling is the signal that the latent is becoming trash-aware. If reconstruction collapses or losses NaN, stop and inspect.

---

## Self-Review Notes (for the executor)

- **Reward scale assumption:** `success_reward = 2.0` assumes `n_trash=2` and +1 per pickup. If `n_trash` changes, update it in `agent.train`.
- **`detect_weight = 5.0`** mirrors the probe's effective weighting; it is the first knob to tune if the latent stays trash-blind (raise) or if reconstruction/dynamics degrade (lower). It is a named constant on `WorldModel`, not a magic literal.
- **Label timing:** `label_rows(self.env.unwrapped)` is read AFTER `env.step` (post-move robot/viewport), matching the stored `next`-frame viewport geometry used by the probe. Keep that ordering.
- **Multi-goal extension path:** add a channel id in `goal_labels.label_rows` for any future *small movable* object and bump `DETECT_CHANNELS`; the head, target builder, and loss already iterate channels. Large fixtures (drink/package delivery targets) intentionally get no channel — reconstruction already captures them.
- **alpha stays fixed (0.1).** Do NOT wire automatic entropy tuning — it has consistently failed here. (`alpha_loss` remains a no-op.)
- **GRU/sequences:** the world model needs `EpisodeReplayBuffer.sample_sequences`; AC training also draws real transitions via `sample_nstep`. Both come from the same buffer — do not swap in the flat `ReplayBuffer`.
```
