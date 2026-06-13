# goal_buffer.py
"""Goal-conditioned HER replay buffer.

Stores image(s/s') + geometry (robot pose at s and s', desired goal px) and an
O(1) per-slot episode-end index. At sample time, with prob her_prob, relabels the
desired goal to a future achieved position within the same episode (future
strategy), then recomputes the polar goal vectors and the reach reward.

Geometry (not precomputed goal vectors) is stored precisely so relabeling can
recompute both the polar goal and the reward. Sampling is fully vectorised
(numpy) so the per-transition trig does not become a Python-loop bottleneck.
"""
import numpy as np
import torch
from goal_geometry import (
    GOAL_DIM, GOAL_RANGE_NORM, GOAL_RADIUS, SUCCESS_REWARD, SHAPING_SCALE,
)


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
        # geometry columns: rx, ry, rtheta, nrx, nry, nrtheta, gx, gy
        self.geom = np.zeros((max_size, 8), dtype=np.float32)
        # absolute (exclusive) end index of the episode each slot belongs to;
        # filled when the episode closes -> O(1) future-index range at sample time.
        self.ep_end_abs = np.zeros(max_size, dtype=np.int64)

        self._ep_start = 0   # abs index where the in-progress episode began
        self._closed_until = 0  # abs index; transitions in [abs_min, _closed_until) are sampleable

    def store(self, img_s, action, img_ns, rx, ry, rth, nrx, nry, nrth, goal_px, done):
        i = self.ctr % self.max_size
        self.img_s[i] = torch.as_tensor(img_s, dtype=torch.uint8, device=self.device)
        self.img_ns[i] = torch.as_tensor(img_ns, dtype=torch.uint8, device=self.device)
        self.action[i] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.geom[i] = (rx, ry, rth, nrx, nry, nrth, goal_px[0], goal_px[1])
        self.ctr += 1
        if done:
            end = self.ctr
            for abs_idx in range(self._ep_start, end):
                self.ep_end_abs[abs_idx % self.max_size] = end
            self._ep_start = end
            self._closed_until = end

    def can_sample(self, batch_size):
        filled = min(self.ctr, self.max_size)
        abs_min = self.ctr - filled
        closed = self._closed_until - abs_min
        return self.ctr >= batch_size * 10 and closed >= batch_size

    def sample(self, batch_size, gamma):
        filled = min(self.ctr, self.max_size)
        abs_min = self.ctr - filled
        # sample only from closed episodes so ep_end_abs is valid
        idxs = self.rng.integers(abs_min, max(self._closed_until, abs_min + 1), size=batch_size)
        slots = idxs % self.max_size

        g = self.geom[slots]
        rx, ry, rth = g[:, 0], g[:, 1], g[:, 2]
        nrx, nry, nrth = g[:, 3], g[:, 4], g[:, 5]
        gx, gy = g[:, 6].copy(), g[:, 7].copy()

        # HER future relabel
        her_mask = self.rng.random(batch_size) < self.her_prob
        ep_ends = self.ep_end_abs[slots]
        # future in [idx, ep_end); guard high>low
        highs = np.maximum(ep_ends, idxs + 1)
        future_abs = self.rng.integers(idxs, highs)
        fut_slots = future_abs % self.max_size
        ach = self.geom[fut_slots]  # achieved at future = (nrx, nry) cols 3,4
        gx = np.where(her_mask, ach[:, 3], gx)
        gy = np.where(her_mask, ach[:, 4], gy)

        def polar(px, py, th):
            dx, dy = gx - px, gy - py
            rng = np.hypot(dx, dy)
            bearing = np.arctan2(dy, dx) - th
            return np.stack([rng / GOAL_RANGE_NORM, np.sin(bearing), np.cos(bearing)], axis=1)

        goal_s = polar(rx, ry, rth).astype(np.float32)
        goal_ns = polar(nrx, nry, nrth).astype(np.float32)
        dist_s = np.hypot(gx - rx, gy - ry)
        dist_ns = np.hypot(gx - nrx, gy - nry)
        success = dist_ns < GOAL_RADIUS
        shaping = SHAPING_SCALE * (dist_s - gamma * dist_ns) / GOAL_RANGE_NORM
        reward = (shaping + np.where(success, SUCCESS_REWARD, 0.0)).astype(np.float32)
        done = success.astype(np.float32)

        t_slots = torch.as_tensor(slots, device=self.device)
        return (
            self.img_s[t_slots].float(),
            torch.as_tensor(goal_s, device=self.device),
            self.action[t_slots],
            torch.as_tensor(reward, device=self.device),
            self.img_ns[t_slots].float(),
            torch.as_tensor(goal_ns, device=self.device),
            torch.as_tensor(done, device=self.device),
        )
