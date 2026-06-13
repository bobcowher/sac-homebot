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
