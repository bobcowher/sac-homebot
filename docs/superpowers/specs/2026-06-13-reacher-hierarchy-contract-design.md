# HomeBot Hierarchy Contract — A/B/C Interface Design

**Status:** design / spec
**Date:** 2026-06-13
**Purpose:** Lock the interface between the high-level planner and the low-level
policy so we stop wandering up and down the stack. Defines *what each tier owns*
and *exactly what crosses each boundary*. The first implementation is the Tier-C
reacher + a Tier-B grounding stub; the LLM planner and the real detector/search
modules are documented but out of scope for the first build.

---

## Why this exists

We have repeatedly redesigned the low-level policy (DQN→SAC→world-model→SAC-AE)
without a fixed contract for what the planner hands it. Every redesign silently
assumed a different amount of upstream machinery. This spec fixes the contract so
the tiers can evolve independently. Notably: **DQN+HER already reached ~80% on
cross-room goals** (memoryless, egocentric obs + relative goal) before we
abandoned it. That is the asset this contract is built around.

## The three tiers

| Tier | Owns | Does NOT own |
|---|---|---|
| **A — LLM planner** | task → ordered *semantic* subgoals over named places/objects ("go_to_fridge", "go_to_human"); semantic decomposition ("doorway, then fridge") | geometry, coordinates, bearings |
| **B — grounding + search/memory** | recognition (what a fridge looks like); locating a target (perception now, memory if not visible); turning a semantic target into a **bearing**; recovery (retry/re-ground/escalate); exploration/search for unseen targets | locomotion; metric global planning (not required) |
| **C — reacher** | locomotion: drive toward a given bearing, avoid local obstacles, report outcome | what/where the target is; search; memory of the map |

Key principle: **the stack is coordinate-free.** The LLM thinks in
places/relations; B produces directions; C drives directions. Metric SLAM/coords
are an *optional precision upgrade*, never a requirement.

## The contract (what crosses each boundary)

### A → B
An ordered list of **semantic subgoals** (named targets). B executes them one at
a time and reports completion/failure so A can advance or replan.

### B → C  (per control step)
- **`goal`: bearing-primary, egocentric, coordinate-free.**
  `[sin(bearing), cos(bearing)]` toward the current target, where
  `bearing = angle_to_target − robot_heading`. Optionally a **coarse** range
  bucket (near/mid/far); **never** a precise metric distance.
- **`visible`: bool.** If the target is not currently perceivable, B emits
  `visible=False`; C should not be asked to chase an unseen target — B runs
  *search* instead.
- The goal is **refreshed every step** by B (mechanism — live detection vs
  remembered-location-plus-odometry — is B's internal business, invisible to C).
- C also consumes its own **egocentric observation** (for obstacle avoidance).

### C → B
A single outcome enum, polled when an episode/leg ends:
- **`reached`** — within interaction range of the target (detection-based arrival:
  target fills view / coarse-near).
- **`timed_out`** — step budget exhausted without reaching.
- **`stuck`** — no net progress toward the goal for K steps.

B owns recovery: **re-ground → retry → escalate to A** (`reached` advances the
plan; `timed_out` re-grounds and retries; repeated `stuck`/`timed_out` escalates
"can't reach X" to A).

### C's guarantees / non-guarantees
- **Guarantees:** given a bearing to a *perceivable* target, navigate toward it
  avoiding local obstacles, reaching it with target success ≥80% within the step
  budget, and report `reached/timed_out/stuck`.
- **Does NOT guarantee:** finding targets it can't see (search = B), routing to
  targets with no visible path (reports failure), any global localization.

Because C consumes only a bearing, B's internals (detector, tracker, memory) can
change without touching C or the contract.

## Tier C — the reacher (first build)

- **Revive DQN + HER** (the ~80% asset), not SAC. Discrete 8-direction actions
  are fine for navigation in sim; continuous is a later deployment detail, not a
  research blocker.
- **Condition on the bearing** `[sin, cos]` (+ optional coarse range), NOT the
  precise relative coordinate the old version used. This matches what B can
  actually produce at deploy.
- **HER stays compatible:** relabel the *target position* with an achieved
  position, then recompute the bearing — same geometry-storage pattern already in
  `goal_buffer` (store positions, recompute the goal representation at sample).
- **Memoryless baseline (Level 3, per semantic leg).** Memory (recurrent C) is a
  future upgrade that shares this exact contract — adding it never touches A/B.
- **Train against a degraded bearing** (domain randomization on the goal input):
  random dropout (`visible=False` for a few steps), small angular jitter, and
  occasional wrong-target — so a sim-trained C survives a real, flaky detector.
- **Greedy-eval-first**, and the eval must be **un-gameable by circling**: a fixed
  sweep policy must not be able to "pass" (e.g., a step budget short enough that a
  circle can't sweep the area, and/or scoring that requires goal-directed
  approach). This is a hard requirement — the current eval was gamed by a circle.

## Tier B — grounding stub (first build)

In sim we do not run a detector on rendered pixels. The honest, cheap stand-in is
a **geometry-gated oracle**: ground-truth target position, gated by what the robot
could actually perceive, returned as a bearing.

```python
def bearing_to(base, target):              # "door", "fridge", "human", ...
    gx, gy = base._map.tile_to_pixel(*base._map.fixtures[target])
    if not in_view(base, gx, gy):          # inside the robot-centered viewport box
        return None                         # -> visible=False (B searches)
    if not line_of_sight(base, gx, gy):    # wall between robot and target -> occluded
        return None
    dx, dy = gx - base._robot.x, gy - base._robot.y
    bearing = math.atan2(dy, dx) - base._robot.angle
    return math.sin(bearing), math.cos(bearing)
```

- The **gates are the honest part** — they reproduce a real detector's only
  fundamental limit (no bearing to what you can't see); the ground-truth position
  just stands in for "the detector localized it."
- **Uniform across targets** — same function for door/fridge/human; the label
  selects the entry. This *is* the entire B-perception stub.
- **Noise wrapper** dials it from oracle → realistically flaky (dropout, jitter,
  wrong-target) for C's robustness training.
- **Real detector swaps into the same slot** later (Grounding-DINO / YOLO-World +
  a tracker) with zero downstream change.

## Search / memory (Tier B) — future

Out of scope for the first build. For now, "target not visible" triggers a
**simple scan** (rotate to bring candidates into view). True search of unknown
space + a topological/landmark memory ("remember the fridge") is a named future
Tier-B module — the real home-robot capability — designed separately, feeding C
the same bearing contract.

## Success criteria

- **C primitive:** greedy reach-rate ≥80% on *visible-bearing* goals, eval
  un-gameable by circling, robust to bearing dropout/jitter.
- **Contract:** adding memory to C, or swapping B's detector, requires **no change
  to the boundary** — the test that we got the interface right.

## First-build scope (one implementation plan)

IN: Tier-C bearing-conditioned DQN+HER reacher; `bearing_to` grounding stub with
noise; un-gameable greedy eval; (env tweak) randomized robot spawn.
OUT: the LLM planner (A), the real detector, B's search/memory, continuous
actions, metric SLAM.

## Recommended environment tweaks (simpler / better training)

1. **Randomized robot spawn per reset (highest value).** The fixed spawn created a
   strong positional prior that confounded earlier results (a do-nothing prior
   "located" goals 79% of the time) and hurts generalization. Randomizing start
   tile + heading removes it and is the single biggest training-quality win.
2. **Expose doorway/passage points as queryable targets.** Landmark-hopping needs
   "head to the doorway"; if the env lists passage locations between rooms,
   `bearing_to(base, "doorway")` is trivial instead of inferred from the wall grid.
3. **Unified target registry** `env.unwrapped.targets -> {name: (x,y)}` covering
   fixtures *and* dynamic objects (trash, person), so `bearing_to` is uniform.
4. **Env visibility helper** `unwrapped.visible(x, y)` (FOV/viewport + occlusion)
   so the stub's gating matches the renderer's true occlusion exactly.
5. *(Optional, sim-to-real)* a forward-FOV-cone render option instead of the
   top-down omnidirectional viewport, to match a real forward camera.

## Open decisions (deferred, not blocking the first build)

- Coarse range bucket in the goal, or pure bearing? (Lean: start pure bearing.)
- Detector class assumption for the noise model: open-vocab (more dropout) vs fast
  closed-set (less). (Lean: model moderate dropout; revisit with the real detector.)
