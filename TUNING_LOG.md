# Q-Homebot Tuning Log

Overnight autonomous tuning session. Goal: get HER + Double-DQN agent on
`HomeBot2D-Goal-v1` (`collect_trash`, n_trash=2, max_steps=1000) to solve the
task **consistently** (env max reward = 1.0 per episode → success = consistent 1s).

Method: change one variable at a time. Kick off a Beekeeper run with a clear
TensorBoard tag, watch it, compare to baseline. If it doesn't beat baseline,
**roll it back** before trying the next lever. Each run is tagged via the
remote branch name, so tuning variants live on dedicated `tuning-*` branches
(or are noted here when the tag is reused).

Single GPU, parallel runs disabled → one run at a time. Baseline TB data is
retained (tb_logs_max_runs=10) for comparison.

**Branch hygiene:** each experiment gets a `tuning-*` branch (drives the TB
tag). When an experiment concludes — loser: roll code back + delete local and
remote branch; winner: fold into `tuning` + delete the experiment branch. Goal
is at most one or two live `tuning-*` branches, not a graveyard. Cleanup status
tracked per experiment below.

### Branch cleanup ledger
- `tuning-huber` (run 224): WON → merged into `tuning`, branch deleted.
- `tuning-gamma` (run 225): NEUTRAL → reverted, branch deleted (not merged).
- `tuning-herk` (run 226): REGRESSION → reverted, branch deleted (not merged).
- `tuning-polyak` (run 227): REGRESSION → reverted, branch deleted (not merged).
- `tuning-lr` (Exp 5): live — pending Exp 5 verdict.

---

## Baseline — Run 223 (`her` branch)

Config: lr=1e-4, MSE loss, 800 grad-steps/episode, batch=64, gamma=0.99,
epsilon 1.0→0.1 decay 0.977 (min at ep ~100), hard target update every 1000
steps, HER K=4 (future strategy).

Result after ~530 episodes:
- `Train/best_score` = **1.0** — architecture is sound, the goal is reachable.
- `Train/episode_reward` smoothed ≈ **0.26**, peak ≈ 0.47 — ~1-in-4 success.
- Successes are fast (1–120 steps); failures burn the full 1000 steps (reward 0).
- `Train/avg_q_loss` **worsening / unstable** — spikes to 2000–4600 despite
  grad-norm clip at 1.0. Q-values are diverging. Suspected cap on the policy:
  exploding Q estimates → noisy argmax → inconsistent success.

**Baseline number to beat: smoothed episode_reward ≈ 0.26.**

---

## Experiments

### Exp 1 — Huber (smooth_l1) loss instead of MSE
- **Hypothesis:** MSE squares large TD errors, producing the 2000–4600 loss
  spikes and Q divergence. Huber is linear past delta=1, the textbook DQN fix
  for exactly this symptom. Should stabilize Q and lift/steady success rate.
- **Change:** `agent.py` train_step — `F.mse_loss` → `F.smooth_l1_loss`.
- **Tag/branch:** `tuning-huber` (TB tag derived from branch name).
- **Run:** 224 (baseline 223 stopped to free the GPU; its TB data retained).
  Completed full 1000 episodes in 33 min.
- **Result vs baseline:**
  - Q-loss smoothed **69 → 3.15**, spikes **4600 → 186**. Divergence solved.
  - episode_reward peak **0.47 → 0.57**; actual recent success rate ~30–33%
    (counted from log tail) vs baseline ~26%. (EMA `smoothed_final`=0.147 was
    depressed by a cold streak in the last 9 episodes — ignore it; use the tail
    count and peak.)
- **Verdict: WIN — KEEP.** Folded into `tuning`. Huber stabilizes Q and modestly
  lifts success. Stability headroom now enables raising gamma (Exp 2).

### Exp 2 — gamma 0.99 → 0.995 (long-horizon value propagation)
- **Hypothesis:** failures always burn all 1000 steps because distant goals are
  invisible: `0.99^1000 ≈ 4e-5`. Raising gamma to 0.995 (`0.995^200=0.37` vs
  `0.99^200=0.13`) propagates value further so the agent can "see" far trash.
  Huber + grad-clip should contain the larger targets.
- **Change:** `agent.py` `self.gamma = 0.99` → `0.995`. (On top of Huber.)
- **Tag/branch:** `tuning-gamma`.
- **Run:** 225 — completed full 1000 episodes (34 min).
- **Result vs Exp 1 (Huber, ~32%):**
  - success rate ~34–35% (tail count) vs ~32%; peak episode_reward 0.61 vs 0.57.
    Marginal, within episode-to-episode noise.
  - Q-loss smoothed 3.15 → **5.23**, spikes 186 → **369** — noisier (expected:
    larger targets), still far better than pre-Huber baseline.
  - Dominant failure mode unchanged: ~2/3 of episodes still burn all 1000 steps.
    Gamma alone did not crack long-horizon navigation.
- **Verdict: NEUTRAL → REVERT.** Gain within noise, costs Q-stability and muddies
  the story. Rolled back to gamma 0.99 (clean Huber baseline on `tuning`).

### Exp 3 — HER K 4 → 8 (denser hindsight signal for navigation)
- **Hypothesis:** the failure mode is reaching *distant* goals. HER's whole job is
  to teach "how to get to arbitrary positions" by relabeling. More hindsight goals
  per transition (K 8 vs 4) doubles the relabeled navigation signal per episode,
  which should lift the success rate where gamma couldn't. Single HER variable.
- **Change:** `episode_buffer.py` `K = 4` → `K = 8`. (On clean Huber baseline.)
- **Tag/branch:** `tuning-herk`.
- **Run:** 226 — completed full 1000 episodes (38 min).
- **Result vs Exp 1 (Huber, ~32%):**
  - success rate ~21% (tail count) vs ~32%; peak episode_reward 0.45 vs 0.57.
  - Longer cold streaks (e.g. ~28 consecutive failures, ep 881–909).
  - Q-loss smoothed 3.15 → 8.70 (worse).
- **Verdict: REGRESSION → REVERT.** Doubling relabels floods the buffer with
  easy near-future goals, shrinking the share of real desired-goal transitions →
  policy overfits trivial nearby goals, generalizes worse to distant trash.
  Rolled back to K=4 (clean Huber baseline on `tuning`).

### Exp 4 — Polyak soft target updates (replace hard update every 1000 steps)
- **Hypothesis:** three levers in, the wall is policy *instability across
  episodes* (cold streaks), not value reach or signal density. Hard target sync
  every 1000 grad-steps (≈ once/episode with 800/ep) makes the bootstrap target
  lurch once per episode. Polyak (tau=0.005, every step) tracks the online net
  smoothly → steadier targets → fewer cold streaks, higher floor. Pairs with the
  Huber stability story.
- **Change:** `agent.py` — add `self.tau = 0.005`; in train_step replace the
  `total_steps % target_update_interval` hard copy with a per-step soft update
  `tp = (1-tau)*tp + tau*p`. (On clean Huber baseline, K=4, gamma 0.99.)
- **Tag/branch:** `tuning-polyak`.
- **Run:** 227 — completed full 1000 episodes (36 min).
- **Result vs Exp 1 (Huber, ~32%):**
  - success rate ~17–18% (tail count) vs ~32%; peak episode_reward 0.49 vs 0.57.
  - Q-loss *noisier*: smoothed 6.68, spikes to 502 (vs 3.15 / 186).
  - Severe ~30-episode dry patch (ep 843–874).
- **Verdict: REGRESSION → REVERT.** With 800 grad-steps/episode, tau=0.005 every
  step tracks the online net ~98%/episode — *tighter* coupling than the hard sync,
  so it amplified the target-chasing feedback instead of damping it. Rolled back to
  hard target update. Lesson: this system is over-aggressive on updates.

### Exp 5 — learning rate 1e-4 → 5e-5 (calm the over-aggressive updates)
- **Hypothesis:** Polyak made things worse by tightening update coupling → the
  system is over-stepping. The persistent signature across all runs is policy
  thrashing (cold streaks) on a noisy ~18–35% band. Halving the LR, now safe under
  stable Huber loss, should let the policy settle into a steadier, higher floor.
- **Change:** `agent.py` Adam `lr=0.0001` → `lr=0.00005`. (Clean Huber baseline.)
- **Tag/branch:** `tuning-lr`.
- **Run:** 228.
- **Status:** RUNNING — compare success rate vs Exp 1 (~32%).

> **Meta-note:** n=1 run per config; the 18–35% reward band is partly seed noise.
> Huber is the one robust win (Q-loss 69→3 is consistent, not noise). If LR also
> lands within the band, treat ~32% as a noise-limited plateau — the remaining
> ceiling is likely representational/exploration, not a single hyperparameter.
