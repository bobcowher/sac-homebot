"""Greedy evaluation of bearing-reacher checkpoints.

Runs N budget-limited, random-spawn greedy episodes (epsilon=0) per checkpoint
and reports the reach rate. The budget per episode is computed from the initial
distance so a circling policy cannot game the metric.

Usage:
    python3 evaluate.py                     # 100 episodes, q_model.pt + best.pt
    python3 evaluate.py --episodes 20
    python3 evaluate.py --checkpoints checkpoints/q_model_best.pt
"""

import argparse
import math
import random

import cv2
import gymnasium as gym
import numpy as np
import torch

import homebot  # noqa: F401  (side-effect env registration)
from goal_geometry import bearing as compute_bearing, distance, eval_step_budget, GOAL_RADIUS
from models.q_model import QModel


def make_env(max_steps=500):
    # Local homebot registers -V1 (capital), remote registers -v1.
    for env_id in ("HomeBot2D-Goal-v1", "HomeBot2D-Goal-V1"):
        try:
            env = gym.make(
                env_id,
                render_mode="rgb_array",
                action_mode="discrete",
                obs_resolution=(96, 96),
                n_trash=2,
                max_steps=max_steps,
                map_name="default",
                goals=["collect_trash"],
            )
            print(f"Env: {env_id}")
            return env
        except gym.error.Error:
            continue
    raise RuntimeError("No HomeBot2D-Goal env id registered")


def load_q_model(path, n_actions, device):
    state = torch.load(path, map_location=device)
    if "q_model" in state:  # best.pt wraps the state_dict with metadata
        ep = state.get("episode", "?")
        rr = state.get("reach_rate", float("nan"))
        print(f"  ({path} is best-checkpoint from episode {ep}, reach_rate={rr:.3f})")
        state = state["q_model"]
    model = QModel(action_dim=n_actions, goal_scale=(1.0, 1.0)).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def process_observation(obs):
    obs = cv2.resize(obs, (96, 96), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(obs).permute(2, 0, 1)


def random_spawn(env):
    """Teleport robot to random valid tile/heading, return fresh full obs dict."""
    base = env.unwrapped
    tiles = base._map.valid_floor_tiles()
    tx, ty = random.choice(tiles)
    px, py = base._map.tile_to_pixel(tx, ty)
    base._robot.x = px
    base._robot.y = py
    base._robot.angle = random.uniform(-math.pi, math.pi)
    return base._build_obs()


def evaluate(model, env, episodes, device):
    """Budget-limited greedy eval with random spawn. Returns reach rate [0, 1]."""
    successes = 0
    budgets_used = []

    for episode in range(episodes):
        env.reset()
        fresh        = random_spawn(env)
        obs          = process_observation(fresh["observation"])
        desired_goal = fresh["desired_goal"]
        base         = env.unwrapped
        r            = base._robot

        init_dist = distance(r.x, r.y, desired_goal[0], desired_goal[1])
        budget    = eval_step_budget(init_dist)

        reached = False
        steps_used = 0
        for _ in range(budget):
            goal_bearing = compute_bearing(r.x, r.y, r.angle,
                                           desired_goal[0], desired_goal[1])
            with torch.no_grad():
                obs_t  = obs.unsqueeze(0).float().to(device) / 255.0
                goal_t = torch.as_tensor(goal_bearing, dtype=torch.float32,
                                         device=device).unsqueeze(0)
                action = model(obs_t, goal_t).argmax(dim=1).item()

            raw_next, _, term, trunc, _ = env.step(action)
            obs = process_observation(raw_next["observation"])
            steps_used += 1

            if distance(r.x, r.y, desired_goal[0], desired_goal[1]) < GOAL_RADIUS:
                reached = True
                break
            if term or trunc:
                break

        if reached:
            successes += 1
            budgets_used.append(steps_used)

        print(f"Episode {episode} | reached={reached} | steps={steps_used}/{budget} "
              f"| init_dist={init_dist:.0f}")

    rate = successes / episodes
    avg_steps = sum(budgets_used) / len(budgets_used) if budgets_used else float("nan")
    print(f"\nReach rate: {successes}/{episodes} = {rate:.3f} "
          f"| avg steps on success: {avg_steps:.0f}")
    return rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--checkpoints", nargs="+",
        default=["checkpoints/q_model.pt", "checkpoints/q_model_best.pt"],
    )
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = make_env()
    n_actions = env.action_space.n  # type: ignore[union-attr]

    results = {}
    for path in args.checkpoints:
        import os
        if not os.path.exists(path):
            print(f"\n=== Skipping {path} (not found) ===")
            continue
        print(f"\n=== {path} | {args.episodes} episodes ===")
        model = load_q_model(path, n_actions, device)
        results[path] = evaluate(model, env, args.episodes, device)

    print("\n=== Summary ===")
    for path, rate in results.items():
        print(f"{path}: {rate:.3f} ({rate * 100:.0f}%)")


if __name__ == "__main__":
    main()
