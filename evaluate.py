"""Greedy evaluation of downloaded checkpoints on HomeBot2D-V1 (trash task).

Runs N episodes per checkpoint and reports trash collected per episode plus
the full-clear rate (all trash collected = success). Env config matches
train.py exactly.

Usage:
    python3 evaluate.py                  # 100 greedy episodes, q_model.pt
    python3 evaluate.py --checkpoints checkpoints/q_model_best.pt
    python3 evaluate.py --epsilon 0.1    # reproduce training conditions
"""

import argparse
import random

import cv2
import gymnasium as gym
import torch

import homebot  # noqa: F401  (side-effect env registration)
from models.q_model import QModel

N_TRASH = 2


def make_env():
    for env_id in ("HomeBot2D-v1", "HomeBot2D-V1"):
        try:
            env = gym.make(
                env_id,
                render_mode="rgb_array",
                action_mode="discrete",
                obs_resolution=(96, 96),
                n_trash=N_TRASH,
                max_steps=1000,
                map_name="default",
                goals=["trash"],
            )
            print(f"Env: {env_id}")
            return env
        except gym.error.Error:
            continue
    raise RuntimeError("No HomeBot2D env id registered")


def load_q_model(path, n_actions, device):
    state = torch.load(path, map_location=device)
    if "q_model" in state:
        state = state["q_model"]
    model = QModel(action_dim=n_actions).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def process_observation(obs):
    obs = cv2.resize(obs, (96, 96), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(obs).permute(2, 0, 1)


def evaluate(model, env, episodes, device, epsilon=0.0):
    full_clears = 0
    total_trash = 0.0

    for episode in range(episodes):
        raw_obs, _ = env.reset()
        obs = process_observation(raw_obs)

        done = False
        steps = 0
        episode_reward = 0.0

        while not done:
            if epsilon > 0 and random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_t = obs.unsqueeze(0).float().to(device) / 255.0
                    action = model(obs_t).argmax(dim=1).item()
            raw_next, reward, term, trunc, _ = env.step(action)
            obs = process_observation(raw_next)
            done = term or trunc
            episode_reward += float(reward)
            steps += 1

        total_trash += episode_reward
        if episode_reward >= N_TRASH:
            full_clears += 1
        print(f"Episode {episode} | trash: {episode_reward:.1f} | steps: {steps}")

    pct = 100.0 * full_clears / episodes
    avg = total_trash / episodes
    print(f"\nFull clears: {full_clears}/{episodes} = {pct:.0f}% | avg trash/episode: {avg:.2f}")
    return pct, avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--checkpoints", nargs="+",
        default=["checkpoints/q_model.pt"],
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.0,
        help="random-action rate; 0.1 reproduces training conditions",
    )
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = make_env()
    n_actions = env.action_space.n  # type: ignore[union-attr]

    results = {}
    for path in args.checkpoints:
        print(f"\n=== {path} | {args.episodes} episodes | epsilon={args.epsilon} ===")
        model = load_q_model(path, n_actions, device)
        results[path] = evaluate(model, env, args.episodes, device, args.epsilon)

    print("\n=== Summary ===")
    for path, (pct, avg) in results.items():
        print(f"{path}: {pct:.0f}% full clears, {avg:.2f} avg trash")


if __name__ == "__main__":
    main()
