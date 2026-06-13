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
