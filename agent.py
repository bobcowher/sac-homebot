import os
import math
import random
import subprocess
import datetime

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter

from buffer import ReplayBuffer
from episode_buffer import EpisodeBuffer
from goal_geometry import bearing as compute_bearing, distance, eval_step_budget, GOAL_RADIUS
from models.q_model import QModel


class Agent:

    def __init__(self, env: gym.Env,
                       max_buffer_size: int = 100000,
                       target_update_interval: int = 1000) -> None:
        self.env = env
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("runs", exist_ok=True)

        raw_obs, _ = self.env.reset()
        obs = self.process_observation(raw_obs["observation"])

        self.n_actions = self.env.action_space.n  # type: ignore[union-attr]

        self.memory = ReplayBuffer(
            max_size=max_buffer_size,
            input_shape=obs.shape,
            input_device=self.device,
            output_device=self.device,
        )

        # Bearing is already in [-1, 1]; pass goal_scale=(1,1) so encode_goal
        # is a no-op division (divides by 1.0 instead of raw pixel extents).
        self.q_model = QModel(
            action_dim=self.n_actions,
            input_shape=obs.shape,
            goal_scale=(1.0, 1.0),
        ).to(self.device)

        self.target_q_model = QModel(
            action_dim=self.n_actions,
            input_shape=obs.shape,
            goal_scale=(1.0, 1.0),
        ).to(self.device)
        self.target_q_model.load_state_dict(self.q_model.state_dict())

        self.q_optimizer = torch.optim.Adam(self.q_model.parameters(), lr=0.00005)

        self.gamma = 0.99
        self.epsilon = 1.0
        self.min_epsilon = 0.1
        self.epsilon_decay = 0.977

        self.target_update_interval = target_update_interval
        self.total_steps = 0
        self.episode_buffer = EpisodeBuffer()

        self.best_reach_rate = -1.0

    def process_observation(self, obs):
        obs = cv2.resize(obs, (96, 96), interpolation=cv2.INTER_NEAREST)
        obs = torch.from_numpy(obs).permute(2, 0, 1)
        return obs

    def _robot_bearing(self, desired_goal):
        """Compute egocentric bearing to desired_goal from current robot pose."""
        base = self.env.unwrapped
        r = base._robot
        return compute_bearing(r.x, r.y, r.angle, desired_goal[0], desired_goal[1])

    def _random_spawn(self):
        """Teleport the robot to a random valid floor tile and random heading.

        After mutating the robot pose, we MUST re-fetch a fresh observation from
        _build_obs() — the stale obs from env.reset() still shows the old pose.
        Returns the fresh full dict {observation, achieved_goal, desired_goal}.
        """
        base = self.env.unwrapped
        tiles = base._map.valid_floor_tiles()
        tx, ty = random.choice(tiles)
        px, py = base._map.tile_to_pixel(tx, ty)
        base._robot.x = px
        base._robot.y = py
        base._robot.angle = random.uniform(-math.pi, math.pi)
        return base._build_obs()

    def select_action(self, obs, goal_bearing):
        """Select action given obs and 2-D bearing goal vector."""
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            obs_t  = obs.unsqueeze(0).float().to(self.device) / 255.0
            goal_t = torch.as_tensor(goal_bearing, dtype=torch.float32,
                                     device=self.device).unsqueeze(0)
            return self.q_model(obs_t, goal_t).argmax(dim=1).item()

    def _greedy_action(self, obs, goal_bearing):
        """Pure greedy (no epsilon) action selection for eval."""
        with torch.no_grad():
            obs_t  = obs.unsqueeze(0).float().to(self.device) / 255.0
            goal_t = torch.as_tensor(goal_bearing, dtype=torch.float32,
                                     device=self.device).unsqueeze(0)
            return self.q_model(obs_t, goal_t).argmax(dim=1).item()

    def train_step(self, batch_size):
        obs, actions, rewards, next_obs, dones, goals, next_goals = self.memory.sample_buffer(batch_size)

        obs      = obs      / 255.0
        next_obs = next_obs / 255.0

        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones   = dones.unsqueeze(1).float()

        q_sa = self.q_model(obs, goals).gather(1, actions)

        with torch.no_grad():
            next_actions = self.q_model(next_obs, next_goals).argmax(dim=1, keepdim=True)
            next_q       = self.target_q_model(next_obs, next_goals).gather(1, next_actions)
            targets      = rewards + (1 - dones) * self.gamma * next_q

        loss = F.smooth_l1_loss(q_sa, targets)

        self.q_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_model.parameters(), max_norm=1.0)
        self.q_optimizer.step()

        if self.total_steps % self.target_update_interval == 0:
            self.target_q_model.load_state_dict(self.q_model.state_dict())

        self.total_steps += 1
        return loss.item()

    def save(self):
        self.q_model.save_the_model("q_model", verbose=True)

    def save_best(self, episode, reach_rate):
        path = "checkpoints/q_model_best.pt"
        torch.save({
            "q_model": self.q_model.state_dict(),
            "episode": episode,
            "reach_rate": reach_rate,
        }, path)
        print(f"  New best checkpoint saved (episode={episode}, reach_rate={reach_rate:.3f})")

    def load(self):
        self.q_model.load_the_model("q_model", device=self.device)
        self.target_q_model.load_state_dict(self.q_model.state_dict())

    def greedy_eval(self, n_episodes: int = 20) -> float:
        """Un-gameable greedy evaluation.

        Per episode:
          1. Random spawn (same as training).
          2. Compute step budget from initial distance.
          3. Run greedy (argmax, epsilon=0) for at most budget steps.
          4. Success = distance to goal < GOAL_RADIUS at any point (or at end).

        Returns reach-rate in [0, 1].
        """
        self.q_model.eval()
        successes = 0

        for _ in range(n_episodes):
            raw_obs, _ = self.env.reset()
            fresh = self._random_spawn()

            obs          = self.process_observation(fresh["observation"])
            desired_goal = fresh["desired_goal"]
            base         = self.env.unwrapped
            r            = base._robot

            init_dist = distance(r.x, r.y, desired_goal[0], desired_goal[1])
            budget    = eval_step_budget(init_dist)

            reached = False
            for _ in range(budget):
                goal_bearing = compute_bearing(r.x, r.y, r.angle,
                                               desired_goal[0], desired_goal[1])
                action = self._greedy_action(obs, goal_bearing)
                raw_next, _, term, trunc, _ = self.env.step(action)
                obs = self.process_observation(raw_next["observation"])

                if distance(r.x, r.y, desired_goal[0], desired_goal[1]) < GOAL_RADIUS:
                    reached = True
                    break
                if term or trunc:
                    break

            if reached:
                successes += 1

        self.q_model.train()
        return successes / n_episodes

    def train(self, episodes=3000, batch_size=64, run_tag=None,
              eval_interval=50, eval_episodes=20):
        if run_tag is None:
            try:
                refs = subprocess.check_output(
                    ['git', 'for-each-ref', '--format=%(refname:short)',
                     '--points-at', 'HEAD', 'refs/remotes/origin/'],
                    stderr=subprocess.DEVNULL).decode().strip()
                if refs:
                    run_tag = refs.splitlines()[0].replace('origin/', '')
                if not run_tag:
                    run_tag = subprocess.check_output(
                        ['git', 'branch', '--show-current'],
                        stderr=subprocess.DEVNULL).decode().strip()
                if not run_tag:
                    run_tag = 'unknown'
            except Exception:
                run_tag = 'unknown'

        writer = SummaryWriter(
            f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{run_tag}')

        for episode in range(episodes):
            # Reset env, then random-spawn the robot
            raw_obs, _ = self.env.reset()
            fresh = self._random_spawn()

            obs          = self.process_observation(fresh["observation"])
            desired_goal = fresh["desired_goal"]
            base         = self.env.unwrapped
            r            = base._robot

            done = False
            episode_reward = 0.0
            episode_loss   = 0.0
            episode_steps  = 0

            while not done:
                heading_prev  = r.angle
                pos_prev      = np.array([r.x, r.y], dtype=np.float32)
                goal_bearing  = compute_bearing(r.x, r.y, r.angle,
                                                desired_goal[0], desired_goal[1])
                action        = self.select_action(obs, goal_bearing)

                raw_next, reward, term, trunc, _ = self.env.step(action)
                next_obs      = self.process_observation(raw_next["observation"])
                heading_next  = r.angle
                pos_next      = np.array([r.x, r.y], dtype=np.float32)
                done          = term or trunc

                # Store term (not trunc): timeout is not a terminal state.
                self.episode_buffer.store(
                    obs, action, reward, next_obs, term,
                    achieved_prev=pos_prev,
                    achieved_next=pos_next,
                    heading_prev=heading_prev,
                    heading_next=heading_next,
                )
                episode_reward += float(reward)
                episode_steps  += 1
                obs = next_obs

            self.episode_buffer.send_to(
                self.memory,
                desired_goal=desired_goal,
                compute_reward=self.env.unwrapped.compute_reward,  # type: ignore[attr-defined]
            )
            self.episode_buffer.clear()

            for _ in range(800):
                if self.memory.can_sample(batch_size):
                    episode_loss += self.train_step(batch_size)

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            avg_loss = episode_loss / episode_steps if episode_steps > 0 else 0.0
            print(f"Episode {episode} | reward: {episode_reward:.1f} | "
                  f"epsilon: {self.epsilon:.3f} | steps: {episode_steps}")

            writer.add_scalar("Train/episode_reward", episode_reward, episode)
            writer.add_scalar("Train/epsilon",         self.epsilon,   episode)
            writer.add_scalar("Train/avg_q_loss",      avg_loss,       episode)
            writer.add_scalar("Train/episode_steps",   episode_steps,  episode)
            writer.add_scalar("Buffer/fill",
                              min(self.memory.mem_ctr, self.memory.mem_size), episode)

            if episode % 10 == 0:
                self.save()

            # Periodic greedy eval
            if episode % eval_interval == 0:
                reach_rate = self.greedy_eval(n_episodes=eval_episodes)
                writer.add_scalar("Eval/greedy_reach_rate", reach_rate, episode)
                print(f"  [Eval] episode {episode}: greedy reach_rate={reach_rate:.3f}")
                if reach_rate > self.best_reach_rate:
                    self.best_reach_rate = reach_rate
                    self.save_best(episode, reach_rate)

    def test(self, episodes=10):
        """Legacy test method (uses greedy policy, random spawn, budget)."""
        self.q_model.eval()
        total_rewards = []

        for episode in range(episodes):
            raw_obs, _ = self.env.reset()
            fresh        = self._random_spawn()
            obs          = self.process_observation(fresh["observation"])
            desired_goal = fresh["desired_goal"]
            base         = self.env.unwrapped
            r            = base._robot

            done = False
            episode_reward = 0.0

            while not done:
                goal_bearing = compute_bearing(r.x, r.y, r.angle,
                                               desired_goal[0], desired_goal[1])
                with torch.no_grad():
                    obs_t  = obs.unsqueeze(0).float().to(self.device) / 255.0
                    goal_t = torch.as_tensor(goal_bearing, dtype=torch.float32,
                                             device=self.device).unsqueeze(0)
                    action = self.q_model(obs_t, goal_t).argmax(dim=1).item()
                raw_next, reward, term, trunc, _ = self.env.step(action)
                next_obs = self.process_observation(raw_next["observation"])
                done = term or trunc
                episode_reward += float(reward)
                obs = next_obs

            total_rewards.append(episode_reward)
            print(f"Test episode {episode} | reward: {episode_reward:.1f}")

        avg = sum(total_rewards) / len(total_rewards)
        print(f"Average reward over {episodes} episodes: {avg:.1f}")
        self.q_model.train()
        return total_rewards
