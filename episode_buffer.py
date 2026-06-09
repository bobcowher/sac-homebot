from dataclasses import dataclass
from typing import Any
import torch


@dataclass
class Transition:
    obs:           torch.Tensor
    action:        int
    reward:        float
    next_obs:      torch.Tensor
    done:          bool
    achieved_goal: Any  # environment-specific; filled in by agent


class EpisodeBuffer:
    """Caches one episode's transitions for HER relabeling.

    Usage:
        # each step:
        episode_buffer.store(obs, action, reward, next_obs, done, achieved_goal)

        # end of episode:
        episode_buffer.flush_to(replay_buffer, desired_goal)
        episode_buffer.clear()
    """

    def __init__(self):
        self._transitions: list[Transition] = []

    def store(self, obs, action, reward, next_obs, done, achieved_goal):
        self._transitions.append(Transition(
            obs=obs,
            action=action,
            reward=float(reward),
            next_obs=next_obs,
            done=bool(done),
            achieved_goal=achieved_goal,
        ))

    def __len__(self):
        return len(self._transitions)

    def clear(self):
        self._transitions.clear()

    def flush_to(self, replay_buffer, desired_goal):  # noqa: ARG002
        """Store original transitions, then hindsight-relabeled transitions.

        TODO: implement goal relabeling once the environment exposes goal info.
              Steps:
              1. Store each original transition as-is (reward from env).
              2. For each step i, sample a hindsight goal from achieved_goals[i+1:].
                 (strategy='future' — any achieved goal strictly after step i)
              3. Recompute reward using hindsight goal.
              4. Store the relabeled transition.
        """
        # --- original transitions ---
        for t in self._transitions:
            replay_buffer.store_transition(t.obs, t.action, t.reward, t.next_obs, t.done)

        # --- hindsight transitions (TODO) ---
        # for i, t in enumerate(self._transitions):
        #     future = self._transitions[i + 1:]
        #     if not future:
        #         continue
        #     hindsight_goal = random.choice(future).achieved_goal
        #     hindsight_reward = compute_reward(t.next_obs, hindsight_goal)
        #     replay_buffer.store_transition(
        #         t.obs, t.action, hindsight_reward, t.next_obs, t.done
        #     )
