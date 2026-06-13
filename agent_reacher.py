# agent_reacher.py
# SAC reacher with a shared conv encoder (SAC-AE, Yarats et al.). The encoder is
# shaped by BOTH the critic's TD gradient (task signal — recon ALONE is task-agnostic
# and goes blind to task-relevant detail, the world-model lesson) AND a reconstruction
# loss (stabilising regulariser). The ACTOR is detached from the encoder so the policy
# cannot destabilise the representation. Polar goal is fed alongside the embedding.
import os
import subprocess
import datetime
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard.writer import SummaryWriter

from models.encoder import Encoder, Decoder
from models.ssim_loss import ssim_loss
from models.goal_actor import GoalActor
from models.goal_critic import GoalCritic
from goal_buffer import GoalHERBuffer
from goal_manager import GoalManager
from goal_geometry import GOAL_DIM, GOAL_RADIUS, distance

EMBED_DIM = 256


def _hard_update(target, source):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(s.data)


def _soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)


class ReacherAgent:
    def __init__(self, env, max_buffer_size=100000, alpha=0.05, tau=0.005,
                 gamma=0.97, start_radius=150.0, max_radius=600.0):
        self.env = env
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.alpha, self.tau, self.gamma = alpha, tau, gamma
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("runs", exist_ok=True)

        obs, _ = env.reset()
        self.input_shape = tuple(self.process_observation(obs).shape)  # (3,96,96)
        self.action_space = env.action_space
        self.n_actions = int(self.action_space.shape[0])
        hid = 256

        # Shared encoder (grounded by reconstruction) + decoder.
        self.encoder = Encoder(observation_shape=self.input_shape, embed_dim=EMBED_DIM).to(self.device)
        self.decoder = Decoder(observation_shape=self.input_shape, embed_dim=EMBED_DIM,
                               conv_output_shape=self.encoder.get_output_shape(),
                               conv_channels=self.encoder.get_conv_channels()).to(self.device)
        # Target encoder for the critic's bootstrap (soft-updated, like critic_target).
        self.encoder_target = Encoder(observation_shape=self.input_shape, embed_dim=EMBED_DIM).to(self.device)
        _hard_update(self.encoder_target, self.encoder)

        self.actor = GoalActor(EMBED_DIM, GOAL_DIM, self.n_actions, hid, self.action_space).to(self.device)
        self.critic = GoalCritic(EMBED_DIM, GOAL_DIM, self.n_actions, hid).to(self.device)
        self.critic_target = GoalCritic(EMBED_DIM, GOAL_DIM, self.n_actions, hid).to(self.device)
        _hard_update(self.critic_target, self.critic)

        self.encoder_optim = Adam(self.encoder.parameters(), lr=1e-3)
        self.decoder_optim = Adam(self.decoder.parameters(), lr=1e-3)
        self.actor_optim = Adam(self.actor.parameters(), lr=3e-5)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-4)

        self.memory = GoalHERBuffer(max_buffer_size, self.input_shape, self.device,
                                    self.n_actions, her_prob=0.0)
        self.goals = GoalManager(radius_px=start_radius)
        self.max_radius = max_radius

    def process_observation(self, obs):
        obs = cv2.resize(obs, (96, 96), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(obs).permute(2, 0, 1)

    @torch.no_grad()
    def _encode(self, img_t):
        return self.encoder(img_t)

    def _act(self, img_t, goal_t, evaluate):
        embed = self._encode(img_t)
        with torch.no_grad():
            a, _, mean = self.actor.sample(embed, goal_t)
        out = mean if evaluate else a
        return out.detach().cpu().numpy()[0]

    def warmup_action(self):
        if np.random.random() < 0.5:
            return np.array([np.random.uniform(0.6, 1.0), np.random.uniform(-0.3, 0.3)], np.float32)
        return np.array([np.random.uniform(0.3, 0.8), np.random.uniform(-1.0, 1.0)], np.float32)

    def _goal_tensor(self, base):
        return torch.as_tensor(self.goals.goal_vector(base)).unsqueeze(0).to(self.device)

    def train_step(self, batch_size):
        img_s, goal_s, action, reward, img_ns, goal_ns, done = self.memory.sample(batch_size, self.gamma)
        img_s = (img_s / 255.0).to(self.device)
        img_ns = (img_ns / 255.0).to(self.device)
        reward = reward.unsqueeze(1).to(self.device)
        done = done.unsqueeze(1).to(self.device)

        # ---- Critic: TD gradient flows INTO the encoder (task-shapes the representation) ----
        embed = self.encoder(img_s)
        with torch.no_grad():
            next_embed = self.encoder_target(img_ns)
            na, nlogp, _ = self.actor.sample(next_embed, goal_ns)
            q1t, q2t = self.critic_target(next_embed, goal_ns, na)
            min_q = torch.min(q1t, q2t) - self.alpha * nlogp
            target_q = reward + (1.0 - done) * self.gamma * min_q
        q1, q2 = self.critic(embed, goal_s, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_optim.zero_grad(); self.encoder_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        self.critic_optim.step(); self.encoder_optim.step()

        # ---- Autoencoder: recon GROUNDS/regularises the encoder (more than TD alone) ----
        embed_ae = self.encoder(img_s)
        recon = self.decoder(embed_ae)
        recon_loss = F.l1_loss(recon, img_s) + 0.2 * ssim_loss(recon, img_s)
        self.encoder_optim.zero_grad(); self.decoder_optim.zero_grad()
        recon_loss.backward()
        self.encoder_optim.step(); self.decoder_optim.step()

        # ---- Actor: encoder DETACHED (policy must not destabilise the representation) ----
        with torch.no_grad():
            embed_d = self.encoder(img_s)
        pi, logp, _ = self.actor.sample(embed_d, goal_s)
        q1pi, q2pi = self.critic(embed_d, goal_s, pi)
        actor_loss = (self.alpha * logp - torch.min(q1pi, q2pi)).mean()
        self.actor_optim.zero_grad(); actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0); self.actor_optim.step()

        _soft_update(self.critic_target, self.critic, self.tau)
        _soft_update(self.encoder_target, self.encoder, self.tau)
        return critic_loss.item(), actor_loss.item(), recon_loss.item()

    def greedy_eval(self, episodes=50, max_steps=300):
        self.actor.eval(); self.encoder.eval()
        reached = 0
        for _ in range(episodes):
            obs, _ = self.env.reset()
            base = self.env.unwrapped
            self.goals.reset(base)
            obs = self.process_observation(obs)
            for _ in range(max_steps):
                img_t = (obs.unsqueeze(0).float() / 255.0).to(self.device)
                action = self._act(img_t, self._goal_tensor(base), evaluate=True)
                nobs, _, _, trunc, _ = self.env.step(action)
                obs = self.process_observation(nobs)
                if distance(base._robot.x, base._robot.y, *self.goals.goal_px) < GOAL_RADIUS:
                    reached += 1
                    break
                if trunc:
                    break
        self.actor.train(); self.encoder.train()
        return reached / episodes

    def save(self):
        self.actor.save_the_model("goal_actor")
        self.critic.save_the_model("goal_critic")
        self.encoder.save_the_model("goal_encoder")

    def train(self, episodes=2000, max_steps=300, batch_size=256, warmup_episodes=10,
              grad_steps=50, eval_every=25, run_tag=None):
        if run_tag is None:
            try:
                refs = subprocess.check_output(
                    ['git', 'for-each-ref', '--format=%(refname:short)', '--points-at', 'HEAD',
                     'refs/remotes/origin/'], stderr=subprocess.DEVNULL).decode().strip()
                run_tag = (refs.splitlines()[0].replace('origin/', '') if refs else
                           subprocess.check_output(['git', 'branch', '--show-current']).decode().strip()) or 'unknown'
            except Exception:
                run_tag = 'unknown'
        writer = SummaryWriter(f'runs/{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}_{run_tag}')
        best_greedy = -1.0

        for episode in range(episodes):
            obs, _ = self.env.reset()
            base = self.env.unwrapped
            self.goals.reset(base)
            obs = self.process_observation(obs)
            ep_reaches = 0

            for _ in range(max_steps):
                rx, ry, rth = base._robot.x, base._robot.y, base._robot.angle
                if episode < warmup_episodes:
                    action = self.warmup_action()
                else:
                    img_t = (obs.unsqueeze(0).float() / 255.0).to(self.device)
                    action = self._act(img_t, self._goal_tensor(base), evaluate=False)

                nobs, _, _, trunc, _ = self.env.step(action)
                nobs_t = self.process_observation(nobs)
                nrx, nry, nrth = base._robot.x, base._robot.y, base._robot.angle
                reached = distance(nrx, nry, *self.goals.goal_px) < GOAL_RADIUS

                self.memory.store(obs, action, nobs_t, rx, ry, rth, nrx, nry, nrth,
                                  self.goals.goal_px, reached or trunc)
                obs = nobs_t
                if reached:
                    ep_reaches += 1
                # Single-goal episodes: end on reach or trunc (grounding).
                if reached or trunc:
                    break

            closs = aloss = rloss = 0.0
            if episode >= warmup_episodes and self.memory.can_sample(batch_size):
                for _ in range(grad_steps):
                    closs, aloss, rloss = self.train_step(batch_size)
                writer.add_scalar("SAC/critic_loss", closs, episode)
                writer.add_scalar("SAC/actor_loss", aloss, episode)
                writer.add_scalar("AE/recon_loss", rloss, episode)
            writer.add_scalar("Train/reaches_per_episode", ep_reaches, episode)

            if episode % eval_every == 0 and episode >= warmup_episodes:
                gr = self.greedy_eval()
                writer.add_scalar("Eval/greedy_reach_rate", gr, episode)
                writer.add_scalar("Curriculum/radius_px", self.goals.radius_px, episode)
                print(f"Episode {episode} | greedy reach-rate: {gr:.2f} | radius: {self.goals.radius_px:.0f}", flush=True)
                if gr > best_greedy:
                    best_greedy = gr
                    self.actor.save_the_model("goal_actor_best", verbose=True)
                    self.encoder.save_the_model("goal_encoder_best", verbose=True)
                if gr >= 0.8 and self.goals.radius_px < self.max_radius:
                    self.goals.set_radius(min(self.max_radius, self.goals.radius_px + 75.0))

            if episode % 50 == 0:
                self.save()
