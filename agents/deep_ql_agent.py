# deep_ql_agent.py
from __future__ import annotations
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2

# BaseAgent compatibility
try:
    from agents.base import BaseAgent
except Exception:
    class BaseAgent:
        def __init__(self, action_space, config=None): ...
        def get_action(self, observation): ...
        def update(self, state, action, reward, next_state, needs_to_reset): ...
        def save(self, filepath): ...
        def load(self, filepath): ...


# Observation conversion
def chw_from_obs(obs: np.ndarray) -> torch.Tensor:
    """
    Convert observation to CHW float tensor in [0, 1].

    Handles:
    - (4, 84, 84) or (1, 84, 84) uint8 (FrameStack + AtariPreprocessing)
    - (84, 84) grayscale
    - (H, W, 3) RGB → grayscale 84x84
    """
    arr = np.asarray(obs)

    # (4, 84, 84) or (1, 84, 84)
    if arr.ndim == 3 and arr.shape[-2:] == (84, 84):
        return torch.as_tensor(arr, dtype=torch.float32) / 255.0

    # (84, 84)
    if arr.ndim == 2:
        return torch.as_tensor(arr, dtype=torch.float32).unsqueeze(0) / 255.0

    # (H, W, 3)
    if arr.ndim == 3 and arr.shape[2] == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return torch.as_tensor(gray, dtype=torch.float32).unsqueeze(0) / 255.0

    raise ValueError(f"Unsupported observation shape: {arr.shape}")


# CNN Q-Network
class CNNQ(nn.Module):
    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            n_flat = self.conv(dummy).view(1, -1).size(1)

        self.head = nn.Sequential(
            nn.Linear(n_flat, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.head(x)


# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.idx = 0
        self.full = False

        self.s = [None] * capacity
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = [None] * capacity
        self.d = np.zeros(capacity, dtype=np.bool_)

    def push(self, s, a, r, s2, done):
        i = self.idx
        self.s[i] = np.asarray(s)
        self.a[i] = int(a)
        self.r[i] = float(r)
        self.s2[i] = np.asarray(s2)
        self.d[i] = bool(done)
        self.idx = (i + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def __len__(self):
        return self.capacity if self.full else self.idx

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, len(self), size=batch_size)
        s = torch.stack([chw_from_obs(self.s[i]) for i in idxs]).to(self.device)
        s2 = torch.stack([chw_from_obs(self.s2[i]) for i in idxs]).to(self.device)
        a = torch.as_tensor(self.a[idxs], device=self.device, dtype=torch.long)
        r = torch.as_tensor(self.r[idxs], device=self.device, dtype=torch.float32)
        d = torch.as_tensor(self.d[idxs], device=self.device, dtype=torch.float32)
        return s, a, r, s2, d


# DQL Agent
class DQLAgent(BaseAgent):
    def __init__(self, action_space, config):
        super().__init__(action_space, config)
        self.action_space = action_space
        self.cfg = config
        self.cfg.device = "cpu"
        self.device = torch.device("cpu")


        self.q: Optional[nn.Module] = None
        self.q_target: Optional[nn.Module] = None
        self.opt: Optional[optim.Optimizer] = None
        self.buffer: Optional[ReplayBuffer] = None
        self.loss_fn = nn.SmoothL1Loss()

        self.step_count = 0  # counts env steps

    # Act
    def get_action(self, observation):
        if self.q is None:
            self._init_from_obs(observation)

        eps = self._epsilon()

        # ε-greedy
        if random.random() < eps:
            return self.action_space.sample()

        x = chw_from_obs(observation).to(self.device)
        with torch.no_grad():
            q = self.q(x)
        return int(torch.argmax(q, dim=1).item())

    # Train
    def update(self, state, action, reward, next_state, done):
        if self.buffer is None:
            self._init_from_obs(state)
            
        self.step_count += 1

        self.buffer.push(state, action, reward, next_state, done)

        # Warmup
        if len(self.buffer) < self.cfg.learning_starts:
            return

        # Train only every train_freq steps
        if self.step_count % self.cfg.train_freq != 0:
            return

        s, a, r, s2, d = self.buffer.sample(self.cfg.batch_size)

        with torch.no_grad():
            if self.cfg.double_dqn:
                best_next = torch.argmax(self.q(s2), dim=1)
                q_next = self.q_target(s2).gather(1, best_next.unsqueeze(1)).squeeze(1)
            else:
                q_next = self.q_target(s2).max(1).values

            target = r + (1.0 - d) * self.cfg.gamma * q_next

        q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(q_sa, target)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.max_grad_norm)
        self.opt.step()

        # Target network update
        if self.step_count % self.cfg.target_update_freq == 0:
            self.q_target.load_state_dict(self.q.state_dict())

    # Initialization
    def _init_from_obs(self, obs):
        x = chw_from_obs(obs)
        c = x.shape[0]  # channels (should be 4 for MsPacman + FrameStack)

        self.q = CNNQ(c, self.action_space.n).to(self.device)
        self.q_target = CNNQ(c, self.action_space.n).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()

        self.opt = optim.AdamW(self.q.parameters(), lr=self.cfg.lr, amsgrad=True)
        self.buffer = ReplayBuffer(self.cfg.buffer_size, self.device)

    def _epsilon(self):
        """Exponential decay is more stable for Atari DQN."""
        step = self.step_count
        eps_start = self.cfg.eps_start
        eps_final = self.cfg.eps_final
        decay = self.cfg.eps_decay_steps

        return eps_final + (eps_start - eps_final) * np.exp(-step / decay)

    # Saving / loading (simple: weights only, CPU-friendly)
    def save(self, path: str):
        if self.q is None:
            raise RuntimeError("Cannot save before network is initialized.")
        torch.save(self.q.state_dict(), path)

    def load(self, path: str):
        # Fixed 4-channel input for MsPacman (FrameStack(4))
        obs_channels = 4
        self.q = CNNQ(obs_channels, self.action_space.n).to(self.device)
        self.q_target = CNNQ(obs_channels, self.action_space.n).to(self.device)

        state_dict = torch.load(path, map_location=self.device)
        self.q.load_state_dict(state_dict)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()
