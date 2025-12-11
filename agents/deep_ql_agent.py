from __future__ import annotations
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2

# try to import BaseAgent; fall back to a stub if not available
try:
    from agents.base import BaseAgent
except Exception:
    class BaseAgent:
        def __init__(self, action_space, config=None): ...
        def get_action(self, observation): ...
        def update(self, state, action, reward, next_state, needs_to_reset): ...
        def save(self, filepath): ...
        def load(self, filepath): ...


# convert observation to normalized CHW tensor
def chw_from_obs(obs: np.ndarray) -> torch.Tensor:
    """
    Convert observation to CHW float tensor in [0, 1].

    Handles:
    - (4, 84, 84) or (1, 84, 84) uint8 (FrameStack + AtariPreprocessing)
    - (84, 84) grayscale
    - (H, W, 3) RGB → grayscale 84x84
    """
    arr = np.asarray(obs)

    # stacked frames or single channel 
    # (4, 84, 84) or (1, 84, 84)
    if arr.ndim == 3 and arr.shape[-2:] == (84, 84):
        return torch.as_tensor(arr, dtype=torch.float32) / 255.0

    # single grayscale frame 
    # (84, 84)
    if arr.ndim == 2:
        return torch.as_tensor(arr, dtype=torch.float32).unsqueeze(0) / 255.0

    # RGB frame -> grayscale
    # (H, W, 3)
    if arr.ndim == 3 and arr.shape[2] == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return torch.as_tensor(gray, dtype=torch.float32).unsqueeze(0) / 255.0

    raise ValueError(f"Unsupported observation shape: {arr.shape}")


# CNN Q-Network for 84x84 image input
class CNNQ(nn.Module):
    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()
        # convolutional feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
        )
        # compute flattened size using dummy input
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            n_flat = self.conv(dummy).view(1, -1).size(1)

        # fully connected head to output Q-values
        self.head = nn.Sequential(
            nn.Linear(n_flat, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # allow single observation (C, H, W) or batch (B, C, H, W)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.head(x)


# simple replay bugger for experience replay 
class ReplayBuffer:
    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.idx = 0
        self.full = False

        # store raw numpy observations and transition data
        self.s = [None] * capacity
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = [None] * capacity
        self.d = np.zeros(capacity, dtype=np.bool_)

    # add one transition to the buffer
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

    # current number of stored transitions
    def __len__(self):
        return self.capacity if self.full else self.idx

    # sample a random batch and convert to tensors
    def sample(self, batch_size: int):
        idxs = np.random.randint(0, len(self), size=batch_size)
        s = torch.stack([chw_from_obs(self.s[i]) for i in idxs]).to(self.device)
        s2 = torch.stack([chw_from_obs(self.s2[i]) for i in idxs]).to(self.device)
        a = torch.as_tensor(self.a[idxs], device=self.device, dtype=torch.long)
        r = torch.as_tensor(self.r[idxs], device=self.device, dtype=torch.float32)
        d = torch.as_tensor(self.d[idxs], device=self.device, dtype=torch.float32)
        return s, a, r, s2, d


# DQL Agent (DQN for MsPacman with frame stacking)
class DQLAgent(BaseAgent):
    def __init__(self, action_space, config):
        super().__init__(action_space, config)
        self.action_space = action_space
        self.cfg = config
        
        # force cpu
        self.cfg.device = "cpu"
        self.device = torch.device("cpu")

        # networks / optimizer / buffer
        self.q: Optional[nn.Module] = None
        self.q_target: Optional[nn.Module] = None
        self.opt: Optional[optim.Optimizer] = None
        self.buffer: Optional[ReplayBuffer] = None
        
        # huber loss
        self.loss_fn = nn.SmoothL1Loss()

        # counts environemnt steps for scheduling
        self.step_count = 0 

    # choose action using ε-greedy policy
    def get_action(self, observation):
        # initialize networks on first observation
        if self.q is None:
            self._init_from_obs(observation)

        eps = self._epsilon()

        # exploratioon with probability ε
        if random.random() < eps:
            return self.action_space.sample()

        # exploitation: take argmax Q(s,a)
        x = chw_from_obs(observation).to(self.device)
        with torch.no_grad():
            q = self.q(x)
        return int(torch.argmax(q, dim=1).item())

    # store transition and update networks
    def update(self, state, action, reward, next_state, done):
        # initialize on first update if needed
        if self.buffer is None:
            self._init_from_obs(state)
            
        self.step_count += 1

        # add to replay buffer
        self.buffer.push(state, action, reward, next_state, done)

        # wait until enough samples
        if len(self.buffer) < self.cfg.learning_starts:
            return

        # only train every train_freq steps
        if self.step_count % self.cfg.train_freq != 0:
            return

        # sample a minibatch
        s, a, r, s2, d = self.buffer.sample(self.cfg.batch_size)

        # compute target Q-values
        with torch.no_grad():
            if self.cfg.double_dqn:
                # doubel DQN: actions from online network, values from target network
                best_next = torch.argmax(self.q(s2), dim=1)
                q_next = self.q_target(s2).gather(1, best_next.unsqueeze(1)).squeeze(1)
            else:
                # standard dqn: max over target Q-vals
                q_next = self.q_target(s2).max(1).values

            # bellman target
            target = r + (1.0 - d) * self.cfg.gamma * q_next

        # q(s,a) for chosen actions
        q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(q_sa, target)

        # optimize Q-network
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.max_grad_norm)
        self.opt.step()

        # update target network periodically
        if self.step_count % self.cfg.target_update_freq == 0:
            self.q_target.load_state_dict(self.q.state_dict())

    # build networks and replay buffer from first obersvation
    def _init_from_obs(self, obs):
        x = chw_from_obs(obs)
        c = x.shape[0]  # number of channels

        # online and target Q-networks
        self.q = CNNQ(c, self.action_space.n).to(self.device)
        self.q_target = CNNQ(c, self.action_space.n).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()

        # optimizer and replay buffer
        self.opt = optim.AdamW(self.q.parameters(), lr=self.cfg.lr, amsgrad=True)
        self.buffer = ReplayBuffer(self.cfg.buffer_size, self.device)

    # episolon schedule for ε-greedy
    def _epsilon(self):
        """Exponential epsilon decay."""
        step = self.step_count
        eps_start = self.cfg.eps_start
        eps_final = self.cfg.eps_final
        decay = self.cfg.eps_decay_steps

        return eps_final + (eps_start - eps_final) * np.exp(-step / decay)

    # save only the online Q-network parameters
    def save(self, path: str):
        if self.q is None:
            raise RuntimeError("Cannot save before network is initialized.")
        torch.save(self.q.state_dict(), path)

    # load Q-network weights and rebuild target network
    def load(self, path: str):
        # assume fixed 4-channel input for MsPacman (FrameStack(4))
        obs_channels = 4
        self.q = CNNQ(obs_channels, self.action_space.n).to(self.device)
        self.q_target = CNNQ(obs_channels, self.action_space.n).to(self.device)

        state_dict = torch.load(path, map_location=self.device)
        self.q.load_state_dict(state_dict)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()