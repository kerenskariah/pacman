# agents/dqn_agent.py
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base import BaseAgent


def to_flat_float(obs, device):
    """
    Works with ALE MsPacman observations (H,W,C uint8) or vectors.
    - uint8 images -> scale to [0,1]
    - flatten to 1D tensor on the right device
    """
    if isinstance(obs, torch.Tensor):
        x = obs
        if x.dtype != torch.float32:
            x = x.float()
    else:
        arr = np.array(obs, copy=False)
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = arr.astype(np.float32)
        x = torch.from_numpy(arr)
    return x.view(-1).to(device)


class TinyMLP(nn.Module):
    """Small MLP: flat obs -> 256 -> 256 -> Q-values."""
    def __init__(self, in_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buf)


class DQNAgent(BaseAgent):
    """
    Minimal, readable DQN that matches your BaseAgent and plugs into main.train()/play().

    Simplicity choices:
      - Single network (no target net) for clarity.
      - Flatten observations (works on raw ALE frames or vectors).
      - Linear epsilon decay; MSE TD loss; Adam optimizer.
    """

    def __init__(self, action_space, config):
        super().__init__(action_space, config)
        self.n_actions = action_space.n

        # Device selection
        if getattr(self.config, "DEVICE", None) is None:
            if torch.cuda.is_available():
                dev = "cuda"
            elif torch.backends.mps.is_available():
                dev = "mps"
            else:
                dev = "cpu"
        else:
            dev = self.config.DEVICE
        self.device = torch.device(dev)

        # Hyperparameters
        self.gamma = getattr(self.config, "GAMMA", 0.99)
        self.lr = getattr(self.config, "LR", 1e-4)

        self.replay_capacity = getattr(self.config, "REPLAY_CAPACITY", 20_000)
        self.batch_size = getattr(self.config, "BATCH_SIZE", 64)
        self.train_start_size = getattr(self.config, "TRAIN_START_SIZE", 2_000)

        self.eps = getattr(self.config, "EPSILON_START", 1.0)
        self.eps_end = getattr(self.config, "EPSILON_END", 0.05)
        decay_steps = max(1, getattr(self.config, "EPSILON_DECAY_STEPS", 100_000))
        self.eps_decay = (self.eps - self.eps_end) / decay_steps

        # State
        self.replay = ReplayBuffer(self.replay_capacity)
        self.total_steps = 0

        # Lazy-built model / optim (we don't know obs dim until first call)
        self.q = None
        self.optim = None
        self.criterion = nn.MSELoss()

        # For load-before-first-obs
        self._pending_ckpt = None

    # --- BaseAgent API ---

    def get_action(self, observation):
        if self.q is None:
            self._lazy_build(observation)

        self.total_steps += 1

        # epsilon-greedy
        if random.random() < self.eps:
            action = self.action_space.sample()
        else:
            s = to_flat_float(observation, self.device).unsqueeze(0)  # [1, D]
            with torch.no_grad():
                qvals = self.q(s)
                action = int(torch.argmax(qvals, dim=1).item())

        # linear epsilon decay
        if self.eps > self.eps_end:
            self.eps = max(self.eps_end, self.eps - self.eps_decay)

        return action

    def update(self, state, action, reward, next_state, needs_to_reset):
        if self.q is None:
            self._lazy_build(state)

        done = bool(needs_to_reset)
        self.replay.push(state, int(action), float(reward), next_state, done)

        # wait until buffer is warm
        if len(self.replay) < self.train_start_size:
            return

        self._learn_step()

    def save(self, filepath):
        torch.save({
            "q": None if self.q is None else self.q.state_dict(),
            "optim": None if self.optim is None else self.optim.state_dict(),
            "eps": self.eps,
            "steps": self.total_steps,
            "in_dim": None if self.q is None else self.q.net[0].in_features,
            "n_actions": self.n_actions,
        }, filepath)

    def load(self, filepath):
        self._pending_ckpt = torch.load(
        filepath,
        map_location=self.device,
        weights_only=False  # <-- key change
    )

    # --- internals ---

    def _lazy_build(self, sample_obs):
        x = to_flat_float(sample_obs, self.device)
        in_dim = x.numel()
        self.q = TinyMLP(in_dim, self.n_actions).to(self.device)
        self.optim = optim.Adam(self.q.parameters(), lr=self.lr)

        if self._pending_ckpt is not None:
            ck = self._pending_ckpt
            if ck.get("q") is not None:
                self.q.load_state_dict(ck["q"])
            if ck.get("optim") is not None:
                self.optim.load_state_dict(ck["optim"])
            self.eps = ck.get("eps", self.eps)
            self.total_steps = ck.get("steps", self.total_steps)
            self._pending_ckpt = None

    def _learn_step(self):
        # sample minibatch
        s, a, r, s2, d = self.replay.sample(self.batch_size)

        S  = torch.stack([to_flat_float(x, self.device) for x in s])
        S2 = torch.stack([to_flat_float(x, self.device) for x in s2])
        A  = torch.tensor(a, dtype=torch.long, device=self.device).view(-1, 1)
        R  = torch.tensor(r, dtype=torch.float32, device=self.device)
        D  = torch.tensor(d, dtype=torch.float32, device=self.device)

        # Q(s,a)
        q_sa = self.q(S).gather(1, A).squeeze(1)

        # target with same network (simple baseline)
        with torch.no_grad():
            q_next_max = self.q(S2).max(dim=1).values
            target = R + (1.0 - D) * self.gamma * q_next_max

        loss = self.criterion(q_sa, target)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
