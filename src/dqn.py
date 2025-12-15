from __future__ import annotations

import math
import random
from collections import deque
from typing import Deque, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: Deque[Tuple] = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(
            lambda x: torch.tensor(x, device=device),
            zip(*batch),
        )
        action = action.long().unsqueeze(1)
        reward = reward.float().unsqueeze(1)
        done = done.float().unsqueeze(1)
        return state.float(), action, reward, next_state.float(), done

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def compute_epsilon(frame_idx: int, eps_start: float, eps_end: float, eps_decay: float) -> float:
    return eps_end + (eps_start - eps_end) * math.exp(-1.0 * frame_idx / eps_decay)


def make_optimizer(model: nn.Module, lr: float) -> optim.Optimizer:
    return optim.Adam(model.parameters(), lr=lr)


