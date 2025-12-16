from __future__ import annotations

import math
import random
from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, capacity: int, n_step: int = 1, gamma: float = 0.99) -> None:
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.buffer: Deque[Tuple] = deque(maxlen=capacity)
        # For n-step: store (state, action, reward, next_state, done, n_step_return, n_step_state)
        self.n_step_buffer: Deque[Tuple] = deque(maxlen=n_step)

    def push(self, state, action, reward, next_state, done) -> None:
        """Push transition, computing n-step returns if n_step > 1."""
        if self.n_step == 1:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            # Store in n-step buffer
            self.n_step_buffer.append((state, action, reward, next_state, done))
            
            if len(self.n_step_buffer) == self.n_step or done:
                # Compute n-step return
                n_step_return = 0.0
                n_step_state = self.n_step_buffer[0][0]  # initial state
                n_step_action = self.n_step_buffer[0][1]
                n_step_done = done
                
                for i, (_, _, r, _, d) in enumerate(self.n_step_buffer):
                    n_step_return += (self.gamma ** i) * r
                    if d:
                        n_step_done = True
                        break
                
                final_next_state = self.n_step_buffer[-1][3]
                self.buffer.append((n_step_state, n_step_action, n_step_return, final_next_state, n_step_done))
                
                if done:
                    self.n_step_buffer.clear()

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))

        # Handle both 1D (features) and 3D (image) states
        if state.ndim == 1 or (state.ndim == 2 and state.shape[1] < 100):
            # Feature-based: (batch, features)
            state = torch.tensor(state, device=device, dtype=torch.float32)
            next_state = torch.tensor(next_state, device=device, dtype=torch.float32)
        else:
            # Image-based: (batch, channels, height, width)
            state = torch.tensor(state, device=device, dtype=torch.float32)
            next_state = torch.tensor(next_state, device=device, dtype=torch.float32)
        
        action = torch.tensor(action, device=device, dtype=torch.long).unsqueeze(1)
        reward = torch.tensor(reward, device=device, dtype=torch.float32).unsqueeze(1)
        done = torch.tensor(done, device=device, dtype=torch.float32).unsqueeze(1)
        return state, action, reward, next_state, done

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    """MLP Q-Network for feature-based observations."""
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 256) -> None:
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


class CNNQNetwork(nn.Module):
    """CNN Q-Network for image-based observations (3 channels: body, head, food)."""
    def __init__(self, grid_size: tuple, output_dim: int, channels: int = 3) -> None:
        super().__init__()
        h, w = grid_size
        self.grid_size = grid_size
        
        # Small CNN: 3 conv layers + 2 FC layers
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Calculate flattened size after conv layers
        conv_out_size = 64 * h * w
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        x = self.conv(x)
        return self.fc(x)


def compute_epsilon(frame_idx: int, eps_start: float, eps_end: float, eps_decay: float) -> float:
    return eps_end + (eps_start - eps_end) * math.exp(-1.0 * frame_idx / eps_decay)


def make_optimizer(model: nn.Module, lr: float) -> optim.Optimizer:
    return optim.Adam(model.parameters(), lr=lr)


