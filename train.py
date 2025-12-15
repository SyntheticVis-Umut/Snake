from __future__ import annotations

import argparse
import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from src.dqn import QNetwork, ReplayBuffer, compute_epsilon, make_optimizer
from src.env import SnakeEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN agent to play Snake")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=float, default=3000)
    parser.add_argument("--target-update", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--grid", type=int, nargs=2, default=(20, 20))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="models/dqn_snake.pt")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to continue training")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = SnakeEnv(grid_size=tuple(args.grid), render_mode=None, seed=args.seed)

    state_dim = env.reset().shape[0]
    action_dim = len(env.ACTIONS)

    policy_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = make_optimizer(policy_net, lr=args.lr)
    memory = ReplayBuffer(args.buffer_size)
    criterion = nn.MSELoss()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    frame_idx = 0
    best_score = float("-inf")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        policy_net.load_state_dict(checkpoint["policy_state_dict"])
        target_net.load_state_dict(policy_net.state_dict())
        best_score = checkpoint.get("best_score", best_score)
        print(f"Resumed policy from {args.resume} (best_score={best_score})")

    for episode in range(1, args.episodes + 1):
        state = env.reset()
        episode_reward = 0.0

        for step in range(args.max_steps):
            epsilon = compute_epsilon(frame_idx, args.eps_start, args.eps_end, args.eps_decay)
            if random.random() < epsilon:
                action = random.choice(env.ACTIONS)
            else:
                with torch.no_grad():
                    state_v = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
                    q_values = policy_net(state_v)
                    action = int(torch.argmax(q_values).item())

            next_state, reward, done, info = env.step(action)
            memory.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            frame_idx += 1

            if len(memory) >= args.batch_size and frame_idx > args.warmup:
                loss = optimize_model(
                    policy_net,
                    target_net,
                    memory,
                    optimizer,
                    criterion,
                    args.batch_size,
                    args.gamma,
                    device,
                )

            if frame_idx % args.target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        if episode_reward > best_score:
            best_score = episode_reward
            torch.save(
                {
                    "policy_state_dict": policy_net.state_dict(),
                    "args": vars(args),
                    "best_score": best_score,
                },
                args.save_path,
            )

        if episode % 10 == 0 or episode == 1:
            print(
                f"Episode {episode}/{args.episodes} "
                f"Reward: {episode_reward:.2f} "
                f"Epsilon: {epsilon:.3f} "
                f"Best: {best_score:.2f}"
            )

    env.close()
    print(f"Training complete. Best episodic reward: {best_score:.2f}")
    print(f"Model saved to: {args.save_path}")


def optimize_model(
    policy_net: QNetwork,
    target_net: QNetwork,
    memory: ReplayBuffer,
    optimizer,
    criterion,
    batch_size: int,
    gamma: float,
    device: torch.device,
):
    states, actions, rewards, next_states, dones = memory.sample(batch_size, device)

    q_values = policy_net(states).gather(1, actions)
    with torch.no_grad():
        next_q_values = target_net(next_states).max(1, keepdim=True).values
        targets = rewards + gamma * (1 - dones) * next_q_values

    loss = criterion(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


if __name__ == "__main__":
    args = parse_args()
    train(args)


