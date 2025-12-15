from __future__ import annotations

import argparse
import time

import torch

from src.dqn import QNetwork
from src.env import SnakeEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Watch a trained DQN agent play Snake")
    parser.add_argument("--model", type=str, default="models/dqn_snake.pt")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--grid", type=int, nargs=2, default=(20, 20))
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = SnakeEnv(grid_size=tuple(args.grid), render_mode="human")
    state_dim = env.reset().shape[0]
    action_dim = len(env.ACTIONS)

    checkpoint = torch.load(args.model, map_location=device)
    model = QNetwork(state_dim, action_dim).to(device)
    model.load_state_dict(checkpoint["policy_state_dict"])
    model.eval()

    for ep in range(1, args.episodes + 1):
        state = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            with torch.no_grad():
                state_v = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
                q_values = model(state_v)
                action = int(torch.argmax(q_values).item())
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            env.render()
            time.sleep(0.05)
        print(f"Episode {ep}: reward={ep_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()


