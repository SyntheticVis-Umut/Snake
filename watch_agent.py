from __future__ import annotations

import argparse
import time

import torch

from src.dqn import QNetwork, CNNQNetwork
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

    checkpoint = torch.load(args.model, map_location=device)
    checkpoint_args = checkpoint.get("args", {})
    observation_type = checkpoint_args.get("observation_type", "features")
    grid_size = checkpoint_args.get("grid", args.grid)
    dueling = checkpoint_args.get("dueling", False)
    
    env = SnakeEnv(
        grid_size=tuple(grid_size),
        render_mode="human",
        observation_type=observation_type,
    )
    action_dim = len(env.ACTIONS)

    policy_state = checkpoint["policy_state_dict"]
    
    # Determine model type from checkpoint
    if observation_type == "image" or "conv.0.weight" in policy_state:
        # CNN model
        model = CNNQNetwork(grid_size=tuple(grid_size), output_dim=action_dim, dueling=dueling).to(device)
    else:
        # MLP model - infer hidden size from weights
        sample_state = env.reset()
        state_dim = sample_state.shape[0]
        hidden_size = policy_state["net.0.weight"].shape[0]
        model = QNetwork(state_dim, action_dim, hidden=hidden_size, dueling=dueling).to(device)
    
    model.load_state_dict(policy_state)
    model.eval()

    for ep in range(1, args.episodes + 1):
        state = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            with torch.no_grad():
                if observation_type == "image":
                    # Image: (C, H, W) -> (1, C, H, W)
                    state_v = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
                else:
                    # Features: (features,) -> (1, features)
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


