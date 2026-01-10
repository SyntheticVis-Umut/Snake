from __future__ import annotations

import argparse
import time

import torch

from src.dqn import QNetwork, CNNQNetwork
from src.env import SnakeEnv
from src.mcts import MCTSPlanner, SimpleLookaheadPlanner


def parse_args():
    parser = argparse.ArgumentParser(description="Watch a trained DQN agent play Snake")
    parser.add_argument("--model", type=str, default="models/dqn_snake.pt")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--grid", type=int, nargs=2, default=(20, 20))
    parser.add_argument(
        "--planner",
        type=str,
        choices=["none", "mcts", "lookahead"],
        default="none",
        help="Planning method: 'none' (greedy Q), 'mcts' (Monte Carlo Tree Search), 'lookahead' (simple depth-limited search)",
    )
    parser.add_argument(
        "--mcts-simulations",
        type=int,
        default=100,
        help="Number of MCTS simulations (only for --planner mcts)",
    )
    parser.add_argument(
        "--mcts-depth",
        type=int,
        default=50,
        help="Maximum depth for MCTS rollout (only for --planner mcts)",
    )
    parser.add_argument(
        "--lookahead-depth",
        type=int,
        default=10,
        help="Lookahead depth for simple planner (only for --planner lookahead)",
    )
    parser.add_argument(
        "--lookahead-samples",
        type=int,
        default=3,
        help="Number of action sequences to sample per action (only for --planner lookahead)",
    )
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
    if observation_type == "image" or any(k.startswith("conv.") for k in policy_state.keys()):
        # CNN model
        model = CNNQNetwork(grid_size=tuple(grid_size), output_dim=action_dim, dueling=dueling).to(device)
        print("Using CNN model (image observations)")
    else:
        # MLP model - infer hidden size from weights
        sample_state = env.reset()
        state_dim = sample_state.shape[0]
        # Try different possible weight key names
        if "feature.0.weight" in policy_state:
            hidden_size = policy_state["feature.0.weight"].shape[0]
        else:
            # Fallback: try to infer from first linear layer
            for key in policy_state.keys():
                if "weight" in key and len(policy_state[key].shape) == 2:
                    hidden_size = policy_state[key].shape[0]
                    break
            else:
                hidden_size = 256  # Default
        model = QNetwork(state_dim, action_dim, hidden=hidden_size, dueling=dueling).to(device)
        print(f"Using MLP model (feature observations, hidden={hidden_size}, dueling={dueling})")
    
    model.load_state_dict(policy_state)
    model.eval()

    # Initialize planner if requested
    planner = None
    if args.planner == "mcts":
        planner = MCTSPlanner(
            q_network=model,
            env=env,
            device=device,
            num_simulations=args.mcts_simulations,
            max_depth=args.mcts_depth,
            observation_type=observation_type,
        )
        print(f"Using MCTS planner: {args.mcts_simulations} simulations, max depth {args.mcts_depth}")
    elif args.planner == "lookahead":
        planner = SimpleLookaheadPlanner(
            q_network=model,
            env=env,
            device=device,
            lookahead_depth=args.lookahead_depth,
            num_samples=args.lookahead_samples,
            observation_type=observation_type,
        )
        print(f"Using lookahead planner: depth {args.lookahead_depth}, {args.lookahead_samples} samples per action")
    else:
        print("Using greedy Q-network (no planning)")

    for ep in range(1, args.episodes + 1):
        state = env.reset()
        done = False
        ep_reward = 0.0
        step_count = 0
        while not done:
            if planner is not None:
                # Use planner for lookahead
                action = planner.plan(state)
            else:
                # Greedy Q-network
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
            step_count += 1
            env.render()
            time.sleep(0.05)
        print(f"Episode {ep}: reward={ep_reward:.2f}, steps={step_count}")

    env.close()


if __name__ == "__main__":
    main()


