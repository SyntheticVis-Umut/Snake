from __future__ import annotations

import argparse
import os
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.dqn import QNetwork, CNNQNetwork, ReplayBuffer, compute_epsilon, make_optimizer
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
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Device to use: "auto", "cuda", "cpu", or explicit device id (e.g., "cuda:0")',
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Global norm gradient clipping value; set <=0 to disable",
    )
    parser.add_argument(
        "--double-dqn",
        action="store_true",
        default=True,
        help="Use Double DQN target selection (on by default)",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Run greedy evaluation every N episodes (0 disables)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of episodes per evaluation run",
    )
    parser.add_argument(
        "--observation-type",
        type=str,
        default="features",
        choices=["features", "image"],
        help="Observation type: 'features' (11 booleans) or 'image' (3-channel CNN input)",
    )
    parser.add_argument(
        "--n-step",
        type=int,
        default=1,
        help="n-step returns (1 = standard, >1 for multi-step learning)",
    )
    parser.add_argument(
        "--dueling",
        action="store_true",
        default=True,
        help="Use dueling architecture (value + advantage heads)",
    )
    parser.add_argument(
        "--per-alpha",
        type=float,
        default=0.6,
        help="Prioritized replay alpha (0 = uniform, higher = more priority focus)",
    )
    parser.add_argument(
        "--per-beta",
        type=float,
        default=0.4,
        help="Prioritized replay beta for importance sampling weights",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save checkpoint every N episodes (0 disables)",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args: argparse.Namespace, drive_save_path: Optional[str] = None) -> None:
    # Support drive_save_path as both parameter and args attribute
    if drive_save_path is None:
        drive_save_path = getattr(args, 'drive_save_path', None)
    # Debug: Always print what we found
    print(f"DEBUG: drive_save_path parameter = {drive_save_path}")
    print(f"DEBUG: args.drive_save_path attribute = {getattr(args, 'drive_save_path', 'NOT SET')}")
    if drive_save_path:
        print(f"✓ Drive save path configured: {drive_save_path}")
    else:
        print("⚠ WARNING: No drive save path configured - checkpoints will NOT be saved to Drive!")
    set_seed(args.seed)
    device = _resolve_device(args.device)
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        print(f"Using CUDA device: {gpu_name}")
    else:
        print("CUDA not available; falling back to CPU.")
    env = SnakeEnv(
        grid_size=tuple(args.grid),
        render_mode=None,
        seed=args.seed,
        observation_type=args.observation_type,
    )
    eval_env = (
        SnakeEnv(
            grid_size=tuple(args.grid),
            render_mode=None,
            seed=args.seed + 1234,
            observation_type=args.observation_type,
        )
        if args.eval_every > 0
        else None
    )

    sample_state = env.reset()
    action_dim = len(env.ACTIONS)

    # Create appropriate network based on observation type
    if args.observation_type == "image":
        # CNN for image observations
        policy_net = CNNQNetwork(grid_size=tuple(args.grid), output_dim=action_dim, dueling=args.dueling).to(device)
        target_net = CNNQNetwork(grid_size=tuple(args.grid), output_dim=action_dim, dueling=args.dueling).to(device)
    else:
        # MLP for feature observations
        state_dim = sample_state.shape[0]
        policy_net = QNetwork(state_dim, action_dim, dueling=args.dueling).to(device)
        target_net = QNetwork(state_dim, action_dim, dueling=args.dueling).to(device)
    
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = make_optimizer(policy_net, lr=args.lr)
    memory = ReplayBuffer(args.buffer_size, n_step=args.n_step, gamma=args.gamma, per_alpha=args.per_alpha)
    criterion = nn.SmoothL1Loss()

    # Ensure local save directory exists
    local_save_dir = os.path.dirname(args.save_path)
    if local_save_dir:
        try:
            os.makedirs(local_save_dir, exist_ok=True)
            print(f"✓ Created/verified save directory: {local_save_dir}")
        except OSError as e:
            print(f"⚠ Warning: Could not create save directory {local_save_dir}: {e}")
            print(f"  Current working directory: {os.getcwd()}")
            # Try to save to current directory instead
            if local_save_dir != "":
                args.save_path = os.path.basename(args.save_path)
                print(f"  Falling back to saving in current directory: {args.save_path}")

    frame_idx = 0
    best_score = float("-inf")
    best_eval = float("-inf")
    start_episode = 0

    if args.resume:
        # Check if resume path exists
        if not os.path.exists(args.resume):
            print(f"⚠ Warning: Resume checkpoint not found at {args.resume}, starting fresh training.")
            args.resume = None
        else:
            try:
                checkpoint = torch.load(args.resume, map_location=device)
                policy_net.load_state_dict(checkpoint["policy_state_dict"])
                target_state = checkpoint.get("target_state_dict")
                if target_state:
                    target_net.load_state_dict(target_state)
                else:
                    target_net.load_state_dict(policy_net.state_dict())
                opt_state = checkpoint.get("optimizer_state_dict")
                if opt_state:
                    optimizer.load_state_dict(opt_state)
                best_score = checkpoint.get("best_score", best_score)
                best_eval = checkpoint.get("best_eval", best_eval)
                frame_idx = checkpoint.get("frame_idx", frame_idx)
                start_episode = checkpoint.get("episode", start_episode)
                print(f"✓ Resumed training from {args.resume} (episode={start_episode}, best_score={best_score:.2f})")
            except (RuntimeError, KeyError, OSError) as e:
                print(f"⚠ Warning: Failed to load checkpoint from {args.resume}: {e}")
                print(f"  Starting fresh training instead. (The checkpoint file may be corrupted or incomplete.)")
                args.resume = None

    for episode in range(start_episode + 1, args.episodes + 1):
        state = env.reset()
        episode_reward = 0.0

        for step in range(args.max_steps):
            epsilon = compute_epsilon(frame_idx, args.eps_start, args.eps_end, args.eps_decay)
            if random.random() < epsilon:
                action = random.choice(env.ACTIONS)
            else:
                with torch.no_grad():
                    if args.observation_type == "image":
                        # Image: (C, H, W) -> (1, C, H, W)
                        state_v = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
                    else:
                        # Features: (features,) -> (1, features)
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
                    args.grad_clip,
                    args.double_dqn,
                    args.per_beta,
                )

            if frame_idx % args.target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        if episode_reward > best_score:
            best_score = episode_reward
            _save_checkpoint(
                args.save_path,
                policy_net,
                target_net,
                optimizer,
                args,
                best_score,
                best_eval,
                episode,
                frame_idx,
                drive_path=drive_save_path,
            )

        if eval_env and args.eval_every > 0 and episode % args.eval_every == 0:
            eval_stats = evaluate_policy(
                policy_net, eval_env, args.eval_episodes, device, args.max_steps, args.observation_type
            )
            mean_r, median_r, max_r, std_r = eval_stats
            print(
                f"[Eval @ episode {episode}] "
                f"mean: {mean_r:.2f} median: {median_r:.2f} "
                f"max: {max_r:.2f} std: {std_r:.2f}"
            )
            if mean_r > best_eval:
                best_eval = mean_r
                _save_checkpoint(
                    args.save_path,
                    policy_net,
                    target_net,
                    optimizer,
                    args,
                    best_score,
                    best_eval,
                    episode,
                    frame_idx,
                    drive_path=drive_save_path,
                )

        if episode % 10 == 0 or episode == 1:
            print(
                f"Episode {episode}/{args.episodes} "
                f"Reward: {episode_reward:.2f} "
                f"Epsilon: {epsilon:.3f} "
                f"Best: {best_score:.2f}"
            )
        if args.save_every and episode % args.save_every == 0:
            _save_checkpoint(
                args.save_path,
                policy_net,
                target_net,
                optimizer,
                args,
                best_score,
                best_eval,
                episode,
                frame_idx,
                drive_path=drive_save_path,
            )

    env.close()
    if eval_env:
        eval_env.close()
    print(f"Training complete. Best episodic reward: {best_score:.2f}")
    if best_eval > float("-inf"):
        print(f"Best eval mean reward: {best_eval:.2f}")
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
    grad_clip: float,
    double_dqn: bool,
    per_beta: float,
):
    states, actions, rewards, next_states, dones, indices, weights = memory.sample(batch_size, device, per_beta=per_beta)

    q_values = policy_net(states).gather(1, actions)
    with torch.no_grad():
        if double_dqn:
            next_actions = policy_net(next_states).argmax(1, keepdim=True)
            next_q_values = target_net(next_states).gather(1, next_actions)
        else:
            next_q_values = target_net(next_states).max(1, keepdim=True).values
        targets = rewards + gamma * (1 - dones) * next_q_values

    loss = criterion(q_values, targets)
    loss = loss * weights  # PER importance sampling
    loss = loss.mean()
    optimizer.zero_grad()
    loss.backward()
    if grad_clip and grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), grad_clip)
    optimizer.step()
    # Update priorities
    td_errors = (targets - q_values).detach().abs().squeeze(1) + 1e-6
    memory.update_priorities(indices, td_errors)
    return loss.item()


def evaluate_policy(
    policy_net,
    env: SnakeEnv,
    episodes: int,
    device: torch.device,
    max_steps: int,
    observation_type: str = "features",
):
    rewards = []
    policy_net.eval()
    with torch.no_grad():
        for idx in range(episodes):
            state = env.reset(seed=idx)
            ep_reward = 0.0
            for _ in range(max_steps):
                if observation_type == "image":
                    # Image: (C, H, W) -> (1, C, H, W)
                    state_v = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
                else:
                    # Features: (features,) -> (1, features)
                    state_v = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
                q_values = policy_net(state_v)
                action = int(torch.argmax(q_values).item())
                state, reward, done, _ = env.step(action)
                ep_reward += reward
                if done:
                    break
            rewards.append(ep_reward)
    policy_net.train()
    rewards_arr = np.array(rewards, dtype=np.float32)
    return (
        float(rewards_arr.mean()),
        float(np.median(rewards_arr)),
        float(rewards_arr.max()),
        float(rewards_arr.std()),
    )


def _save_checkpoint(
    path: str,
    policy_net: nn.Module,
    target_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    best_score: float,
    best_eval: float,
    episode: int,
    frame_idx: int,
    drive_path: Optional[str] = None,
) -> None:
    checkpoint_data = {
        "policy_state_dict": policy_net.state_dict(),
        "target_state_dict": target_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "best_score": best_score,
        "best_eval": best_eval,
        "episode": episode,
        "frame_idx": frame_idx,
    }
    
    # Save to local path
    local_dir = os.path.dirname(path)
    if local_dir:  # Only create directory if path has a directory component
        try:
            os.makedirs(local_dir, exist_ok=True)
        except OSError as e:
            print(f"⚠ Warning: Could not create directory {local_dir}: {e}")
            # Try to save anyway (might work if parent dir exists)
    torch.save(checkpoint_data, path)
    
    # Also save to Google Drive if path provided
    print(f"DEBUG _save_checkpoint: drive_path = {drive_path}")
    if drive_path:
        try:
            import shutil
            # Verify Drive is accessible
            if not os.path.exists("/content/drive"):
                print(f"⚠ Warning: Google Drive not mounted at /content/drive. Skipping Drive save.")
            else:
                # Verify local file exists before copying
                if not os.path.exists(path):
                    print(f"⚠ Warning: Local checkpoint {path} not found. Cannot copy to Drive.")
                else:
                    # Copy the saved checkpoint to Drive
                    # MyDrive directory should already exist, but ensure it does
                    drive_dir = os.path.dirname(drive_path)
                    if drive_dir and not os.path.exists(drive_dir):
                        os.makedirs(drive_dir, exist_ok=True)
                    shutil.copy2(path, drive_path)
                    print(f"✓ Checkpoint also saved to Drive: {drive_path}")
        except PermissionError as e:
            print(f"⚠ Warning: Permission denied saving to Drive: {e}")
        except Exception as e:
            print(f"⚠ Warning: Failed to save to Drive ({drive_path}): {e}")
            import traceback
            traceback.print_exc()

def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available on this machine.")
    return device


if __name__ == "__main__":
    args = parse_args()
    train(args)

