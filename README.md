# Snake RL (Educational)

Simple Pygame Snake plus a Deep Q-learning agent written in Python for learning RL basics.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If running headless (CI/server), set `SDL_VIDEODRIVER=dummy` before using Pygame rendering.

## Play manually

```bash
python play.py
```

Use arrow keys. Close the window or collide to end the game.

## Train the agent

```bash
python train.py --episodes 400 --grid 20 20
```

Key flags: `--episodes`, `--max-steps`, `--buffer-size`, `--batch-size`, `--gamma`, `--lr`, `--eps-start/--eps-end/--eps-decay`, `--target-update`, `--warmup`, `--save-path`.

Resume from a saved checkpoint:

```bash
python train.py --episodes 200 --resume models/dqn_snake.pt --eps-start 0.1
```

## Watch the trained agent

```bash
python watch_agent.py --model models/dqn_snake.pt
```

## Project structure

- `src/game.py` — Pygame snake logic and rendering.
- `src/env.py` — Gym-like wrapper with observations/rewards.
- `src/dqn.py` — Q-network, replay buffer, epsilon helper.
- `train.py` — DQN training loop and checkpointing.
- `watch_agent.py` — Render a saved agent.
- `play.py` — Human-playable Snake.
- `tests/test_game.py` — Basic game logic tests.


