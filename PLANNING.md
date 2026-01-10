# Planning and Lookahead for Snake DQN

## Overview

The base DQN model makes decisions based only on the current state (greedy Q-value selection). To enable **multi-step lookahead** (10, 50, 100+ steps forward), we've added two planning methods:

1. **Monte Carlo Tree Search (MCTS)** - Advanced tree search with exploration
2. **Simple Lookahead** - Depth-limited search with Q-network guidance

## Why Planning?

- **Current limitation**: The model only sees the current state and picks the action with highest Q-value
- **With planning**: The model can simulate 10-100+ steps ahead to evaluate long-term consequences
- **Result**: Better decision-making, especially for avoiding traps and planning paths

## Usage

### MCTS Planner (Recommended for Best Performance)

```bash
python watch_agent.py \
    --model models/dqn_snake.pt \
    --planner mcts \
    --mcts-simulations 200 \
    --mcts-depth 100 \
    --episodes 5
```

**Parameters:**
- `--mcts-simulations`: Number of MCTS simulations per move (more = better but slower)
  - 50-100: Fast, good for real-time
  - 200-500: Better quality, slower
  - 1000+: Best quality, very slow
- `--mcts-depth`: Maximum depth of lookahead (how many steps forward)
  - 10-20: Short-term planning
  - 50-100: Medium-term planning
  - 100-200: Long-term planning

### Simple Lookahead Planner (Faster Alternative)

```bash
python watch_agent.py \
    --model models/dqn_snake.pt \
    --planner lookahead \
    --lookahead-depth 20 \
    --lookahead-samples 5 \
    --episodes 5
```

**Parameters:**
- `--lookahead-depth`: How many steps to look ahead (10-50 recommended)
- `--lookahead-samples`: Number of action sequences to try per initial action (3-10 recommended)

### Greedy (No Planning)

```bash
python watch_agent.py \
    --model models/dqn_snake.pt \
    --planner none \
    --episodes 5
```

Uses the trained Q-network directly without any lookahead (fastest, but less smart).

## How It Works

### MCTS Planner

1. **Selection**: Traverses the search tree using UCB1 (balancing exploration vs exploitation)
2. **Expansion**: Adds new nodes to the tree
3. **Simulation**: Rollouts using the Q-network to estimate future value
4. **Backpropagation**: Updates value estimates up the tree
5. **Action Selection**: Chooses the most visited/highest value action

The planner can look **100+ steps ahead** by building a search tree and using the Q-network as a value function.

### Simple Lookahead Planner

1. For each possible action:
   - Generate multiple action sequences (using Q-network to guide subsequent actions)
   - Simulate each sequence forward (10-50 steps)
   - Evaluate the final state using Q-network
2. Select the action with the best average outcome

## Performance Comparison

| Method | Speed | Quality | Lookahead |
|--------|-------|---------|-----------|
| Greedy Q | Fastest | Good | 0 steps |
| Simple Lookahead | Fast | Better | 10-50 steps |
| MCTS | Slower | Best | 50-200+ steps |

## Tips

1. **Start with MCTS**: Use `--mcts-simulations 100 --mcts-depth 50` for a good balance
2. **Increase for harder games**: More simulations and depth for better performance
3. **Use Simple Lookahead for speed**: If MCTS is too slow, try lookahead with depth 20-30
4. **Training still matters**: Planning helps, but a well-trained Q-network is essential

## Technical Details

- Both planners use the trained Q-network as a value function
- They create environment copies for simulation (doesn't affect the real game)
- MCTS uses UCB1 exploration constant (default: 1.41 = √2)
- Discount factor: 0.99 (same as training)

## Example: 100-Step Lookahead

```bash
# Very deep lookahead for maximum planning
python watch_agent.py \
    --model models/dqn_snake.pt \
    --planner mcts \
    --mcts-simulations 500 \
    --mcts-depth 200 \
    --episodes 1
```

This will simulate up to 200 steps ahead with 500 simulations per move, giving the agent excellent long-term planning capabilities.
