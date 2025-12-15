# Training Guide (Snake DQN)

## What is DQN, QNetwork, MLP?
- DQN: Deep Q-Network. Learns a function Q(s, a) that estimates long-term value of taking action a in state s, then acting greedily thereafter.
- QNetwork: The neural net that outputs Q-values for each action given a state. In this project it is a small multilayer perceptron (MLP).
- MLP: A feedforward network of linear layers plus nonlinearities (here: Linear → ReLU → Linear → ReLU → Linear). It maps the state vector to 3 action values (straight, left, right).

## Training loop (conceptual)
1) Reset env → get initial state.
2) Choose action with epsilon-greedy:
   - With probability epsilon: random action (explore).
   - Otherwise: argmax QNetwork(state) (exploit).
3) Step env → get next_state, reward, done. Store (state, action, reward, next_state, done) in replay buffer.
4) After warmup and when buffer is big enough, sample a batch and update:
   - Current Q: policy_net(states).gather(actions)
   - Target Q: reward + gamma * (1 - done) * max_a target_net(next_states)
   - Loss: MSE(current, target) → backprop with Adam.
5) Every target_update steps copy policy_net weights into target_net (stabilizes targets).
6) Repeat for max_steps per episode; track best episodic reward; save best model.

## Key parameters (train.py flags)
- episodes: How many games to train. More = more learning time.
- max-steps: Max moves per episode before forcing reset.
- buffer-size: Replay buffer capacity. Larger holds more diverse experiences.
- batch-size: Samples per training step. Larger = smoother but heavier; smaller = noisier but faster.
- gamma: Discount factor for future rewards (0–1). Higher values weight future more.
- lr: Learning rate for Adam. Higher learns faster but can destabilize; lower is steadier.
- eps-start / eps-end / eps-decay: Epsilon-greedy schedule. Start high to explore, decay toward eps-end.
- target-update: Steps between syncing policy → target. Larger = smoother targets, slower adaptation.
- warmup: Steps collected before any training. Lets buffer fill with diverse data.
- grid: Board size (width height).
- seed: Random seed for reproducibility.
- save-path: Where to write best checkpoint.
- resume: Load an existing checkpoint and continue training.

## Practical tips to improve performance
- More training: Increase episodes and/or max-steps.
- Exploration schedule: Lower eps-end (e.g., 0.02–0.05) and slow decay (larger eps-decay) for better early coverage.
- Learning rate: Try 5e-4 or 1e-4 if updates feel jittery.
- Target updates: Increase target-update (e.g., 1000–2000) for smoother targets if learning oscillates.
- Batch and buffer: If you have GPU/CPU headroom, raise batch-size (e.g., 256) and buffer-size (e.g., 100k).
- Warmup: Increase warmup (e.g., 3000–5000) so early random moves do not dominate learning.
- Reward shaping: Current rewards (+10 food, -10 death, -0.1 step) work; you can try slightly higher food or smaller step penalty to encourage longer survival.
- Resume training: `python train.py --episodes 200 --resume models/dqn_snake.pt --eps-start 0.1`

## Files involved
- train.py: Training script, CLI flags, loop.
- src/dqn.py: QNetwork, replay buffer, epsilon schedule.
- src/env.py: Gym-like wrapper; returns state, reward, done.
- models/dqn_snake.pt: Saved policy checkpoint (best so far).

