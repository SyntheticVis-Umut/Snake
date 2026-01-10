from __future__ import annotations

import copy
from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
import torch

from src.env import SnakeEnv


class MCTSNode:
    """Node in the MCTS search tree."""
    def __init__(self, state: np.ndarray, parent: Optional[MCTSNode] = None, action: Optional[int] = None):
        self.state = state
        self.parent = parent
        self.action = action  # Action that led to this node
        self.children: dict[int, MCTSNode] = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.is_terminal = False
        
    @property
    def value(self) -> float:
        """Average value estimate."""
        return self.total_value / self.visit_count if self.visit_count > 0 else 0.0
    
    def is_fully_expanded(self, num_actions: int = 3) -> bool:
        """Check if all actions have been tried."""
        return len(self.children) == num_actions


class MCTSPlanner:
    """
    Monte Carlo Tree Search planner for Snake.
    Uses the trained Q-network as a value function and performs lookahead search.
    """
    def __init__(
        self,
        q_network,
        env: SnakeEnv,
        device: torch.device,
        num_simulations: int = 100,
        max_depth: int = 50,
        exploration_constant: float = 1.41,  # sqrt(2)
        observation_type: str = "features",
    ):
        self.q_network = q_network
        self.env = env
        self.device = device
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.exploration_constant = exploration_constant
        self.observation_type = observation_type
        
    def _state_to_tensor(self, state: np.ndarray) -> torch.Tensor:
        """Convert state to tensor for network input."""
        if self.observation_type == "image":
            # Image: (C, H, W) -> (1, C, H, W)
            return torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        else:
            # Features: (features,) -> (1, features)
            return torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
    
    def _get_q_value(self, state: np.ndarray, action: int) -> float:
        """Get Q-value for a state-action pair."""
        with torch.no_grad():
            state_t = self._state_to_tensor(state)
            q_values = self.q_network(state_t)
            return float(q_values[0, action].item())
    
    def _get_value_estimate(self, state: np.ndarray) -> float:
        """Get value estimate for a state (max Q-value)."""
        with torch.no_grad():
            state_t = self._state_to_tensor(state)
            q_values = self.q_network(state_t)
            return float(q_values.max().item())
    
    def _simulate_step(self, env_copy: SnakeEnv, action: int) -> Tuple[np.ndarray, float, bool]:
        """Simulate one step in the environment."""
        next_state, reward, done, _ = env_copy.step(action)
        return next_state, reward, done
    
    def _select_action(self, node: MCTSNode, num_actions: int = 3) -> int:
        """Select action using UCB1 formula."""
        if not node.is_fully_expanded(num_actions):
            # Expand: try an unexplored action
            explored_actions = set(node.children.keys())
            for action in range(num_actions):
                if action not in explored_actions:
                    return action
        
        # UCB1: argmax(Q + c * sqrt(ln(N) / n))
        best_action = None
        best_score = float('-inf')
        
        for action in range(num_actions):
            child = node.children[action]
            exploitation = child.value
            exploration = self.exploration_constant * np.sqrt(
                np.log(node.visit_count + 1) / (child.visit_count + 1)
            )
            score = exploitation + exploration
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action
    
    def _expand(self, node: MCTSNode, env_copy: SnakeEnv) -> MCTSNode:
        """Expand a node by trying a new action."""
        num_actions = len(env_copy.ACTIONS)
        
        # Find an unexplored action
        for action in range(num_actions):
            if action not in node.children:
                # Create child node
                next_state, reward, done = self._simulate_step(env_copy, action)
                child = MCTSNode(next_state, parent=node, action=action)
                child.is_terminal = done
                node.children[action] = child
                return child
        
        return node  # Should not happen if called correctly
    
    def _rollout(self, env_copy: SnakeEnv, depth: int = 0) -> float:
        """Rollout simulation using Q-network for value estimation."""
        if depth >= self.max_depth:
            # Use Q-network value estimate
            state = env_copy.state
            return self._get_value_estimate(state)
        
        if env_copy.game.done:
            return -10.0  # Terminal penalty
        
        # Use Q-network to guide rollout (greedy action selection)
        state = env_copy.state
        with torch.no_grad():
            state_t = self._state_to_tensor(state)
            q_values = self.q_network(state_t)
            action = int(torch.argmax(q_values).item())
        
        next_state, reward, done = self._simulate_step(env_copy, action)
        
        if done:
            return reward
        
        # Continue rollout
        return reward + 0.99 * self._rollout(env_copy, depth + 1)
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value up the tree."""
        current = node
        while current is not None:
            current.visit_count += 1
            current.total_value += value
            value = value * 0.99  # Discount factor
            current = current.parent
    
    def _create_env_copy(self) -> SnakeEnv:
        """Create a copy of the environment for simulation."""
        # Create new env with same parameters
        env_copy = SnakeEnv(
            grid_size=(self.env.game.grid_width, self.env.game.grid_height),
            render_mode=None,
            seed=None,  # Don't need seed for simulation
            observation_type=self.observation_type,
        )
        # Copy game state - need to reset first to initialize properly
        env_copy.game.reset()
        # Now copy the actual state
        from collections import deque
        env_copy.game.snake = deque(copy.deepcopy(list(self.env.game.snake)))
        env_copy.game.direction = self.env.game.direction
        env_copy.game.food = self.env.game.food
        env_copy.game.score = self.env.game.score
        env_copy.game.done = self.env.game.done
        # Update the encoded state
        env_copy.state = env_copy._encode_state()
        return env_copy
    
    def plan(self, state: np.ndarray) -> int:
        """
        Perform MCTS planning and return the best action.
        
        Args:
            state: Current state of the environment
            
        Returns:
            Best action according to MCTS search
        """
        root = MCTSNode(state)
        
        for _ in range(self.num_simulations):
            # Selection: traverse from root to leaf
            node = root
            env_copy = self._create_env_copy()
            path = []
            
            while not node.is_terminal and node.is_fully_expanded() and len(path) < self.max_depth:
                action = self._select_action(node)
                node = node.children[action]
                path.append(action)
                env_copy.step(action)
                if env_copy.game.done:
                    break
            
            # Expansion: add a new child if not terminal
            if not node.is_terminal and not node.is_fully_expanded():
                node = self._expand(node, env_copy)
            
            # Simulation: rollout from this node
            if node.is_terminal:
                value = -10.0  # Terminal penalty
            else:
                value = self._rollout(env_copy, depth=len(path))
            
            # Backpropagation: update values up the tree
            self._backpropagate(node, value)
        
        # Select best action from root (most visited or highest value)
        best_action = None
        best_visits = -1
        
        for action in range(len(self.env.ACTIONS)):
            if action in root.children:
                child = root.children[action]
                if child.visit_count > best_visits:
                    best_visits = child.visit_count
                    best_action = action
        
        # Fallback to Q-network if MCTS didn't explore
        if best_action is None:
            with torch.no_grad():
                state_t = self._state_to_tensor(state)
                q_values = self.q_network(state_t)
                best_action = int(torch.argmax(q_values).item())
        
        return best_action


class SimpleLookaheadPlanner:
    """
    Simpler depth-limited search planner.
    Evaluates all action sequences up to a certain depth.
    """
    def __init__(
        self,
        q_network,
        env: SnakeEnv,
        device: torch.device,
        lookahead_depth: int = 10,
        num_samples: int = 3,  # Number of action sequences to try per action
        observation_type: str = "features",
    ):
        self.q_network = q_network
        self.env = env
        self.device = device
        self.lookahead_depth = lookahead_depth
        self.num_samples = num_samples
        self.observation_type = observation_type
    
    def _state_to_tensor(self, state: np.ndarray) -> torch.Tensor:
        """Convert state to tensor for network input."""
        if self.observation_type == "image":
            return torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        else:
            return torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
    
    def _get_value_estimate(self, state: np.ndarray) -> float:
        """Get value estimate for a state."""
        with torch.no_grad():
            state_t = self._state_to_tensor(state)
            q_values = self.q_network(state_t)
            return float(q_values.max().item())
    
    def _create_env_copy(self) -> SnakeEnv:
        """Create a copy of the environment for simulation."""
        env_copy = SnakeEnv(
            grid_size=(self.env.game.grid_width, self.env.game.grid_height),
            render_mode=None,
            seed=None,
            observation_type=self.observation_type,
        )
        # Copy game state - need to reset first to initialize properly
        env_copy.game.reset()
        # Now copy the actual state
        from collections import deque
        env_copy.game.snake = deque(copy.deepcopy(list(self.env.game.snake)))
        env_copy.game.direction = self.env.game.direction
        env_copy.game.food = self.env.game.food
        env_copy.game.score = self.env.game.score
        env_copy.game.done = self.env.game.done
        # Update the encoded state
        env_copy.state = env_copy._encode_state()
        return env_copy
    
    def _evaluate_sequence(self, env_copy: SnakeEnv, actions: list[int], depth: int = 0) -> float:
        """Evaluate a sequence of actions."""
        if depth >= len(actions) or env_copy.game.done:
            return self._get_value_estimate(env_copy.state)
        
        action = actions[depth]
        next_state, reward, done, _ = env_copy.step(action)
        
        if done:
            return reward
        
        if depth == len(actions) - 1:
            # Last action in sequence, use value estimate
            return reward + 0.99 * self._get_value_estimate(next_state)
        else:
            # Continue sequence
            return reward + 0.99 * self._evaluate_sequence(env_copy, actions, depth + 1)
    
    def plan(self, state: np.ndarray) -> int:
        """
        Plan by evaluating lookahead sequences.
        
        Args:
            state: Current state
            
        Returns:
            Best action based on lookahead evaluation
        """
        best_action = None
        best_value = float('-inf')
        
        # Try each initial action
        for initial_action in range(len(self.env.ACTIONS)):
            action_values = []
            
            # Sample multiple action sequences starting with this action
            for _ in range(self.num_samples):
                env_copy = self._create_env_copy()
                
                # Generate random sequence of actions
                actions = [initial_action]
                for _ in range(self.lookahead_depth - 1):
                    if env_copy.game.done:
                        break
                    # Use Q-network to guide subsequent actions (greedy)
                    with torch.no_grad():
                        state_t = self._state_to_tensor(env_copy.state)
                        q_values = self.q_network(state_t)
                        next_action = int(torch.argmax(q_values).item())
                    actions.append(next_action)
                
                # Evaluate this sequence
                env_copy = self._create_env_copy()
                value = self._evaluate_sequence(env_copy, actions)
                action_values.append(value)
            
            # Average value for this initial action
            avg_value = np.mean(action_values)
            
            if avg_value > best_value:
                best_value = avg_value
                best_action = initial_action
        
        # Fallback to Q-network
        if best_action is None:
            with torch.no_grad():
                state_t = self._state_to_tensor(state)
                q_values = self.q_network(state_t)
                best_action = int(torch.argmax(q_values).item())
        
        return best_action
