from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from src.game import DIRECTIONS, SnakeGame


class SnakeEnv:
    """Lightweight Gym-like wrapper for the Snake game."""

    ACTIONS = (0, 1, 2)  # straight, left, right relative to current heading

    def __init__(
        self,
        grid_size: Tuple[int, int] = (20, 20),
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        observation_type: str = "features",  # "features" or "image"
    ) -> None:
        self.game = SnakeGame(grid_size=grid_size, render_mode=render_mode, seed=seed)
        self.observation_type = observation_type
        self.directions = [
            DIRECTIONS["UP"],
            DIRECTIONS["RIGHT"],
            DIRECTIONS["DOWN"],
            DIRECTIONS["LEFT"],
        ]
        self.state: np.ndarray = self._encode_state()

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.game.random.seed(seed)
            np.random.seed(seed)
        self.game.reset()
        self.state = self._encode_state()
        return self.state

    def step(self, action: int):
        assert action in self.ACTIONS, "Invalid action"
        new_direction = self._next_direction(action)
        result = self.game.step(new_direction)
        self.state = self._encode_state()
        reward = self._reward(result.ate_food, result.done)
        info = {"score": result.score}
        return self.state, reward, result.done, info

    def render(self) -> None:
        self.game.render()

    def close(self) -> None:
        self.game.close()

    def _next_direction(self, action: int):
        idx = self.directions.index(self.game.direction)
        if action == 0:  # straight
            return self.directions[idx]
        if action == 1:  # left turn
            return self.directions[(idx - 1) % len(self.directions)]
        if action == 2:  # right turn
            return self.directions[(idx + 1) % len(self.directions)]
        return self.directions[idx]

    def _reward(self, ate_food: bool, done: bool) -> float:
        if done:
            return -10.0
        if ate_food:
            return 10.0
        return -0.1

    def _encode_state(self) -> np.ndarray:
        if self.observation_type == "image":
            return self._encode_image_state()
        else:
            return self._encode_feature_state()

    def _encode_feature_state(self) -> np.ndarray:
        """Original 11-feature state encoding."""
        head_x, head_y = self.game.snake[0]
        dir_x, dir_y = self.game.direction

        left_dir = self._rotate_left(self.game.direction)
        right_dir = self._rotate_right(self.game.direction)

        danger_straight = self._is_danger(add_vec((head_x, head_y), self.game.direction))
        danger_left = self._is_danger(add_vec((head_x, head_y), left_dir))
        danger_right = self._is_danger(add_vec((head_x, head_y), right_dir))

        food_left = self.game.food[0] < head_x
        food_right = self.game.food[0] > head_x
        food_up = self.game.food[1] < head_y
        food_down = self.game.food[1] > head_y

        direction_encoding = (
            dir_x == 0 and dir_y == -1,  # up
            dir_x == 1 and dir_y == 0,  # right
            dir_x == 0 and dir_y == 1,  # down
            dir_x == -1 and dir_y == 0,  # left
        )

        state = np.array(
            [
                danger_straight,
                danger_left,
                danger_right,
                *direction_encoding,
                food_left,
                food_right,
                food_up,
                food_down,
            ],
            dtype=np.float32,
        )
        return state

    def _encode_image_state(self) -> np.ndarray:
        """CNN-friendly image state: 3 channels (body, head, food)."""
        h, w = self.game.grid_height, self.game.grid_width
        # Channel 0: snake body (1 where body exists, 0 otherwise)
        # Channel 1: snake head (1 at head position, 0 otherwise)
        # Channel 2: food (1 at food position, 0 otherwise)
        state = np.zeros((3, h, w), dtype=np.float32)
        
        # Channel 0: body (all snake segments)
        for x, y in self.game.snake:
            if 0 <= x < w and 0 <= y < h:
                state[0, y, x] = 1.0
        
        # Channel 1: head (only first segment)
        if self.game.snake:
            head_x, head_y = self.game.snake[0]
            if 0 <= head_x < w and 0 <= head_y < h:
                state[1, head_y, head_x] = 1.0
        
        # Channel 2: food
        food_x, food_y = self.game.food
        if 0 <= food_x < w and 0 <= food_y < h:
            state[2, food_y, food_x] = 1.0
        
        return state

    def _is_danger(self, pos) -> bool:
        x, y = pos
        if x < 0 or x >= self.game.grid_width or y < 0 or y >= self.game.grid_height:
            return True
        if pos in list(self.game.snake):
            return True
        return False

    def _rotate_left(self, direction):
        idx = self.directions.index(direction)
        return self.directions[(idx - 1) % len(self.directions)]

    def _rotate_right(self, direction):
        idx = self.directions.index(direction)
        return self.directions[(idx + 1) % len(self.directions)]


def add_vec(a, b):
    return a[0] + b[0], a[1] + b[1]


