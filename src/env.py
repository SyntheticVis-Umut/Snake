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
    ) -> None:
        self.game = SnakeGame(grid_size=grid_size, render_mode=render_mode, seed=seed)
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


