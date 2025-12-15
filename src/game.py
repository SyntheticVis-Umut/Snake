from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

try:
    import pygame  # type: ignore
except ImportError:  # pragma: no cover - pygame not installed in some envs
    pygame = None

Vec2 = Tuple[int, int]


def add_pos(a: Vec2, b: Vec2) -> Vec2:
    return a[0] + b[0], a[1] + b[1]


def opposite(a: Vec2, b: Vec2) -> bool:
    return a[0] == -b[0] and a[1] == -b[1]


DIRECTIONS = {
    "UP": (0, -1),
    "RIGHT": (1, 0),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
}


@dataclass
class StepResult:
    snake: List[Vec2]
    food: Vec2
    score: int
    done: bool
    ate_food: bool
    collision: bool


class SnakeGame:
    def __init__(
        self,
        grid_size: Tuple[int, int] = (20, 20),
        cell_size: int = 20,
        speed: int = 10,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.grid_width, self.grid_height = grid_size
        self.cell_size = cell_size
        self.speed = speed
        self.render_mode = render_mode
        self.random = random.Random(seed)

        self.snake: Deque[Vec2] = deque()
        self.direction: Vec2 = DIRECTIONS["RIGHT"]
        self.food: Vec2 = (0, 0)
        self.score = 0
        self.done = False

        self._window = None
        self._clock = None

        if self.render_mode:
            self._init_render()

        self.reset()

    def _init_render(self) -> None:
        if pygame is None:
            raise ImportError("pygame is required for rendering")

        pygame.init()
        width_px = self.grid_width * self.cell_size
        height_px = self.grid_height * self.cell_size
        self._window = pygame.display.set_mode((width_px, height_px))
        pygame.display.set_caption("Snake RL")
        self._clock = pygame.time.Clock()

    def reset(self) -> StepResult:
        self.snake.clear()
        center = (self.grid_width // 2, self.grid_height // 2)
        self.direction = DIRECTIONS["RIGHT"]
        self.snake.append(center)
        self.snake.append(add_pos(center, (-1, 0)))
        self.snake.append(add_pos(center, (-2, 0)))

        self.score = 0
        self.done = False
        self.food = self._random_food()

        return self._result(ate_food=False, collision=False)

    def _random_food(self) -> Vec2:
        available = [
            (x, y)
            for x in range(self.grid_width)
            for y in range(self.grid_height)
            if (x, y) not in self.snake
        ]
        return self.random.choice(available) if available else (-1, -1)

    def step(self, new_direction: Optional[Vec2] = None) -> StepResult:
        if self.done:
            return self._result(ate_food=False, collision=True)

        if new_direction and not opposite(new_direction, self.direction):
            self.direction = new_direction

        new_head = add_pos(self.snake[0], self.direction)
        collision = self._is_collision(new_head)
        ate_food = False

        if collision:
            self.done = True
            return self._result(ate_food=False, collision=True)

        self.snake.appendleft(new_head)

        if new_head == self.food:
            self.score += 1
            ate_food = True
            self.food = self._random_food()
        else:
            self.snake.pop()

        return self._result(ate_food=ate_food, collision=False)

    def _is_collision(self, pos: Vec2) -> bool:
        x, y = pos
        if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
            return True
        if pos in list(self.snake):
            return True
        return False

    def _result(self, ate_food: bool, collision: bool) -> StepResult:
        return StepResult(
            snake=list(self.snake),
            food=self.food,
            score=self.score,
            done=self.done,
            ate_food=ate_food,
            collision=collision,
        )

    def render(self) -> None:
        if not self.render_mode:
            return
        if pygame is None:
            raise ImportError("pygame is required for rendering")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True

        assert self._window is not None
        assert self._clock is not None

        self._window.fill((20, 20, 20))
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(self._window, (30, 30, 30), rect, 1)

        for i, (x, y) in enumerate(self.snake):
            color = (0, 200, 0) if i == 0 else (0, 150, 0)
            rect = pygame.Rect(
                x * self.cell_size,
                y * self.cell_size,
                self.cell_size,
                self.cell_size,
            )
            pygame.draw.rect(self._window, color, rect)

        fx, fy = self.food
        if fx >= 0 and fy >= 0:
            food_rect = pygame.Rect(
                fx * self.cell_size,
                fy * self.cell_size,
                self.cell_size,
                self.cell_size,
            )
            pygame.draw.rect(self._window, (200, 50, 50), food_rect)

        pygame.display.flip()
        self._clock.tick(self.speed)

    def close(self) -> None:
        if pygame:
            pygame.quit()


