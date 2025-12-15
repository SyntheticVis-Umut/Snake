from __future__ import annotations

import sys

import pygame

from src.game import DIRECTIONS, SnakeGame


def main() -> None:
    game = SnakeGame(render_mode="human")
    direction = game.direction

    while not game.done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    direction = DIRECTIONS["UP"]
                elif event.key == pygame.K_DOWN:
                    direction = DIRECTIONS["DOWN"]
                elif event.key == pygame.K_LEFT:
                    direction = DIRECTIONS["LEFT"]
                elif event.key == pygame.K_RIGHT:
                    direction = DIRECTIONS["RIGHT"]

        game.step(direction)
        game.render()

    print(f"Game over! Final score: {game.score}")
    game.close()


if __name__ == "__main__":
    main()


