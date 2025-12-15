from src.game import DIRECTIONS, SnakeGame


def test_snake_moves_without_collision():
    game = SnakeGame(grid_size=(5, 5), render_mode=None, seed=123)
    start_head = game.snake[0]
    result = game.step(DIRECTIONS["RIGHT"])
    assert not result.collision
    assert len(result.snake) == 3
    assert result.snake[0][0] == start_head[0] + 1


def test_food_not_on_snake():
    game = SnakeGame(grid_size=(5, 5), render_mode=None, seed=42)
    assert game.food not in list(game.snake)


