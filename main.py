from src.minimax_engine import MinimaxEngine
import chess
import time
import os
import yaml


def load_config(config_path="config.yml"):
    """
    Load configuration from YAML file
    :param config_path: Path to the configuration file
    :return: Dictionary with configuration settings
    """
    try:
        with open(config_path, "r") as config_file:
            return yaml.safe_load(config_file)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return {
            "engine": {
                "search_depth": 5,
                "dynamic_depth": True,
                "max_depth_extension": 2,
                "tt_size_mb": 128,
                "time_limit": 5,
            },
            "opening_book": {
                "use_book": True,
                "book_path": "Cerebellum.bin",
            },
        }


# Demo
def play_game(config_path="config.yml"):
    board = chess.Board()

    # Load configuration
    config = load_config(config_path)

    # Get engine configuration
    engine_config = config.get("engine", {})
    book_config = config.get("opening_book", {})

    # Check for opening book file
    book_path = book_config.get("book_path")
    use_book = book_config.get("use_book", True)

    if use_book and book_path:
        if not os.path.exists(book_path):
            print(f"Warning: Opening book file not found at {book_path}")
            print("Engine will run without opening book")
            book_path = None
        else:
            print(f"Found opening book at {book_path}")

    # Initialize engine with configuration settings
    engine = MinimaxEngine(
        depth=engine_config.get("search_depth", 5),
        tt_size_mb=engine_config.get("tt_size_mb", 128),
        time_limit=engine_config.get("time_limit", 5),
        dynamic_depth=engine_config.get("dynamic_depth", True),
        max_depth_extension=engine_config.get("max_depth_extension", 2),
        book_path=book_path,
        use_book=use_book,
    )

    round_number = 2

    # Simulate a simple game
    for _ in range(100):
        if board.is_game_over():
            break

        print(f"\nRound {round_number // 2}")
        round_number += 1
        print("Current Board:")
        print(board)

        # Get time before move calculation to test time management
        start_time = time.time()
        move = engine.get_move(board)
        actual_time = time.time() - start_time
        print(f"Actual time used: {actual_time:.2f} seconds")

        board.push(move)

    print("\nFinal Board:")
    print(board)

    # print the result of the game
    if board.is_checkmate():
        winner = "black" if board.turn else "white"
        print(f"Game Over: {winner} wins (Checkmate)")
    elif board.is_stalemate():
        print("Game Over: Stalemate")
    elif board.is_insufficient_material():
        print("Game Over: Insufficient Material")
    elif board.is_repetition(3):
        print("Game Over: Threefold Repetition")
    elif board.is_fifty_moves():
        print("Game Over: Fifty-Move Rule")
    else:
        print("Game Over")


if __name__ == "__main__":
    play_game()
