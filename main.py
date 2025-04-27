from src.minimax_engine import MinimaxEngine
import chess
import time


# Demo
def play_game():
    board = chess.Board()
    # Initialize engine with max depth 5, 128MB transposition table, 5 second time limit,
    # dynamic depth adjustment, and max depth extension of 2
    engine = MinimaxEngine(
        depth=5, tt_size_mb=128, time_limit=5, dynamic_depth=True, max_depth_extension=2
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
