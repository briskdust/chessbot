from src.minimax_engine import MinimaxEngine
import chess


# Demo
def play_game():
    board = chess.Board()
    engine = MinimaxEngine(depth=5)

    round_number = 2

    # Simulate a simple game
    for _ in range(100):
        if board.is_game_over():
            break

        print(f"\nRound {round_number // 2}")
        round_number += 1
        print("Current Board:")
        print(board)

        move = engine.get_move(board)
        board.push(move)

    print("\nFinal Board:")
    print(board)

    # print the result of the game
    if board.is_checkmate():
        winner = "black" if board.turn else "white"
        print(f"Game Over: {winner}wins(Checkmate)")
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
