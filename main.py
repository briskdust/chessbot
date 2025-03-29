from src.minimax_engine import MinimaxEngine
import chess


# Demo
def play_game():
    board = chess.Board()
    engine = MinimaxEngine(depth=3)

    # Simulate a simple game
    for _ in range(40):
        if board.is_game_over():
            break

        print("\nCurrent Board:")
        print(board)

        move = engine.get_move(board)
        board.push(move)

    print("\nFinal Board:")
    print(board)


if __name__ == "__main__":
    play_game()
