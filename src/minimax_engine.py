from .utils import *
import time


class MinimaxEngine:
    """
    A simple chess engine using the Minimax algorithm.
    This engine evaluates the board state and selects the best move
    based on the Minimax algorithm with a specified depth.
    """

    def __init__(self, depth=3):
        self.depth = depth

    def get_move(self, board):
        """
        Get the best move for the current board state.
        :param board: Current board state
        :return: Best move for the current board state
        """
        print(f"Thinking... (Depth: {self.depth})")
        start_time = time.time()
        best_move, value = find_best_move(board, self.depth)
        end_time = time.time()

        print(f"Best Move: {board.san(best_move)}")
        print(f"Evaluation Score: {value}")
        print(f"Thinking Time: {end_time - start_time:.2f} 秒")

        return best_move
