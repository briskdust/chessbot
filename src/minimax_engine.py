from .utils import *
import time


class MinimaxEngine:
    """
    A simple chess engine using the Minimax algorithm.
    This engine evaluates the board state and selects the best move
    based on the Minimax algorithm with a specified depth.
    """

    def __init__(self, depth=3, tt_size_mb=128):
        self.depth = depth
        self.transposition_table = TranspositionTable(size_mb=tt_size_mb)

    def get_move(self, board):
        """
        Get the best move for the current board state.
        :param board: Current board state
        :return: Best move for the current board state
        """
        print(f"Thinking... (Depth: {self.depth})")
        start_time = time.time()
        best_move, value = find_best_move(board, self.depth, self.transposition_table)
        end_time = time.time()

        print(f"Best Move: {board.san(best_move)}")
        print(f"Evaluation Score: {value}")
        print(f"Thinking Time: {end_time - start_time:.2f} seconds")

        return best_move
