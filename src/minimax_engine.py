from .utils import *
import time


class MinimaxEngine:
    """
    A chess engine using the Minimax algorithm with iterative deepening.
    This engine evaluates the board state and selects the best move
    based on the Minimax algorithm with Alpha-Beta pruning,
    transposition tables, and iterative deepening.
    It can also use an opening book for the early game.
    """

    def __init__(
        self,
        depth=3,
        tt_size_mb=128,
        time_limit=None,
        dynamic_depth=True,
        max_depth_extension=2,
        book_path=None,
        use_book=True,
    ):
        """
        Initialize the engine.
        :param depth: Maximum search depth
        :param tt_size_mb: Transposition table size in megabytes
        :param time_limit: Time limit for move calculation in seconds (None for no limit)
        :param dynamic_depth: Whether to use dynamic depth adjustment
        :param max_depth_extension: Maximum additional depth allowed through extensions
        :param book_path: Path to the opening book file
        :param use_book: Whether to use the opening book
        """
        self.max_depth = depth
        self.time_limit = time_limit
        self.dynamic_depth = dynamic_depth
        self.max_depth_extension = max_depth_extension
        self.transposition_table = TranspositionTable(size_mb=tt_size_mb)
        self.opening_book = OpeningBook(book_path) if use_book else None
        self.use_book = use_book

    def get_move(self, board, time_limit=None):
        """
        Get the best move for the current board state.
        First tries the opening book if enabled, then falls back to search.
        :param board: Current board state
        :param time_limit: Optional time limit override for this move
        :return: Best move for the current board state
        """
        # Try opening book first if enabled
        if self.use_book and self.opening_book and self.opening_book.is_enabled():
            book_move = self.opening_book.get_move(board)
            if book_move:
                print(f"Book Move: {board.san(book_move)}")
                return book_move

        # Fall back to search if no book move or book is disabled
        # Use instance time limit if no override is provided
        actual_time_limit = time_limit if time_limit is not None else self.time_limit

        print(
            f"Thinking... (Max Depth: {self.max_depth}, Dynamic Depth: {self.dynamic_depth})"
        )
        start_time = time.time()

        # Use iterative deepening to find the best move
        best_move, value, reached_depth = iterative_deepening_search(
            board,
            self.max_depth,
            actual_time_limit,
            self.transposition_table,
            self.dynamic_depth,
            self.max_depth_extension,
        )

        end_time = time.time()
        thinking_time = end_time - start_time

        print(f"Best Move: {board.san(best_move)}")
        print(f"Evaluation Score: {value}")
        print(f"Depth Reached: {reached_depth}")
        print(f"Thinking Time: {thinking_time:.2f} seconds")

        return best_move

    def adjust_time_management(self, remaining_time, moves_to_go=30):
        """
        Adjust time limit based on game phase and remaining time.
        :param remaining_time: Remaining time in seconds
        :param moves_to_go: Estimated number of moves remaining in the game
        :return: Suggested time limit for the current move
        """
        # Basic time management - allocate time based on estimated moves remaining
        base_time = remaining_time / moves_to_go

        # Use more time in the opening/middle game, less in the endgame
        # This is a simple approach and can be enhanced
        return min(
            remaining_time * 0.25, base_time * 1.5
        )  # Never use more than 25% of remaining time
