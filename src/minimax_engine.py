import chess
import time


def evaluate_board(board):
    """
    Evaluate the current board state.
    Positive values favor white, negative values favor black.
    Returns a score based on material balance, center control, and mobility.
    :param board: Current board state
    :return: Evaluation score for the board
    """
    if board.is_checkmate():
        return -10000 if board.turn else 10000

    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_fifty_moves()
        or board.is_repetition(3)
    ):
        return 0

    # basic piece values
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000,
    }

    # calculate material balance
    evaluation = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            # If the piece is white, add its value; if black, subtract its value
            value = piece_values[piece.piece_type]
            evaluation += value if piece.color == chess.WHITE else -value

    # Center control evaluation - reward control of center squares
    center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
    for square in center_squares:
        # Examine the number of pieces attacking the center squares
        attackers = board.attackers(chess.WHITE, square)
        evaluation += len(attackers) * 10
        attackers = board.attackers(chess.BLACK, square)
        evaluation -= len(attackers) * 10

    # Mobility evaluation - reward the number of legal moves
    original_turn = board.turn

    board.turn = chess.WHITE
    white_moves = len(list(board.legal_moves))

    board.turn = chess.BLACK
    black_moves = len(list(board.legal_moves))

    board.turn = original_turn

    evaluation += (white_moves - black_moves) * 5

    return evaluation


def minimax(board, depth, maximizing_player):
    """
    Minimax algorithm implementation.
    :param board: Current board state
    :param depth: Depth of the search
    :param maximizing_player: True if it's the maximizing player's turn
    :return: Evaluation score for the board
    """
    # Check for terminal conditions: checkmate, stalemate, or depth limit
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    legal_moves = list(board.legal_moves)

    if maximizing_player:
        max_eval = float("-inf")
        for move in legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, False)
            board.pop()
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float("inf")
        for move in legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, True)
            board.pop()
            min_eval = min(min_eval, eval)
        return min_eval


def find_best_move(board, depth):
    """
    Find the best move using minimax algorithm.
    :param board: Current board state
    :param depth: Depth of the search
    :return: Best move and its evaluation score
    """
    legal_moves = list(board.legal_moves)

    if len(legal_moves) == 1:
        return legal_moves[0], evaluate_board(board)

    # Determine if it's the maximizing player's turn
    # If it's white's turn, maximizing_player is True; otherwise, it's False
    maximizing_player = board.turn == chess.WHITE

    best_move = None
    best_value = float("-inf") if maximizing_player else float("inf")

    for move in legal_moves:
        board.push(move)

        # Fetch the evaluation score for the move
        board_value = minimax(board, depth - 1, not maximizing_player)

        board.pop()

        # Update the best move and value based on the evaluation score
        if maximizing_player and board_value > best_value:
            best_value = board_value
            best_move = move
        elif not maximizing_player and board_value < best_value:
            best_value = board_value
            best_move = move

    return best_move, best_value


class MinimaxChessEngine:
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


# Demo
def play_game():
    board = chess.Board()
    engine = MinimaxChessEngine(depth=3)

    # Simulate a simple game
    for _ in range(4):
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
