import chess


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


def minimax(board, depth, alpha, beta, maximizing_player):
    """
    Minimax algorithm with Alpha-Beta pruning implementation.
    :param board: Current board state
    :param depth: Depth of the search
    :param alpha: Alpha value (best already explored option for MAX)
    :param beta: Beta value (best already explored option for MIN)
    :param maximizing_player: True if it's the maximizing player's turn
    :return: Evaluation score for the board
    """
    # Check for terminal conditions: checkmate, stalemate, or depth limit
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    legal_moves = list(board.legal_moves)

    if maximizing_player:
        v = float("-inf")
        for move in legal_moves:
            board.push(move)
            v = max(v, minimax(board, depth - 1, alpha, beta, False))
            board.pop()
            alpha = max(alpha, v)
            if beta <= alpha:
                break  # Beta cutoff
        return v
    else:
        v = float("inf")
        for move in legal_moves:
            board.push(move)
            v = min(v, minimax(board, depth - 1, alpha, beta, True))
            board.pop()
            beta = min(beta, v)
            if beta <= alpha:
                break  # Alpha cutoff
        return v


def find_best_move(board, depth):
    """
    Find the best move using minimax algorithm with Alpha-Beta pruning.
    :param board: Current board state
    :param depth: Depth of the search
    :return: Best move and its evaluation score
    """
    legal_moves = list(board.legal_moves)

    if len(legal_moves) == 1:
        return legal_moves[0], evaluate_board(board)

    # Determine if it's the maximizing player's turn
    maximizing_player = board.turn == chess.WHITE

    best_move = None
    best_value = float("-inf") if maximizing_player else float("inf")
    alpha = float("-inf")
    beta = float("inf")

    for move in legal_moves:
        board.push(move)

        # Fetch the evaluation score for the move using Alpha-Beta pruning
        if maximizing_player:
            board_value = minimax(board, depth - 1, alpha, beta, False)
        else:
            board_value = minimax(board, depth - 1, alpha, beta, True)

        board.pop()

        # Update the best move and value based on the evaluation score
        if maximizing_player and board_value > best_value:
            best_value = board_value
            best_move = move
            alpha = max(alpha, best_value)
        elif not maximizing_player and board_value < best_value:
            best_value = board_value
            best_move = move
            beta = min(beta, best_value)

    return best_move, best_value
