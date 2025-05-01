import chess
import random
import time
import os
import chess.polyglot


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


class ZobristHashing:
    """
    Implements Zobrist hashing for chess positions.
    Used to generate unique hash keys for board positions.
    """

    def __init__(self):
        # Initialize random numbers for each piece at each position
        self.piece_position = {}
        for piece in [
            chess.PAWN,
            chess.KNIGHT,
            chess.BISHOP,
            chess.ROOK,
            chess.QUEEN,
            chess.KING,
        ]:
            for color in [chess.WHITE, chess.BLACK]:
                for square in chess.SQUARES:
                    self.piece_position[(piece, color, square)] = random.randint(
                        0, 2**64 - 1
                    )

        # Random number for side to move
        self.side_to_move = random.randint(0, 2**64 - 1)

        # Random numbers for castling rights
        self.castling = {}
        for c in [chess.WHITE, chess.BLACK]:
            for side in ["K", "Q"]:
                self.castling[(c, side)] = random.randint(0, 2**64 - 1)

        # Random numbers for en passant files
        self.en_passant = {}
        for file in range(8):
            self.en_passant[file] = random.randint(0, 2**64 - 1)

    def compute_hash(self, board):
        """
        Compute the Zobrist hash for a given board position.
        """
        h = 0

        # Hash pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                h ^= self.piece_position[(piece.piece_type, piece.color, square)]

        # Hash side to move
        if board.turn == chess.WHITE:
            h ^= self.side_to_move

        # Hash castling rights
        if board.has_kingside_castling_rights(chess.WHITE):
            h ^= self.castling[(chess.WHITE, "K")]
        if board.has_queenside_castling_rights(chess.WHITE):
            h ^= self.castling[(chess.WHITE, "Q")]
        if board.has_kingside_castling_rights(chess.BLACK):
            h ^= self.castling[(chess.BLACK, "K")]
        if board.has_queenside_castling_rights(chess.BLACK):
            h ^= self.castling[(chess.BLACK, "Q")]

        # Hash en passant
        if board.ep_square:
            file = chess.square_file(board.ep_square)
            h ^= self.en_passant[file]

        return h


# Transposition table entry flags
EXACT = 0  # Exact score
LOWERBOUND = 1  # Beta cutoff, score is a lower bound
UPPERBOUND = 2  # Alpha cutoff, score is an upper bound


class TranspositionTable:
    """
    Implements a transposition table for chess positions.
    Used to store and lookup previously computed positions.
    """

    def __init__(self, size_mb=64):
        # Calculate number of entries based on size in MB
        # Each entry is approximately 32 bytes
        self.size = int((size_mb * 1024 * 1024) / 32)
        self.table = {}
        self.zobrist = ZobristHashing()

    def store(self, board, depth, score, flag, best_move=None):
        """
        Store a position's evaluation in the table.
        :param board: Current board state
        :param depth: Depth of the search
        :param score: Evaluation score
        :param flag: Type of score (EXACT, LOWERBOUND, UPPERBOUND)
        :param best_move: Best move for the position
        """
        hash_key = self.zobrist.compute_hash(board)

        # If table is full, replace entries (simple replacement strategy)
        if len(self.table) >= self.size:
            # Remove a random entry
            # A more sophisticated approach would be to use aging or depth-preferred replacement
            self.table.pop(next(iter(self.table)))

        self.table[hash_key] = {
            "depth": depth,
            "score": score,
            "flag": flag,
            "best_move": best_move,
        }

    def lookup(self, board):
        """
        Look up a position in the table.
        Returns None if the position is not found.
        :param board: Current board state
        :return: Transposition table entry or None
        """
        hash_key = self.zobrist.compute_hash(board)
        return self.table.get(hash_key)


def minimax(
    board,
    depth,
    alpha,
    beta,
    maximizing_player,
    tt,
    dynamic_depth=True,
    max_depth_extension=2,
    uci_mode=False,
    engine=None,
    use_quiescence=True,
    max_q_depth=5,
):
    """
    Minimax algorithm with Alpha-Beta pruning and transposition table.
    :param board: Current board state
    :param depth: Depth of the search
    :param alpha: Alpha value (best already explored option for MAX)
    :param beta: Beta value (best already explored option for MIN)
    :param maximizing_player: True if it's the maximizing player's turn
    :param tt: Transposition table
    :param dynamic_depth: Whether to use dynamic depth adjustment
    :param max_depth_extension: Maximum additional depth allowed through extensions
    :param uci_mode: Whether to output UCI protocol information
    :param engine: Reference to the engine for node counting
    :param use_quiescence: Whether to use quiescence search at leaf nodes
    :param max_q_depth: Maximum depth for quiescence search
    :return: Evaluation score for the board
    """
    # Count this node
    if engine:
        engine.increment_nodes()

    # Check transposition table
    tt_entry = tt.lookup(board)
    if tt_entry and tt_entry["depth"] >= depth:
        if tt_entry["flag"] == EXACT:
            return tt_entry["score"]
        elif tt_entry["flag"] == LOWERBOUND and tt_entry["score"] > alpha:
            alpha = tt_entry["score"]
        elif tt_entry["flag"] == UPPERBOUND and tt_entry["score"] < beta:
            beta = tt_entry["score"]

        if alpha >= beta:
            return tt_entry["score"]

    # Check for terminal conditions
    if board.is_game_over():
        return evaluate_board(board)

    # If we've reached our depth limit, perform quiescence search
    if depth <= 0:
        if use_quiescence:
            return quiescence_search(
                board, alpha, beta, maximizing_player, tt, engine, max_q_depth
            )
        else:
            return evaluate_board(board)

    legal_moves = list(board.legal_moves)
    best_move = None

    # Implement move ordering for better pruning efficiency
    # Order moves to examine captures and checks first
    if dynamic_depth:
        legal_moves.sort(key=lambda move: board.is_capture(move), reverse=True)

    if maximizing_player:
        best_value = float("-inf")
        for move in legal_moves:
            board.push(move)

            # Dynamic depth adjustment with a limit on extension
            extension = 0
            if (
                dynamic_depth
                and max_depth_extension > 0
                and should_extend_search(board, move)
            ):
                extension = 1

            value = minimax(
                board,
                depth - 1 + extension,
                alpha,
                beta,
                False,
                tt,
                dynamic_depth,
                max_depth_extension - extension,
                uci_mode,
                engine,
                use_quiescence,
                max_q_depth,
            )
            board.pop()

            if value > best_value:
                best_value = value
                best_move = move

            alpha = max(alpha, best_value)
            if beta <= alpha:
                break  # Beta cutoff

        # Store the result in the transposition table
        if best_value <= alpha:
            tt.store(board, depth, best_value, UPPERBOUND, best_move)
        else:
            tt.store(board, depth, best_value, EXACT, best_move)

        return best_value
    else:
        best_value = float("inf")
        for move in legal_moves:
            board.push(move)

            # Dynamic depth adjustment with a limit on extension
            extension = 0
            if (
                dynamic_depth
                and max_depth_extension > 0
                and should_extend_search(board, move)
            ):
                extension = 1

            value = minimax(
                board,
                depth - 1 + extension,
                alpha,
                beta,
                True,
                tt,
                dynamic_depth,
                max_depth_extension - extension,
                uci_mode,
                engine,
                use_quiescence,
                max_q_depth,
            )
            board.pop()

            if value < best_value:
                best_value = value
                best_move = move

            beta = min(beta, best_value)
            if beta <= alpha:
                break  # Alpha cutoff

        # Store the result in the transposition table
        if best_value >= beta:
            tt.store(board, depth, best_value, LOWERBOUND, best_move)
        else:
            tt.store(board, depth, best_value, EXACT, best_move)

        return best_value


def find_best_move(
    board,
    depth,
    tt=None,
    dynamic_depth=True,
    max_depth_extension=2,
    uci_mode=False,
    engine=None,
    use_quiescence=True,
    max_q_depth=5,
):
    """
    Find the best move using minimax algorithm with Alpha-Beta pruning and transposition table.
    :param board: Current board state
    :param depth: Depth of the search
    :param tt: Transposition table
    :param dynamic_depth: Whether to use dynamic depth adjustment
    :param max_depth_extension: Maximum additional depth allowed through extensions
    :param uci_mode: Whether to output UCI protocol information
    :param engine: Reference to the engine for node counting
    :param use_quiescence: Whether to use quiescence search at leaf nodes
    :param max_q_depth: Maximum depth for quiescence search
    :return: Best move and its evaluation score
    """
    if tt is None:
        tt = TranspositionTable()

    legal_moves = list(board.legal_moves)

    if len(legal_moves) == 1:
        return legal_moves[0], evaluate_board(board)

    maximizing_player = board.turn == chess.WHITE
    best_move = None
    best_value = float("-inf") if maximizing_player else float("inf")
    alpha = float("-inf")
    beta = float("inf")

    # Implement move ordering for better pruning efficiency
    if dynamic_depth:
        legal_moves.sort(key=lambda move: board.is_capture(move), reverse=True)

    for move in legal_moves:
        board.push(move)

        # Count this node
        if engine:
            engine.increment_nodes()

        # Dynamic depth adjustment with limit
        extension = 0
        if (
            dynamic_depth
            and max_depth_extension > 0
            and should_extend_search(board, move)
        ):
            extension = 1

        value = minimax(
            board,
            depth - 1 + extension,
            alpha,
            beta,
            not maximizing_player,
            tt,
            dynamic_depth,
            max_depth_extension - extension,
            uci_mode,
            engine,
            use_quiescence,
            max_q_depth,
        )

        board.pop()

        if maximizing_player and value > best_value:
            best_value = value
            best_move = move
            alpha = max(alpha, best_value)
        elif not maximizing_player and value < best_value:
            best_value = value
            best_move = move
            beta = min(beta, best_value)

    # Store the result in the transposition table
    tt.store(board, depth, best_value, EXACT, best_move)

    return best_move, best_value


def iterative_deepening_search(
    board,
    max_depth,
    time_limit=None,
    tt=None,
    dynamic_depth=True,
    max_depth_extension=2,
    uci_mode=False,
    engine=None,
    use_quiescence=True,
    max_q_depth=5,
):
    """
    Perform iterative deepening search to find the best move.
    Gradually increases search depth and can stop based on time constraints.

    :param board: Current board state
    :param max_depth: Maximum depth to search
    :param time_limit: Maximum time in seconds for the search (None for no limit)
    :param tt: Transposition table
    :param dynamic_depth: Whether to use dynamic depth adjustment
    :param max_depth_extension: Maximum additional depth allowed through extensions
    :param uci_mode: Whether to output UCI protocol information
    :param engine: Reference to the engine for node counting and info reporting
    :param use_quiescence: Whether to use quiescence search at leaf nodes
    :param max_q_depth: Maximum depth for quiescence search
    :return: Best move, its evaluation score, and actual depth reached
    """
    if tt is None:
        tt = TranspositionTable()

    start_time = time.time()
    best_move = None
    best_value = 0
    reached_depth = 0

    # For single legal move, return immediately
    legal_moves = list(board.legal_moves)
    if len(legal_moves) == 1:
        return legal_moves[0], evaluate_board(board), 0

    # Start with depth 1 and iteratively increase
    for current_depth in range(1, max_depth + 1):
        # Check if we've exceeded our time limit
        elapsed = time.time() - start_time
        if time_limit and elapsed > time_limit * 0.8:
            # Return the best move from the previous iteration
            # We don't want to use partial results from an incomplete search
            break

        # Perform search at current depth
        move, value = find_best_move(
            board,
            current_depth,
            tt,
            dynamic_depth,
            max_depth_extension,
            uci_mode=uci_mode,
            engine=engine,
            use_quiescence=use_quiescence,
            max_q_depth=max_q_depth,
        )

        # Update our best move
        best_move = move
        best_value = value
        reached_depth = current_depth

        # Report progress in UCI format
        if uci_mode and engine:
            elapsed_ms = int((time.time() - start_time) * 1000)
            engine.report_search_info(
                current_depth, value, move, engine.nodes_searched, elapsed_ms
            )

        # Early termination conditions
        # If we found a checkmate, no need to search deeper
        if abs(value) > 9000:  # Close to checkmate score
            break

    return best_move, best_value, reached_depth


def should_extend_search(board, move):
    """
    Determines if the search depth should be dynamically extended for this move.
    Extends search for captures, checks, and critical positions.

    :param board: Current board state
    :param move: The move to evaluate
    :return: True if search should be extended, False otherwise
    """
    # Extend search for captures
    if board.is_capture(move):
        return True

    # Extend search for checks
    board.push(move)
    is_check = board.is_check()
    board.pop()
    if is_check:
        return True

    # Can add more conditions for critical positions

    return False


class OpeningBook:
    """
    Manages access to a Polyglot opening book.
    Provides methods to retrieve moves from the opening book based on the current board position.
    """

    def __init__(self, book_path=None):
        """
        Initialize the opening book.
        :param book_path: Path to the Polyglot opening book file
        """
        self.book_path = book_path
        self.enabled = book_path is not None and os.path.exists(book_path)
        self._reader = None

    def get_move(self, board, minimum_weight=1):
        """
        Get a move from the opening book for the current board position.
        :param board: Current board state
        :param minimum_weight: Minimum weight for opening moves
        :return: A selected move or None if no move is found
        """
        if not self.enabled:
            return None

        try:
            with chess.polyglot.open_reader(self.book_path) as reader:
                try:
                    # Try weighted choice based on entry weights
                    entry = reader.weighted_choice(board)
                    return entry.move
                except IndexError:
                    # No entry found
                    return None
        except Exception as e:
            print(f"Error accessing opening book: {e}")
            return None

    def get_all_moves(self, board, minimum_weight=1):
        """
        Get all possible moves from the opening book for the current board position.
        :param board: Current board state
        :param minimum_weight: Minimum weight for opening moves
        :return: List of (move, weight) tuples or empty list if no moves found
        """
        if not self.enabled:
            return []

        moves = []
        try:
            with chess.polyglot.open_reader(self.book_path) as reader:
                for entry in reader.find_all(board, minimum_weight=minimum_weight):
                    moves.append((entry.move, entry.weight))
            return moves
        except Exception as e:
            print(f"Error accessing opening book: {e}")
            return []

    def is_enabled(self):
        """
        Check if opening book is enabled and available.
        :return: True if the opening book is enabled and file exists
        """
        return self.enabled


def quiescence_search(
    board, alpha, beta, maximizing_player, tt, engine=None, max_q_depth=5
):
    """
    Quiescence search to resolve tactical sequences beyond the regular search depth.
    This focuses on captures and checks to reach a "quiet" position before evaluation.

    :param board: Current board state
    :param alpha: Alpha value (best already explored option for MAX)
    :param beta: Beta value (best already explored option for MIN)
    :param maximizing_player: True if it's the maximizing player's turn
    :param tt: Transposition table
    :param engine: Reference to the engine for node counting
    :param max_q_depth: Maximum additional depth for quiescence search
    :return: Evaluation score for the board
    """
    # Count this node
    if engine:
        engine.increment_nodes()

    # Check transposition table
    tt_entry = tt.lookup(board)
    if tt_entry and tt_entry["flag"] == EXACT:
        return tt_entry["score"]

    # Stand-pat evaluation (evaluate without any moves)
    stand_pat = evaluate_board(board)

    # Fail-hard beta cutoff
    if maximizing_player and stand_pat >= beta:
        return beta
    if not maximizing_player and stand_pat <= alpha:
        return alpha

    # Update alpha if stand-pat is better
    if maximizing_player and stand_pat > alpha:
        alpha = stand_pat
    if not maximizing_player and stand_pat < beta:
        beta = stand_pat

    # If max_q_depth reached, return stand-pat evaluation
    if max_q_depth <= 0:
        return stand_pat

    # Generate only captures and checks
    legal_moves = list(board.legal_moves)
    # Filter to keep only captures (and potentially checks)
    captures = [move for move in legal_moves if board.is_capture(move)]

    # Check for checks separately to avoid overhead when there are captures to look at
    if not captures:
        # Look for checks only if no captures exist
        checks = []
        for move in legal_moves:
            board.push(move)
            is_check = board.is_check()
            board.pop()
            if is_check:
                checks.append(move)
        # Add checks to the moves to consider
        captures.extend(checks)

    # If no captures or checks, return stand-pat
    if not captures:
        return stand_pat

    # Order moves by MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
    captures.sort(key=lambda move: mvv_lva_score(board, move), reverse=True)

    if maximizing_player:
        for move in captures:
            board.push(move)
            score = quiescence_search(
                board, alpha, beta, False, tt, engine, max_q_depth - 1
            )
            board.pop()

            if score >= beta:
                return beta  # Fail-hard beta cutoff
            if score > alpha:
                alpha = score

        return alpha
    else:
        for move in captures:
            board.push(move)
            score = quiescence_search(
                board, alpha, beta, True, tt, engine, max_q_depth - 1
            )
            board.pop()

            if score <= alpha:
                return alpha  # Fail-hard alpha cutoff
            if score < beta:
                beta = score

        return beta


def mvv_lva_score(board, move):
    """
    Calculate the MVV-LVA (Most Valuable Victim - Least Valuable Attacker) score for a move.
    This is used to order capture moves in quiescence search.

    :param board: Current board state
    :param move: The move to evaluate
    :return: MVV-LVA score (higher means more promising capture)
    """
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 100,  # Very high value for king captures
    }

    # If it's not a capture, return lowest priority
    if not board.is_capture(move):
        return -1

    # For en passant captures
    if board.is_en_passant(move):
        # Pawn captures pawn
        return piece_values[chess.PAWN] * 10 - piece_values[chess.PAWN]

    # Get victim value (the captured piece)
    victim_square = move.to_square
    victim = board.piece_at(victim_square)
    victim_value = piece_values[victim.piece_type] if victim else 0

    # Get attacker value
    attacker_square = move.from_square
    attacker = board.piece_at(attacker_square)
    attacker_value = piece_values[attacker.piece_type] if attacker else 0

    # Calculate MVV-LVA score: 10 * victim value - attacker value
    # This prioritizes capturing valuable pieces with less valuable pieces
    return victim_value * 10 - attacker_value
