import chess
import random


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
        for piece in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
            for color in [chess.WHITE, chess.BLACK]:
                for square in chess.SQUARES:
                    self.piece_position[(piece, color, square)] = random.randint(0, 2**64 - 1)
        
        # Random number for side to move
        self.side_to_move = random.randint(0, 2**64 - 1)
        
        # Random numbers for castling rights
        self.castling = {}
        for c in [chess.WHITE, chess.BLACK]:
            for side in ['K', 'Q']:
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
            h ^= self.castling[(chess.WHITE, 'K')]
        if board.has_queenside_castling_rights(chess.WHITE):
            h ^= self.castling[(chess.WHITE, 'Q')]
        if board.has_kingside_castling_rights(chess.BLACK):
            h ^= self.castling[(chess.BLACK, 'K')]
        if board.has_queenside_castling_rights(chess.BLACK):
            h ^= self.castling[(chess.BLACK, 'Q')]
        
        # Hash en passant
        if board.ep_square:
            file = chess.square_file(board.ep_square)
            h ^= self.en_passant[file]
        
        return h


# Transposition table entry flags
EXACT = 0    # Exact score
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
            'depth': depth,
            'score': score,
            'flag': flag,
            'best_move': best_move
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


def minimax(board, depth, alpha, beta, maximizing_player, tt):
    """
    Minimax algorithm with Alpha-Beta pruning and transposition table.
    :param board: Current board state
    :param depth: Depth of the search
    :param alpha: Alpha value (best already explored option for MAX)
    :param beta: Beta value (best already explored option for MIN)
    :param maximizing_player: True if it's the maximizing player's turn
    :param tt: Transposition table
    :return: Evaluation score for the board
    """

    # Check transposition table
    tt_entry = tt.lookup(board)
    if tt_entry and tt_entry['depth'] >= depth:
        if tt_entry['flag'] == EXACT:
            return tt_entry['score']
        elif tt_entry['flag'] == LOWERBOUND and tt_entry['score'] > alpha:
            alpha = tt_entry['score']
        elif tt_entry['flag'] == UPPERBOUND and tt_entry['score'] < beta:
            beta = tt_entry['score']
        
        if alpha >= beta:
            return tt_entry['score']
    
    # Check for terminal conditions
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)
    
    legal_moves = list(board.legal_moves)
    best_move = None
    
    if maximizing_player:
        best_value = float("-inf")
        for move in legal_moves:
            board.push(move)
            value = minimax(board, depth - 1, alpha, beta, False, tt)
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
            value = minimax(board, depth - 1, alpha, beta, True, tt)
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


def find_best_move(board, depth, tt=None):
    """
    Find the best move using minimax algorithm with Alpha-Beta pruning and transposition table.
    :param board: Current board state
    :param depth: Depth of the search
    :param tt: Transposition table
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
    
    for move in legal_moves:
        board.push(move)
        
        value = minimax(board, depth - 1, alpha, beta, not maximizing_player, tt)
        
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
