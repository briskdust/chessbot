import chess
import random
import time
import os
import chess.polyglot


def evaluate_board(board):
    """
    Evaluate the current board state.
    Positive values favor white, negative values favor black.
    Returns a score based on material balance, piece activity, king safety, pawn structure, etc.
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
    material_score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            # If the piece is white, add its value; if black, subtract its value
            value = piece_values[piece.piece_type]
            material_score += value if piece.color == chess.WHITE else -value

    # Detect endgame for different piece valuations
    is_endgame = is_endgame_position(board)
    
    # Piece-Square Tables - different for middlegame and endgame
    pst_score = calculate_piece_square_tables(board, is_endgame)

    # Center control evaluation - reward control of center squares
    center_control_score = evaluate_center_control(board)

    # Mobility evaluation - reward the number of legal moves
    mobility_score = evaluate_mobility(board)
    
    # Pawn structure evaluation
    pawn_structure_score = evaluate_pawn_structure(board)
    
    # King safety evaluation
    king_safety_score = evaluate_king_safety(board, is_endgame)
    
    # Piece coordination and development
    piece_coordination_score = evaluate_piece_coordination(board, is_endgame)
    
    # Rooks on open files and on the 7th/8th rank
    rook_placement_score = evaluate_rook_placement(board)
    
    # Bishop pair bonus
    bishop_pair_score = evaluate_bishop_pair(board)
    
    # Knight outpost positions
    knight_outpost_score = evaluate_knight_outposts(board)

    # Compile total score from all evaluation components
    evaluation = (
        material_score 
        + pst_score 
        + center_control_score 
        + mobility_score 
        + pawn_structure_score 
        + king_safety_score
        + piece_coordination_score
        + rook_placement_score
        + bishop_pair_score
        + knight_outpost_score
    )

    return evaluation


def evaluate_center_control(board):
    """Evaluate control of the center squares"""
    center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
    score = 0
    
    for square in center_squares:
        # Examine the number of pieces attacking the center squares
        white_attackers = board.attackers(chess.WHITE, square)
        black_attackers = board.attackers(chess.BLACK, square)
        
        score += len(white_attackers) * 10
        score -= len(black_attackers) * 10
        
        # Additional bonus for occupying center with pawns
        piece = board.piece_at(square)
        if piece and piece.piece_type == chess.PAWN:
            score += 15 if piece.color == chess.WHITE else -15
    
    return score


def evaluate_mobility(board):
    """Evaluate mobility (number of legal moves available)"""
    original_turn = board.turn

    board.turn = chess.WHITE
    white_moves = len(list(board.legal_moves))

    board.turn = chess.BLACK
    black_moves = len(list(board.legal_moves))

    board.turn = original_turn

    return (white_moves - black_moves) * 5


def evaluate_pawn_structure(board):
    """Evaluate pawn structure: doubled, isolated, passed, backward pawns"""
    score = 0
    
    # Evaluate for both colors
    for color in [chess.WHITE, chess.BLACK]:
        multiplier = 1 if color == chess.WHITE else -1
        
        # Get all pawns of this color
        pawns = board.pieces(chess.PAWN, color)
        
        # Check for isolated pawns
        for square in pawns:
            file = chess.square_file(square)
            
            # Isolated pawns (no friendly pawns on adjacent files)
            isolated = True
            for adj_file in [max(0, file-1), min(7, file+1)]:
                adj_file_pawns = [s for s in pawns if chess.square_file(s) == adj_file]
                if adj_file_pawns:
                    isolated = False
                    break
            
            if isolated:
                score -= 20 * multiplier
            
            # Doubled pawns (another pawn on the same file)
            file_pawns = [s for s in pawns if chess.square_file(s) == file]
            if len(file_pawns) > 1:
                score -= 15 * multiplier * (len(file_pawns) - 1)
            
            # Passed pawns (no enemy pawns ahead on same file or adjacent files)
            passed = True
            enemy_pawns = board.pieces(chess.PAWN, not color)
            
            # Determine "ahead" based on color
            if color == chess.WHITE:
                rank = chess.square_rank(square)
                ahead_range = range(rank + 1, 8)
            else:
                rank = chess.square_rank(square)
                ahead_range = range(0, rank)
            
            for r in ahead_range:
                for f in [max(0, file-1), file, min(7, file+1)]:
                    check_square = chess.square(f, r)
                    if check_square in enemy_pawns:
                        passed = False
                        break
                if not passed:
                    break
            
            if passed:
                # Bonus increases as pawn advances
                if color == chess.WHITE:
                    bonus = 10 + 10 * rank
                else:
                    bonus = 10 + 10 * (7 - rank)
                score += bonus * multiplier
    
    return score


def evaluate_king_safety(board, is_endgame):
    """Evaluate king safety based on pawn shield, piece proximity, and open lines"""
    if is_endgame:
        # In endgame, king should be active in the center
        return evaluate_king_activity_endgame(board)
    else:
        # In middlegame, king should be safe behind pawns
        return evaluate_king_safety_middlegame(board)


def evaluate_king_activity_endgame(board):
    """In endgame, kings should be active and centralized"""
    score = 0
    
    # Center distance penalty for kings
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    
    if white_king:
        file = chess.square_file(white_king)
        rank = chess.square_rank(white_king)
        # Distance from center files (d, e) and center ranks (4, 5)
        file_distance = min(abs(file - 3), abs(file - 4))
        rank_distance = min(abs(rank - 3), abs(rank - 4))
        center_distance = file_distance + rank_distance
        score -= center_distance * 10
    
    if black_king:
        file = chess.square_file(black_king)
        rank = chess.square_rank(black_king)
        file_distance = min(abs(file - 3), abs(file - 4))
        rank_distance = min(abs(rank - 3), abs(rank - 4))
        center_distance = file_distance + rank_distance
        score += center_distance * 10
    
    return score


def evaluate_king_safety_middlegame(board):
    """Evaluate king safety in the middlegame"""
    score = 0
    
    for color in [chess.WHITE, chess.BLACK]:
        multiplier = 1 if color == chess.WHITE else -1
        king_square = board.king(color)
        
        if not king_square:
            continue
        
        # Pawn shield: check for pawns in front of the king
        pawn_shield_score = 0
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        # Check for castled position (king on the wings)
        is_castled = (king_file < 3 or king_file > 4)
        
        if is_castled:
            # Define pawn shield squares based on king position and color
            shield_squares = []
            
            if color == chess.WHITE:
                base_rank = 1 if king_rank == 0 else king_rank - 1
                for r in range(base_rank, min(base_rank + 2, 8)):
                    for f in range(max(0, king_file - 1), min(king_file + 2, 8)):
                        shield_squares.append(chess.square(f, r))
            else:
                base_rank = 6 if king_rank == 7 else king_rank + 1
                for r in range(base_rank, max(base_rank - 2, -1), -1):
                    for f in range(max(0, king_file - 1), min(king_file + 2, 8)):
                        shield_squares.append(chess.square(f, r))
            
            # Count pawns on shield squares
            for square in shield_squares:
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    pawn_shield_score += 10
            
            # Penalty for missing shield pawns (more severe if king is castled)
            expected_shield_pawns = 3
            pawn_shield_score -= (expected_shield_pawns - min(pawn_shield_score // 10, expected_shield_pawns)) * 15
        
        # Count attackers near the king
        king_danger_zone = []
        for f in range(max(0, king_file - 2), min(king_file + 3, 8)):
            for r in range(max(0, king_rank - 2), min(king_rank + 3, 8)):
                king_danger_zone.append(chess.square(f, r))
        
        attacker_count = 0
        for square in king_danger_zone:
            if board.piece_at(square) and board.piece_at(square).color != color:
                piece_type = board.piece_at(square).piece_type
                if piece_type != chess.PAWN:  # Pawns aren't as dangerous
                    attacker_count += 1
        
        king_safety_penalty = attacker_count * attacker_count * 10  # Quadratic penalty
        
        # Open files near the king are dangerous
        king_file_danger = 0
        for f in range(max(0, king_file - 1), min(king_file + 2, 8)):
            file_is_open = True
            for r in range(0, 8):
                square = chess.square(f, r)
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    file_is_open = False
                    break
            
            if file_is_open:
                king_file_danger += 25
        
        # Combine king safety factors
        king_safety = pawn_shield_score - king_safety_penalty - king_file_danger
        score += king_safety * multiplier
    
    return score


def evaluate_piece_coordination(board, is_endgame):
    """Evaluate how well pieces coordinate and support each other"""
    score = 0
    
    # Piece development bonus (knights and bishops)
    for color in [chess.WHITE, chess.BLACK]:
        multiplier = 1 if color == chess.WHITE else -1
        
        knights = board.pieces(chess.KNIGHT, color)
        bishops = board.pieces(chess.BISHOP, color)
        
        # Penalty for undeveloped knights and bishops in early game
        if not is_endgame:
            initial_knight_squares = [chess.B1, chess.G1] if color == chess.WHITE else [chess.B8, chess.G8]
            initial_bishop_squares = [chess.C1, chess.F1] if color == chess.WHITE else [chess.C8, chess.F8]
            
            for square in knights:
                if square not in initial_knight_squares:
                    score += 10 * multiplier
            
            for square in bishops:
                if square not in initial_bishop_squares:
                    score += 10 * multiplier
        
        # Coordination: pieces defending each other
        for piece_square in list(knights) + list(bishops) + list(board.pieces(chess.ROOK, color)) + list(board.pieces(chess.QUEEN, color)):
            defenders = board.attackers(color, piece_square)
            if defenders:
                score += 5 * len(defenders) * multiplier
    
    return score


def evaluate_rook_placement(board):
    """Evaluate rook placement: open files, 7th rank"""
    score = 0
    
    for color in [chess.WHITE, chess.BLACK]:
        multiplier = 1 if color == chess.WHITE else -1
        rooks = board.pieces(chess.ROOK, color)
        
        for rook_square in rooks:
            rook_file = chess.square_file(rook_square)
            rook_rank = chess.square_rank(rook_square)
            
            # Rook on open file (no pawns on the file)
            file_is_open = True
            semi_open = True
            
            for rank in range(0, 8):
                square = chess.square(rook_file, rank)
                piece = board.piece_at(square)
                
                if piece and piece.piece_type == chess.PAWN:
                    if piece.color == color:
                        semi_open = True
                    file_is_open = False
            
            if file_is_open:
                score += 25 * multiplier
            elif semi_open:
                score += 10 * multiplier
            
            # Rook on 7th rank (or 2nd rank for black)
            if (color == chess.WHITE and rook_rank == 6) or (color == chess.BLACK and rook_rank == 1):
                # Check if enemy king is on the back rank
                enemy_king_rank = chess.square_rank(board.king(not color)) if board.king(not color) else -1
                if (color == chess.WHITE and enemy_king_rank == 7) or (color == chess.BLACK and enemy_king_rank == 0):
                    score += 35 * multiplier  # Bigger bonus when trapping the king
                else:
                    score += 20 * multiplier
            
            # Connected rooks (on same rank or file)
            for other_rook in rooks:
                if other_rook != rook_square:
                    other_file = chess.square_file(other_rook)
                    other_rank = chess.square_rank(other_rook)
                    
                    if rook_file == other_file or rook_rank == other_rank:
                        score += 15 * multiplier
    
    return score


def evaluate_bishop_pair(board):
    """Evaluate the bishop pair advantage"""
    score = 0
    
    white_bishops = list(board.pieces(chess.BISHOP, chess.WHITE))
    black_bishops = list(board.pieces(chess.BISHOP, chess.BLACK))
    
    # Check if either side has the bishop pair (two bishops on different colored squares)
    if len(white_bishops) >= 2:
        white_colors = set(chess.square_file(sq) % 2 == chess.square_rank(sq) % 2 for sq in white_bishops)
        if len(white_colors) > 1:  # Bishops on squares of different colors
            score += 50
    
    if len(black_bishops) >= 2:
        black_colors = set(chess.square_file(sq) % 2 == chess.square_rank(sq) % 2 for sq in black_bishops)
        if len(black_colors) > 1:  # Bishops on squares of different colors
            score -= 50
    
    return score


def evaluate_knight_outposts(board):
    """Evaluate knights in strong outpost positions"""
    score = 0
    
    for color in [chess.WHITE, chess.BLACK]:
        multiplier = 1 if color == chess.WHITE else -1
        knights = board.pieces(chess.KNIGHT, color)
        
        for knight_square in knights:
            # Potential outpost squares are in enemy territory (ranks 4-6 for white, 3-5 for black)
            # and defended by a friendly pawn
            if color == chess.WHITE:
                rank = chess.square_rank(knight_square)
                if 3 <= rank <= 5:  # Ranks 4-6
                    potential_outpost = True
                else:
                    potential_outpost = False
            else:
                rank = chess.square_rank(knight_square)
                if 2 <= rank <= 4:  # Ranks 3-5
                    potential_outpost = True
                else:
                    potential_outpost = False
            
            if potential_outpost:
                # Check if defended by pawn
                pawn_defenders = [s for s in board.attackers(color, knight_square) if board.piece_at(s).piece_type == chess.PAWN]
                
                # Check if it can be attacked by enemy pawns
                file = chess.square_file(knight_square)
                vulnerable_to_pawns = False
                
                # Check adjacent files for enemy pawns that could attack
                for adj_file in [max(0, file-1), min(7, file+1)]:
                    for r in range(0, 8):
                        square = chess.square(adj_file, r)
                        piece = board.piece_at(square)
                        if piece and piece.piece_type == chess.PAWN and piece.color != color:
                            # Check if this pawn can advance to attack the knight
                            if color == chess.WHITE and chess.square_rank(square) > rank:
                                vulnerable_to_pawns = True
                                break
                            elif color == chess.BLACK and chess.square_rank(square) < rank:
                                vulnerable_to_pawns = True
                                break
                
                if pawn_defenders and not vulnerable_to_pawns:
                    # Strong outpost
                    score += 25 * multiplier
                elif not vulnerable_to_pawns:
                    # Decent outpost but not pawn-defended
                    score += 10 * multiplier
    
    return score


def calculate_piece_square_tables(board, is_endgame):
    """Calculate bonuses/penalties based on piece-square tables"""
    score = 0
    
    # Piece-Square Tables
    pst_pawn_mg = [
        0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5,  5, 10, 25, 25, 10,  5,  5,
        0,  0,  0, 20, 20,  0,  0,  0,
        5, -5,-10,  0,  0,-10, -5,  5,
        5, 10, 10,-20,-20, 10, 10,  5,
        0,  0,  0,  0,  0,  0,  0,  0
    ]
    
    pst_knight_mg = [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
    ]
    
    pst_bishop_mg = [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20
    ]
    
    pst_rook_mg = [
        0,  0,  0,  0,  0,  0,  0,  0,
        5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        0,  0,  0,  5,  5,  0,  0,  0
    ]
    
    pst_queen_mg = [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
        -5,  0,  5,  5,  5,  5,  0, -5,
        0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
    ]
    
    pst_king_mg = [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
        20, 20,  0,  0,  0,  0, 20, 20,
        20, 30, 10,  0,  0, 10, 30, 20
    ]
    
    pst_king_eg = [
        -50,-40,-30,-20,-20,-30,-40,-50,
        -30,-20,-10,  0,  0,-10,-20,-30,
        -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30,
        -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-30,  0,  0,  0,  0,-30,-30,
        -50,-30,-30,-30,-30,-30,-30,-50
    ]
    
    pst_pawn_eg = [
        0,  0,  0,  0,  0,  0,  0,  0,
        80, 80, 80, 80, 80, 80, 80, 80,
        50, 50, 50, 50, 50, 50, 50, 50,
        30, 30, 30, 30, 30, 30, 30, 30,
        20, 20, 20, 20, 20, 20, 20, 20,
        10, 10, 10, 10, 10, 10, 10, 10,
        10, 10, 10, 10, 10, 10, 10, 10,
        0,  0,  0,  0,  0,  0,  0,  0
    ]
    
    # Select endgame or middlegame tables
    pst_pawn = pst_pawn_eg if is_endgame else pst_pawn_mg
    pst_king = pst_king_eg if is_endgame else pst_king_mg
    pst_knight = pst_knight_mg  # Same for both phases
    pst_bishop = pst_bishop_mg  # Same for both phases
    pst_rook = pst_rook_mg      # Same for both phases
    pst_queen = pst_queen_mg    # Same for both phases
    
    # Mapping of piece types to PST arrays
    pst_tables = {
        chess.PAWN: pst_pawn,
        chess.KNIGHT: pst_knight,
        chess.BISHOP: pst_bishop,
        chess.ROOK: pst_rook,
        chess.QUEEN: pst_queen,
        chess.KING: pst_king
    }
    
    # Apply PST bonuses for each piece
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Get the piece-square table for this piece type
            pst = pst_tables[piece.piece_type]
            
            # Apply the bonus based on piece color and position
            if piece.color == chess.WHITE:
                # For white pieces, we index the PST directly
                score += pst[63 - square]
            else:
                # For black pieces, we flip the square and negate the bonus
                score -= pst[square]
    
    return score


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
    use_null_move=True,
    null_move_reduction=3,
    allow_null=True,  # Whether null moves are allowed in this search branch
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
    :param use_null_move: Whether to use null move pruning
    :param null_move_reduction: Depth reduction for null move pruning (R value)
    :param allow_null: Whether to allow null moves in this branch (prevents consecutive null moves)
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

    # Try Null Move Pruning
    # Only try if:
    # 1. Null move pruning is enabled
    # 2. We're not in check
    # 3. We haven't just made a null move
    # 4. We have sufficient depth to make pruning worthwhile
    # 5. We're not in a potential zugzwang position (few pieces left)
    if (
        use_null_move
        and allow_null
        and depth >= 3
        and not board.is_check()
        and not is_endgame_position(board)
        and not likely_zugzwang(board)
    ):
        # Make a null move (switch sides without making a move)
        board.push(chess.Move.null())

        # Search with reduced depth - use R=2 or R=3 typically
        # We use a zero window search since we only care if score exceeds beta
        null_score = -minimax(
            board,
            depth - 1 - null_move_reduction,  # Reduce depth by R
            -beta,
            -beta + 1,  # Zero window search
            not maximizing_player,
            tt,
            dynamic_depth,
            max_depth_extension,
            uci_mode,
            engine,
            use_quiescence,
            max_q_depth,
            use_null_move,
            null_move_reduction,
            False,  # Don't allow consecutive null moves
        )

        # Undo the null move
        board.pop()

        # If the position is still good even after giving opponent a free move,
        # we can assume it's very good and prune this branch
        if null_score >= beta:
            return beta  # Beta cutoff

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
                use_null_move,
                null_move_reduction,
                True,  # Allow null moves again in child nodes
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
                use_null_move,
                null_move_reduction,
                True,  # Allow null moves again in child nodes
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
    use_null_move=True,
    null_move_reduction=3,
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
    :param use_null_move: Whether to use null move pruning
    :param null_move_reduction: Depth reduction for null move pruning
    :return: Best move and its evaluation score
    """
    if tt is None:
        tt = TranspositionTable()

    legal_moves = list(board.legal_moves)

    if not legal_moves:
        # No legal moves, this should not happen in normal play
        # but handle it gracefully
        return None, evaluate_board(board)
        
    if len(legal_moves) == 1:
        return legal_moves[0], evaluate_board(board)

    maximizing_player = board.turn == chess.WHITE
    best_move = legal_moves[0]  # Default to first legal move to ensure we always have a move
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
            use_null_move,
            null_move_reduction,
            True,  # Allow null moves in child nodes
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
    use_null_move=True,
    null_move_reduction=3,
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
    :param use_null_move: Whether to use null move pruning
    :param null_move_reduction: Depth reduction for null move pruning
    :return: Best move, its evaluation score, and actual depth reached
    """
    if tt is None:
        tt = TranspositionTable()

    start_time = time.time()
    reached_depth = 0

    # For single legal move, return immediately
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        # No legal moves available (this should not happen in normal play)
        return None, evaluate_board(board), 0
        
    if len(legal_moves) == 1:
        return legal_moves[0], evaluate_board(board), 0
        
    # Default to the first legal move as a fallback to ensure we always return a valid move
    best_move = legal_moves[0]
    best_value = evaluate_board(board)

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
            use_null_move=use_null_move,
            null_move_reduction=null_move_reduction,
        )

        # Only update our best move if find_best_move returned a valid move
        if move is not None:
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
    if tt_entry and tt_entry["depth"] >= max_q_depth:
        if tt_entry["flag"] == EXACT:
            return tt_entry["score"]
        elif tt_entry["flag"] == LOWERBOUND and tt_entry["score"] > alpha:
            alpha = tt_entry["score"]
        elif tt_entry["flag"] == UPPERBOUND and tt_entry["score"] < beta:
            beta = tt_entry["score"]

        if alpha >= beta:
            return tt_entry["score"]

    # Stand-pat evaluation (evaluate without any moves)
    stand_pat = evaluate_board(board)

    # Fail-hard beta cutoff
    if maximizing_player and stand_pat >= beta:
        tt.store(board, max_q_depth, beta, LOWERBOUND)
        return beta
    if not maximizing_player and stand_pat <= alpha:
        tt.store(board, max_q_depth, alpha, UPPERBOUND)
        return alpha

    # Update alpha if stand-pat is better
    if maximizing_player and stand_pat > alpha:
        alpha = stand_pat
    if not maximizing_player and stand_pat < beta:
        beta = stand_pat

    # If max_q_depth reached, return stand-pat evaluation
    if max_q_depth <= 0:
        tt.store(board, 0, stand_pat, EXACT)
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
        tt.store(board, max_q_depth, stand_pat, EXACT)
        return stand_pat

    # Order moves by MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
    captures.sort(key=lambda move: mvv_lva_score(board, move), reverse=True)

    if maximizing_player:
        best_score = stand_pat
        for move in captures:
            board.push(move)
            score = quiescence_search(
                board, alpha, beta, False, tt, engine, max_q_depth - 1
            )
            board.pop()

            if score > best_score:
                best_score = score

            if score >= beta:
                tt.store(board, max_q_depth, beta, LOWERBOUND)
                return beta  # Fail-hard beta cutoff
            if score > alpha:
                alpha = score

        # Store the result in the transposition table
        flag = EXACT if alpha > stand_pat else UPPERBOUND
        tt.store(board, max_q_depth, alpha, flag)
        return alpha
    else:
        best_score = stand_pat
        for move in captures:
            board.push(move)
            score = quiescence_search(
                board, alpha, beta, True, tt, engine, max_q_depth - 1
            )
            board.pop()

            if score < best_score:
                best_score = score

            if score <= alpha:
                tt.store(board, max_q_depth, alpha, UPPERBOUND)
                return alpha  # Fail-hard alpha cutoff
            if score < beta:
                beta = score

        # Store the result in the transposition table
        flag = EXACT if beta < stand_pat else LOWERBOUND
        tt.store(board, max_q_depth, beta, flag)
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


def is_endgame_position(board):
    """
    Determine if position is an endgame position.
    We consider a position to be an endgame if:
    1. No queens on the board, or
    2. Both sides have <= 1 piece besides king and pawns

    :param board: Current board state
    :return: True if it's an endgame position, False otherwise
    """
    # Count pieces
    white_queens = len(board.pieces(chess.QUEEN, chess.WHITE))
    black_queens = len(board.pieces(chess.QUEEN, chess.BLACK))
    white_pieces = (
        len(board.pieces(chess.KNIGHT, chess.WHITE))
        + len(board.pieces(chess.BISHOP, chess.WHITE))
        + len(board.pieces(chess.ROOK, chess.WHITE))
    )
    black_pieces = (
        len(board.pieces(chess.KNIGHT, chess.BLACK))
        + len(board.pieces(chess.BISHOP, chess.BLACK))
        + len(board.pieces(chess.ROOK, chess.BLACK))
    )

    # Endgame conditions
    no_queens = white_queens + black_queens == 0
    few_pieces = white_pieces <= 1 and black_pieces <= 1

    return no_queens or few_pieces


def likely_zugzwang(board):
    """
    Check if a position is likely to be a zugzwang position.
    Zugzwang is a situation where any move will worsen the position.
    This is common in endgames, especially king and pawn endgames.

    :param board: Current board state
    :return: True if position is likely a zugzwang, False otherwise
    """
    # King and pawn endgames are classic zugzwang positions
    if is_endgame_position(board):
        white_pieces = (
            sum(1 for _ in board.pieces(chess.KNIGHT, chess.WHITE))
            + sum(1 for _ in board.pieces(chess.BISHOP, chess.WHITE))
            + sum(1 for _ in board.pieces(chess.ROOK, chess.WHITE))
            + sum(1 for _ in board.pieces(chess.QUEEN, chess.WHITE))
        )

        black_pieces = (
            sum(1 for _ in board.pieces(chess.KNIGHT, chess.BLACK))
            + sum(1 for _ in board.pieces(chess.BISHOP, chess.BLACK))
            + sum(1 for _ in board.pieces(chess.ROOK, chess.BLACK))
            + sum(1 for _ in board.pieces(chess.QUEEN, chess.BLACK))
        )

        # Pure king+pawn endgames
        if white_pieces == 0 and black_pieces == 0:
            return True

        # Simple endgames with very few pieces
        if white_pieces + black_pieces <= 2:
            return True

    return False
