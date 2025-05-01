import chess
import time
import sys
from .minimax_engine import MinimaxEngine
from .utils import OpeningBook
import os


class UCIEngine:
    """
    Universal Chess Interface implementation for the ChessBot engine.
    This class handles UCI protocol commands and communicates with the underlying engine.
    """

    def __init__(self):
        """Initialize the UCI interface"""
        self.engine = None
        self.name = "ChessBot"
        self.author = "briskdust"
        self.board = chess.Board()

        # Default options
        self.options = {
            "Depth": 5,
            "TranspositionTableSize": 128,
            "UseOpeningBook": True,
            "OpeningBookPath": "Cerebellum3Merge.bin",
            "DynamicDepth": True,
            "MaxDepthExtension": 2,
            "TimeLimit": 5000,  # In milliseconds
            "UseQuiescence": True,  # Whether to use quiescence search
            "MaxQDepth": 5,  # Maximum quiescence search depth
            "UseNullMove": True,  # Whether to use null move pruning
            "NullMoveReduction": 3,  # Depth reduction (R) for null move pruning
        }

    def initialize_engine(self):
        """Initialize or reinitialize the engine with current options"""
        time_limit = self.options["TimeLimit"] / 1000  # Convert to seconds

        # Check if opening book exists
        book_path = (
            self.options["OpeningBookPath"] if self.options["UseOpeningBook"] else None
        )
        if book_path and not os.path.exists(book_path):
            print(f"info string Opening book not found at {book_path}")
            book_path = None

        self.engine = MinimaxEngine(
            depth=self.options["Depth"],
            tt_size_mb=self.options["TranspositionTableSize"],
            time_limit=time_limit,
            dynamic_depth=self.options["DynamicDepth"],
            max_depth_extension=self.options["MaxDepthExtension"],
            book_path=book_path,
            use_book=self.options["UseOpeningBook"],
            uci_mode=True,
            use_quiescence=self.options["UseQuiescence"],
            max_q_depth=self.options["MaxQDepth"],
            use_null_move=self.options["UseNullMove"],
            null_move_reduction=self.options["NullMoveReduction"],
        )

    def uci(self):
        """Identify engine as supporting UCI protocol"""
        print(f"id name {self.name}")
        print(f"id author {self.author}")

        # Send available options
        print("option name Depth type spin default 5 min 1 max 10")
        print(
            "option name TranspositionTableSize type spin default 128 min 16 max 1024"
        )
        print("option name UseOpeningBook type check default true")
        print("option name OpeningBookPath type string default Cerebellum3Merge.bin")
        print("option name DynamicDepth type check default true")
        print("option name MaxDepthExtension type spin default 2 min 0 max 5")
        print("option name TimeLimit type spin default 5000 min 100 max 60000")
        print("option name UseQuiescence type check default true")
        print("option name MaxQDepth type spin default 5 min 0 max 10")
        print("option name UseNullMove type check default true")
        print("option name NullMoveReduction type spin default 3 min 2 max 4")

        print("uciok")

    def set_option(self, name, value):
        """Set an engine option"""
        if name == "Depth":
            self.options["Depth"] = int(value)
        elif name == "TranspositionTableSize":
            self.options["TranspositionTableSize"] = int(value)
        elif name == "UseOpeningBook":
            self.options["UseOpeningBook"] = value.lower() == "true"
        elif name == "OpeningBookPath":
            self.options["OpeningBookPath"] = value
        elif name == "DynamicDepth":
            self.options["DynamicDepth"] = value.lower() == "true"
        elif name == "MaxDepthExtension":
            self.options["MaxDepthExtension"] = int(value)
        elif name == "TimeLimit":
            self.options["TimeLimit"] = int(value)
        elif name == "UseQuiescence":
            self.options["UseQuiescence"] = value.lower() == "true"
        elif name == "MaxQDepth":
            self.options["MaxQDepth"] = int(value)
        elif name == "UseNullMove":
            self.options["UseNullMove"] = value.lower() == "true"
        elif name == "NullMoveReduction":
            self.options["NullMoveReduction"] = int(value)
        else:
            print(f"info string Unknown option: {name}")

    def is_ready(self):
        """Check if engine is ready to receive commands"""
        if self.engine is None:
            self.initialize_engine()
        print("readyok")

    def new_game(self):
        """Reset engine for a new game"""
        self.board = chess.Board()
        if self.engine:
            # Reinitialize engine to clear transposition tables
            self.initialize_engine()

    def set_position(self, command):
        """Set up the board position"""
        parts = command.split(" ", 1)
        if len(parts) >= 2:
            position_type = parts[0]

            # Parse FEN or use starting position
            if position_type == "fen":
                fen_parts = parts[1].split(" moves ")
                fen = fen_parts[0]
                self.board = chess.Board(fen)
                moves_part = fen_parts[1] if len(fen_parts) > 1 else None
            elif position_type == "startpos":
                self.board = chess.Board()
                moves_part = parts[1][6:] if "moves" in parts[1] else None
            else:
                print(f"info string Unknown position type: {position_type}")
                return

            # Apply moves if present
            if moves_part:
                moves = moves_part.strip().split()
                for move in moves:
                    try:
                        chess_move = chess.Move.from_uci(move)
                        if chess_move in self.board.legal_moves:
                            self.board.push(chess_move)
                        else:
                            print(f"info string Illegal move: {move}")
                            break
                    except ValueError:
                        print(f"info string Invalid move format: {move}")
                        break

    def go(self, command):
        """
        Start calculating on the current position.
        Supports parameters like depth, movetime, etc.
        """
        if self.engine is None:
            self.initialize_engine()

        # Parse go command parameters
        params = command.split()
        depth = self.options["Depth"]
        move_time = self.options["TimeLimit"] / 1000

        i = 0
        while i < len(params):
            if params[i] == "depth" and i + 1 < len(params):
                depth = int(params[i + 1])
                i += 2
            elif params[i] == "movetime" and i + 1 < len(params):
                move_time = int(params[i + 1]) / 1000  # Convert to seconds
                i += 2
            else:
                i += 1

        # Set engine parameters for this search
        self.engine.max_depth = depth
        time_limit_for_search = move_time

        # Start the search
        start_time = time.time()
        try:
            best_move = self.engine.get_move(
                self.board, time_limit=time_limit_for_search
            )
            # Return the best move in UCI format
            print(f"bestmove {best_move.uci()}")
        except Exception as e:
            print(f"info string Error in search: {e}")
            # Return a legal move in case of error
            if not self.board.is_game_over():
                legal_moves = list(self.board.legal_moves)
                if legal_moves:
                    print(f"bestmove {legal_moves[0].uci()}")
                else:
                    print("bestmove 0000")  # No move available
            else:
                print("bestmove 0000")  # No move available

    def process_command(self, command):
        """Process a UCI command"""
        if command == "uci":
            self.uci()
        elif command == "isready":
            self.is_ready()
        elif command == "ucinewgame":
            self.new_game()
        elif command.startswith("setoption name "):
            parts = command[15:].split(" value ")
            if len(parts) == 2:
                name, value = parts
                self.set_option(name, value)
            else:
                print(f"info string Invalid setoption format: {command}")
        elif command.startswith("position "):
            self.set_position(command[9:])
        elif command.startswith("go "):
            self.go(command[3:])
        elif command == "quit":
            sys.exit(0)
        elif command == "stop":
            # In our current implementation, the engine doesn't support stopping
            # But in a real implementation, we would signal the engine to stop calculation
            # and return the best move found so far
            if self.engine and not self.board.is_game_over():
                legal_moves = list(self.board.legal_moves)
                if legal_moves:
                    print(f"bestmove {legal_moves[0].uci()}")
                else:
                    print("bestmove 0000")
        elif command == "":
            # Ignore empty lines
            pass
        else:
            print(f"info string Unknown command: {command}")


def main():
    """
    Entry point for running ChessBot as a UCI engine from the command line.
    """
    engine = UCIEngine()

    while True:
        try:
            line = input()
            engine.process_command(line.strip())
        except EOFError:
            break
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"info string Error: {e}")


if __name__ == "__main__":
    main()
