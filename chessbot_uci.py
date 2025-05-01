#!/usr/bin/env python
"""
ChessBot UCI Engine Entry Point

This script starts ChessBot in UCI mode, allowing it to communicate with
GUI chess interfaces and platforms like lichess-bot using the Universal Chess Interface protocol.
"""

from src.uci import main

if __name__ == "__main__":
    # Start the UCI engine
    main()
