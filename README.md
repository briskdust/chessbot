# ChessBot

A powerful Python-based chess engine that achieves approximately 2000 ELO strength through minimax algorithm with alpha-beta pruning, and integrates with Lichess for online play.

## Features

- **Strong Engine**: ~2000 ELO rating through optimized minimax search with alpha-beta pruning
- **Opening Book**: Basic opening theory implementation for stronger early game
- **Position Evaluation**: Sophisticated evaluation function considering:
  - Material balance
  - Piece positioning
  - King safety
  - Pawn structure
  - Mobility
- **Dynamic Search Depths**: Configurable search depth to balance strength and move time
- **Lichess Integration**: Play games automatically on Lichess through the Lichess Bot API

## Requirements

- Python 3.8+
- Poetry (for dependency management)

## Installation

```bash
# Clone the repository
git clone https://github.com/briskdust/chessbot.git
cd chessbot

# Install dependencies with Poetry
poetry install

# Configure your Lichess API token in config.yml and set up the opening book
```

### Setting Up the Opening Book

1. Download a Polyglot opening book file (e.g., Cerebellum)
2. Place the .bin file in the project's root directory
3. Update the `book_path` in your config.yml to point to your opening book file

## Usage

### Local Engine Testing

```bash
# Start a game with the engine using settings from config.yml
poetry run python main.py

# Or specify a custom config file
poetry run python main.py custom_config.yml
```

### UCI Mode (Universal Chess Interface)

ChessBot implements the UCI protocol, allowing it to be used with chess GUIs and platforms like lichess-bot.

```bash
# Run ChessBot in UCI mode
poetry run chessbot-uci

# Or directly using the script
python chessbot_uci.py
```

For use with lichess-bot, configure your `config.yml` in the lichess-bot repository:

```yaml
engine:
  dir: "/path/to/chessbot/"
  name: "python chessbot_uci.py"  # Or "poetry run chessbot-uci" if using poetry
  working_dir: "/path/to/chessbot/"
```

### Lichess Bot Mode

```bash
# Start the bot on Lichess
poetry run python -m chessbot.lichess_bot

# Or with a specific configuration file
poetry run python -m chessbot.lichess_bot --config custom_config.yml
```

## Configuration

Edit the `config.yml` file to customize:

```yaml
engine:
  search_depth: 5  # Default search depth
  dynamic_depth: true  # Enable dynamic depth adjustment
  max_depth_extension: 2  # Maximum depth extension
  tt_size_mb: 128  # Transposition table size in MB
  time_limit: 5  # Default time limit in seconds

opening_book:
  use_book: true  # Whether to use the opening book
  book_path: "cerebellum.bin"  # Path to the opening book file

lichess:
  token: "YOUR_LICHESS_API_TOKEN"
  challenge:
    accept_rated: true
    accept_casual: true
    min_rating: 1600
    max_rating: 2200
  concurrency: 1  # Number of games to play simultaneously
```

## Architecture

The project is structured into several main components:

- **Engine**: Core chess logic and AI implementation
  - Search algorithm (minimax with alpha-beta pruning)
  - Position evaluation
  - Move ordering
- **Lichess API**: Integration with Lichess platform using berserk
- **UI**: Simple command-line interface for local play

## Implementation Details

ChessBot uses several key techniques to achieve approximately 2000 ELO strength:

1. **Alpha-Beta Pruning**: Optimized minimax with alpha-beta pruning
2. **Move Ordering**: Orders moves to improve alpha-beta efficiency
3. **Transposition Tables**: Caches previously evaluated positions
4. **Iterative Deepening**: Progressively increases search depth with time constraints
5. **Dynamic Depth Adjustment**: Extends search depth for captures and checks
6. **Time Management**: Allocates thinking time based on position complexity
7. **Quiescence Search**: Ensures stable position evaluation after captures (planned)
8. **Late Move Reduction**: Reduces search depth for less promising moves (planned)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [python-chess](https://python-chess.readthedocs.io/) for the chess logic library
- [berserk](https://github.com/rhgrant10/berserk) for Lichess API integration
- [Lichess](https://lichess.org/) for their excellent chess platform and API

## Future Improvements

- Neural network evaluation function
- Endgame tablebase integration
- Monte Carlo Tree Search implementation
- Time management improvements
- Support for chess variants

---

## TODO:

- [x] **Basic Engine**: A basic Minimax engine with Alpha-beta pruning and transposition tables
  - [x] Basic minimax implementation
  - [x] Alpha-beta pruning
  - [x] Transposition tables

- [x] **Lichess Integration**: Play games automatically on Lichess through the Lichess Bot API
  - [x] UCI protocol implementation
  - [ ] Game acceptance
  - [ ] Automatic move submission
  - [ ] Rating-based matching

- [x] **Opening Book**: Basic opening theory implementation for stronger early game
  - [x] Book database integration
  - [x] Weighted move selection

- [x] **Dynamic Search Depths**: Configurable search depth to balance strength and move time
  - [x] Dynamic depth adjustment
  - [x] Time management system
  - [x] Iterative deepening

- [ ] **Position Evaluation**: Sophisticated evaluation function considering:
  - [x] Material balance
  - [ ] Piece positioning
  - [ ] King safety
  - [ ] Pawn structure
  - [ ] Mobility

### Current Progress

Completed: 3/5 main features
