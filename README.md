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

# Configure your Lichess API token in config.yml
```

## Usage

### Local Engine Testing

```bash
# Start the engine in local mode
poetry run python -m chessbot.engine

# Play against the engine
poetry run python -m chessbot.play
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
  use_opening_book: true
  evaluation:
    material_weight: 1.0
    position_weight: 0.2
    king_safety_weight: 0.5
    mobility_weight: 0.3

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

- [ ] **Lichess Integration**: Play games automatically on Lichess through the Lichess Bot API
  - [ ] API authentication
  - [ ] Game acceptance
  - [ ] Automatic move submission
  - [ ] Rating-based matching

- [ ] **Opening Book**: Basic opening theory implementation for stronger early game
  - [ ] Book database integration
  - [ ] Weighted move selection
  - [ ] Book learning capabilities

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

Completed: 2/5 main features
