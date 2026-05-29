# Poker RL — Evolution Strategies + Self-Play

Neural network agent for Texas Hold'em trained via Evolution Strategies with self-play, following Prof. Tronci's Verification & Validation protocol.

## Architecture

- **Policy**: 2-layer feedforward NN (26 → 64 → 7) — ~2,183 parameters
- **Features**: 26-dim state vector (hole/community cards, hand strength, pot odds, position, etc.)
- **Training**: ES with antithetic sampling, rank-based fitness, and mirrored evaluation
- **Action masking**: Invalid actions set to −∞ before argmax (core V&V invariant)

## Self-Play Protocol

3-sprint curriculum with warm-start initialization:

| Sprint | Training | Opponent | Init | Purpose |
|--------|----------|----------|------|---------|
| A | Agent A | Random | Random | Learn basic poker |
| B | Agent B | Frozen A | From A's weights | Learn to counter A |
| A' | Agent A' | Frozen B | From B's weights | Converge toward Nash |

Each sprint warm-starts from the previous agent's weights — B inherits A's poker fundamentals and only needs to adapt to exploit A, rather than learning from scratch against a trained opponent.

## Results (sigma=0.04, 50 gens, 500 hands, pop 100, warm-start)

| Opponent | Win Rate | Profit/Hand |
|----------|----------|-------------|
| vs Random | 69.8% | +44.5 |
| vs Agent A | 50.6% | +48.0 |
| vs Agent B | 47.4% | -10.0 |
| vs TightAgent | 85.6% | +62.2 |

**Nash equilibrium**: A' vs A at 50.6%, A' vs B at 47.4% — all agents within 3% of 50%, demonstrating convergence. Warm-start eliminates the early-training collapse that random-init B and A' previously suffered.

## Quick Start

```bash
# Train (defaults: sigma=0.04, alpha=0.02, 25 gens, 200 hands, pop 40)
python -m es_poker.main train

# Full training run
python -m es_poker.main train --gens 50 --hands 500 --pop 100

# Train with parallelism
python -m es_poker.main train --gens 50 --hands 500 --pop 100 --parallel

# Evaluate a trained agent
python -m es_poker.main evaluate training_output/agent_A2.npz

# Run verification tests (action masking, invariants)
python -m es_poker.main verify

# Run unit tests
python -m pytest es_poker/tests/ -v
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--gens` | 25 | Generations per sprint |
| `--hands` | 200 | Hands per individual evaluation |
| `--pop` | 40 | Population size (must be even) |
| `--sigma` | 0.04 | ES noise standard deviation |
| `--alpha` | 0.02 | Learning rate |
| `--seed` | 42 | Random seed |
| `--parallel` | — | Enable multiprocessing |
| `-o` | `./training_output` | Output directory |

## Dependencies

- Python 3.10+
- NumPy
- pokerl (included)
