# Poker RL — Evolution Strategies + Self-Play

Neural network agent for Texas Hold'em trained via Evolution Strategies with self-play, following Prof. Tronci's Verification & Validation protocol.

## Architecture

- **Policy**: 2-layer feedforward NN (26 → 64 → 7) — ~2,183 parameters
- **Features**: 26-dim state vector (hole/community cards, hand strength, pot odds, position, etc.)
- **Training**: ES with antithetic sampling, rank-based fitness, and mirrored evaluation
- **Action masking**: Invalid actions set to −∞ before argmax (core V&V invariant)

## Self-Play Protocol

3-sprint curriculum:

| Sprint | Training | Opponent | Purpose |
|--------|----------|----------|---------|
| A | Agent A (random init) | Random | Learn basic poker |
| B | Agent B (random init) | Agent A | Learn to counter A |
| A' | Agent A' (init from best A) | Agent B | Learn to counter B |

## Results (sigma=0.04, 30 gens, 500 hands, pop 80)

| Opponent | Win Rate | Profit/Hand |
|----------|----------|-------------|
| vs Random | 71.8% | +78.5 |
| vs Agent A | 49.6% | +11.9 |
| vs Agent B | 58.0% | +76.1 |
| vs TightAgent | 94.6% | +55.8 |

**Nash equilibrium**: A' vs A at 49.6% — neither agent dominates the other, demonstrating convergence.

## Quick Start

```bash
# Train
python -m es_poker.main train --gens 30 --hands 500 --pop 80

# Train with parallelism
python -m es_poker.main train --gens 30 --hands 500 --pop 80 --parallel 8

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
| `--parallel` | 1 | Worker processes for evaluation |
| `-o` | `./training_output` | Output directory |

## Dependencies

- Python 3.10+
- NumPy
- pokerl (included)
