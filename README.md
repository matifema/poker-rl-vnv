# Poker RL — Evolution Strategies + Self-Play

Neural network agent for Texas Hold'em trained via Evolution Strategies with self-play, following Prof. Tronci's Verification & Validation protocol.

## Architecture

- **Policy**: 2-layer feedforward NN (26 → 64 → 7) — ~2,183 parameters
- **Features**: 26-dim state vector (hole/community cards, hand strength, pot odds, position, etc.)
- **Training**: ES with antithetic sampling, z-score fitness, OpenAI ES formula (Algo 1)
- **Action masking**: Invalid actions set to −∞ before argmax (core V&V invariant)

## Self-Play Protocol

**Default: simultaneous co-evolution.** Two agents train against each other in lockstep — each generation, A updates against frozen B, then B updates against the updated A. This creates a continuous arms race that prevents gradient collapse near equilibrium.

**Alternative: sequential (`--sequential`).** 3-sprint curriculum with warm-start:

| Sprint | Training | Opponent | Init | Purpose |
|--------|----------|----------|------|---------|
| A | Agent A | Random | Random | Learn basic poker |
| B | Agent B | Frozen A | From A's weights | Learn to counter A |
| A' | Agent A' | Frozen B | From B's weights | Converge toward Nash |

## Results (simultaneous co-evolution, sigma=0.04, alpha=0.001, 30 gens, 500 hands, pop 100)

| Opponent | Win Rate | Profit/Hand |
|----------|----------|-------------|
| vs Random | 57.4% | +32.0 |
| vs Agent A | 47.6% | -6.1 |
| vs Agent B | 53.0% | -1.7 |
| vs TightAgent | 72.8% | +33.2 |

**Nash equilibrium**: A' vs original A at 47.6%, A' vs B at 53.0% — both within 3-4% of 50%. The simultaneous co-evolution learns to exploit B while staying near-Nash against A, avoiding the flat-gradient plateau that plagued the sequential warm-start approach.

## Quick Start

```bash
# Train (defaults: co-evolution, sigma=0.04, alpha=0.001, 25 gens, 200 hands, pop 40)
python -m es_poker.main train

# Full training run
python -m es_poker.main train --gens 30 --hands 500 --pop 100

# Train with parallelism
python -m es_poker.main train --gens 30 --hands 500 --pop 100 --parallel

# Sequential mode (3-sprint warm-start)
python -m es_poker.main train --sequential

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
| `--alpha` | 0.001 | Learning rate |
| `--seed` | 42 | Random seed |
| `--parallel` | — | Enable multiprocessing |
| `--sequential` | — | Use sequential warm-start instead of co-evolution |
| `-o` | `./training_output` | Output directory |

## Dependencies

- Python 3.10+
- NumPy
- pokerl (included)
