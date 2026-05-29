"""Evolution Strategies algorithm for training poker agents.

Key design choices for low-variance fitness estimation:
  - Mirrored evaluation: agent plays from ALL seats, profit averaged.
  - Antithetic sampling: each noise vector ε is evaluated both as θ+ε and θ−ε.
  - Rank-based fitness: raw profits are replaced by population ranks → robust to outliers.
  - Optional multiprocessing: evaluate individuals in parallel across CPU cores.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable

import numpy as np

from .network import SmallNN

# ---------------------------------------------------------------------------
# Core evaluation: one agent vs three opponents, from a fixed seat
# ---------------------------------------------------------------------------


def _play_hands(
    players: list[Callable],
    agent_idx: int,
    game_config: dict,
    num_hands: int,
) -> tuple[float, float]:
    """Play *num_hands* hands.

    Returns (mean_profit_for_agent, win_rate).
    """
    from pokerl.game import Game

    game = Game(**game_config)
    payoff_acc = 0.0
    wins = 0
    hands_played = 0

    game.reset()

    while hands_played < num_hands:
        state = game.active_state
        action = players[game.active_player](state)
        done, hand_over, _ = game.step(action)

        if hand_over:
            profit = game.payoffs[agent_idx]
            payoff_acc += profit
            if profit > 0:
                wins += 1
            hands_played += 1

        if done:
            game.reset()

    return payoff_acc / num_hands, wins / max(num_hands, 1)


# ---------------------------------------------------------------------------
# Public evaluation: mirrored across all 4 seats
# ---------------------------------------------------------------------------


def evaluate_agent(
    agent_fn: Callable,
    opponent_fn: Callable,
    game_config: dict,
    num_hands: int = 200,
) -> tuple[float, float]:
    """Mirrored evaluation: play agent from ALL seats vs copies of opponent.

    Returns (mean_profit_per_hand, win_rate).
    """
    n_players = game_config.get("num_players", 2)
    hands_per_seat = max(1, num_hands // n_players)
    total_profit = 0.0
    total_wins = 0

    for seat in range(n_players):
        players = [opponent_fn] * n_players
        players[seat] = agent_fn
        profit, wr = _play_hands(players, seat, game_config, hands_per_seat)
        total_profit += profit
        total_wins += wr

    return total_profit / n_players, total_wins / n_players


# ---------------------------------------------------------------------------
# Evolution Strategies optimiser
# ---------------------------------------------------------------------------


def _eval_individual(args: tuple) -> tuple[float, float]:
    """Module-level worker for parallel evaluation of one individual.

    Args is a tuple: (weights, game_config, opponent_fn, hands_per_eval).
    Must be defined at module level so ProcessPoolExecutor can pickle it.
    """
    weights, game_config, opponent_fn, hands_per_eval = args

    # Workers need the pokerl package on sys.path
    import sys
    from pathlib import Path

    _pokerl = Path(__file__).resolve().parent.parent / "pokerl"
    if str(_pokerl) not in sys.path:
        sys.path.insert(0, str(_pokerl))

    from .agent import masked_argmax
    from .features import extract_features
    from .network import SmallNN

    nn = SmallNN()
    nn.set_weights(weights)

    def agent_fn(state):
        f = extract_features(
            state,
            game_config.get("start_credits", 1000),
            game_config.get("big_blind", 20),
        )
        logits = nn.forward(f)
        return masked_argmax(logits, state.valid_actions)

    return evaluate_agent(agent_fn, opponent_fn, game_config, hands_per_eval)


class EvolutionStrategies:
    """ES with antithetic sampling and rank-based fitness.

    Update rule (antithetic):
        θ_new = θ_old + α/(N·σ) · Σ ((f⁺_i − f⁻_i) · ε_i)

    where f⁺_i = fitness(θ + ε_i),  f⁻_i = fitness(θ − ε_i),
    and fitness is the **rank** of the profit within the population.
    """

    def __init__(
        self,
        population_size: int = 60,
        sigma: float = 0.04,
        alpha: float = 0.02,
        rng: np.random.Generator | None = None,
    ):
        # Half the pop because we evaluate + and − for each noise vector
        assert population_size % 2 == 0, "pop_size must be even (antithetic pairs)"
        self.half_pop = population_size // 2
        self.sigma = sigma
        self.alpha = alpha
        self.rng = rng or np.random.default_rng()

    def train_generation(
        self,
        parent: SmallNN,
        opponent_fn: Callable,
        game_config: dict,
        hands_per_eval: int = 200,
        parallel: int = 1,
    ) -> tuple[SmallNN, dict]:
        """Run one ES generation.  Returns (updated_parent, stats_dict).

        Parameters
        ----------
        parallel : int
            Number of worker processes to use (1 = sequential, >1 = parallel).
        """

        base = parent.get_weights()
        n_params = parent.num_params

        # --- Generate noise (half_pop vectors) ----------------------------
        eps = self.rng.normal(0, self.sigma, size=(self.half_pop, n_params)).astype(
            np.float32
        )

        # --- Build task list -----------------------------------------------
        total_individuals = self.half_pop * 2
        tasks = []
        for i in range(self.half_pop):
            tasks.append((base + eps[i], game_config, opponent_fn, hands_per_eval))
            tasks.append((base - eps[i], game_config, opponent_fn, hands_per_eval))

        # --- Evaluate all individuals -------------------------------------
        raw = np.zeros(total_individuals, dtype=np.float32)
        winrates = np.zeros(total_individuals, dtype=np.float32)

        if parallel > 1:
            # Parallel across worker processes
            n_workers = min(parallel, total_individuals, os.cpu_count() or 1)
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(_eval_individual, t): idx for idx, t in enumerate(tasks)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    raw[idx], winrates[idx] = future.result()
        else:
            # Sequential
            for idx, task in enumerate(tasks):
                raw[idx], winrates[idx] = _eval_individual(task)

        # --- Rank-based fitness (robust to outliers) -----------------------
        ranks = np.argsort(np.argsort(raw)).astype(np.float32)
        total = float(len(ranks) - 1)
        fitness = (ranks / max(total, 1.0)) - 0.5  # ∈ [-0.5, 0.5]

        # Reconstruct antithetic pairs: pos are even indices, neg are odd
        f_pos = fitness[0::2]  # indices 0, 2, 4, ...
        f_neg = fitness[1::2]  # indices 1, 3, 5, ...
        diff = f_pos - f_neg

        # --- Weighted update -----------------------------------------------
        update = np.zeros(n_params, dtype=np.float32)
        for i in range(self.half_pop):
            update += diff[i] * eps[i]

        update /= self.half_pop * self.sigma

        new_weights = base + self.alpha * update
        parent.set_weights(new_weights)

        stats = {
            "profit_mean": float(np.mean(raw)),
            "profit_std": float(np.std(raw)),
            "profit_max": float(np.max(raw)),
            "profit_min": float(np.min(raw)),
            "winrate_mean": float(np.mean(winrates)),
            "winrate_max": float(np.max(winrates)),
            "update_norm": float(np.linalg.norm(update)),
            "best_weights": tasks[int(np.argmax(winrates))][0],
            "best_profit": float(raw[int(np.argmax(winrates))]),
            "best_winrate": float(np.max(winrates)),
        }
        return parent, stats
