# ottimizzatore ES con campionamento antitetico (OpenAI ES, Algo 1)
# ∇ = 1/(2n·σ²) · Σ_i (F⁺_i − F⁻_i) · ε_i   con ε_i ~ N(0, σ²I)
# update: θ' = θ + α · ∇

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable

import numpy as np

from .network import SmallNN


def _gioca_mani(
    players: list[Callable],
    agent_idx: int,
    game_config: dict,
    num_hands: int,
) -> tuple[float, float]:
    """gioca num_hands mani, restituisce (profit_medio, win_rate)"""
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


def valuta_agente(
    agent_fn: Callable,
    opponent_fn: Callable,
    game_config: dict,
    num_hands: int = 200,
) -> tuple[float, float]:
    """valutazione speculare: agente gioca da ogni posizione"""
    n_players = game_config.get("num_players", 2)
    hands_per_seat = max(1, num_hands // n_players)
    total_profit = 0.0
    total_wins = 0

    for seat in range(n_players):
        players = [opponent_fn] * n_players
        players[seat] = agent_fn
        profit, wr = _gioca_mani(players, seat, game_config, hands_per_seat)
        total_profit += profit
        total_wins += wr

    return total_profit / n_players, total_wins / n_players


def _eval_coppia(args: tuple) -> tuple[float, float, float, float]:
    """worker per coppia antithetica (Algo 2): valuta θ+ε e θ−ε nello stesso job"""
    base, eps_vec, game_config, opponent_fn, hands_per_eval = args

    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parent.parent
    _pokerl = _root / "pokerl"
    for _p in (str(_pokerl), str(_root)):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    from .agent import masked_argmax
    from .features import extract_features
    from .network import SmallNN

    nn = SmallNN()

    def _eval_weights(w):
        nn.set_weights(w)
        def agent_fn(state):
            f = extract_features(
                state,
                game_config.get("start_credits", 1000),
                game_config.get("big_blind", 20),
            )
            logits = nn.forward(f)
            return masked_argmax(logits, state.valid_actions)
        return valuta_agente(agent_fn, opponent_fn, game_config, hands_per_eval)

    profit_pos, wr_pos = _eval_weights(base + eps_vec)
    profit_neg, wr_neg = _eval_weights(base - eps_vec)
    return profit_pos, wr_pos, profit_neg, wr_neg


def _eval_individuo(args: tuple) -> tuple[float, float]:
    """worker per valutazione parallela di un individuo (fallback non-antithetico)"""
    weights, game_config, opponent_fn, hands_per_eval = args

    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parent.parent
    _pokerl = _root / "pokerl"
    for _p in (str(_pokerl), str(_root)):
        if _p not in sys.path:
            sys.path.insert(0, _p)

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

    return valuta_agente(agent_fn, opponent_fn, game_config, hands_per_eval)


class EvolutionStrategies:
    """ottimizzatore ES (Algo 1)"""

    def __init__(
        self,
        population_size: int = 40,
        sigma: float = 0.04,
        alpha: float = 0.02,
        rng: np.random.Generator | None = None,
    ):
        assert population_size % 2 == 0, "pop_size deve essere pari (coppie antitetiche)"
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
        parallel: bool = False,
    ) -> tuple[SmallNN, dict]:
        """esegue una generazione ES, restituisce (rete_aggiornata, stats)"""
        base = parent.get_weights()
        n_params = parent.num_params

        # genera rumore per mezza popolazione (coppie antitetiche)
        eps = self.rng.normal(0, self.sigma, size=(self.half_pop, n_params)).astype(np.float32)

        # costruisce task: θ+ε e θ−ε
        total = self.half_pop * 2
        tasks = []
        for i in range(self.half_pop):
            tasks.append((base + eps[i], game_config, opponent_fn, hands_per_eval))
            tasks.append((base - eps[i], game_config, opponent_fn, hands_per_eval))

        raw = np.zeros(total, dtype=np.float32)
        winrates = np.zeros(total, dtype=np.float32)

        if parallel:
            n_workers = min(os.cpu_count() or 1, self.half_pop)
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                pair_tasks = [
                    (base, eps[i], game_config, opponent_fn, hands_per_eval)
                    for i in range(self.half_pop)
                ]
                futures = {pool.submit(_eval_coppia, t): i for i, t in enumerate(pair_tasks)}
                for future in as_completed(futures):
                    i = futures[future]
                    profit_pos, wr_pos, profit_neg, wr_neg = future.result()
                    raw[2 * i] = profit_pos
                    winrates[2 * i] = wr_pos
                    raw[2 * i + 1] = profit_neg
                    winrates[2 * i + 1] = wr_neg
        else:
            for idx, task in enumerate(tasks):
                raw[idx], winrates[idx] = _eval_individuo(task)

        # fitness: z-score normalization (paper Sec 3, standardizzaz. dei ritorni)
        raw_mean = np.mean(raw)
        raw_std = np.std(raw) + 1e-8
        fitness = (raw - raw_mean) / raw_std

        # ricostruisce coppie antitetiche: pari=+ε, dispari=−ε
        f_pos = fitness[0::2]
        f_neg = fitness[1::2]
        diff = f_pos - f_neg

        # ∇ = 1/(2n·σ²) · Σ_i diff_i · ε_i   (paper Algo 1, n = half_pop)
        update = np.zeros(n_params, dtype=np.float32)
        for i in range(self.half_pop):
            update += diff[i] * eps[i]
        update /= (2 * self.half_pop * self.sigma * self.sigma)

        new_weights = base + self.alpha * update
        parent.set_weights(new_weights)

        best_idx = int(np.argmax(winrates))
        stats = {
            "profit_mean": float(np.mean(raw)),
            "profit_std": float(np.std(raw)),
            "profit_max": float(np.max(raw)),
            "profit_min": float(np.min(raw)),
            "winrate_mean": float(np.mean(winrates)),
            "winrate_max": float(np.max(winrates)),
            "update_norm": float(np.linalg.norm(update)),
            "best_weights": tasks[best_idx][0],
            "best_profit": float(raw[best_idx]),
            "best_winrate": float(np.max(winrates)),
        }
        return parent, stats
