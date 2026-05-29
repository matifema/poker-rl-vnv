"""Training protocol implementing Prof. Tronci's self-play curriculum."""

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .agent import ESAgent, RandomAgent
from .evolution import EvolutionStrategies, evaluate_agent
from .network import SmallNN

# Default game configuration used throughout training
DEFAULT_GAME_CONFIG = {
    "num_players": 2,
    "start_credits": 1000,
    "big_blind": 20,
    "small_blind": 10,
}


def _save_weights(network: SmallNN, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, weights=network.get_weights())


def _load_weights(network: SmallNN, path: Path):
    data = np.load(path)
    network.set_weights(data["weights"])


class TrainingProtocol:
    """Orchestrates the three-sprint self-play curriculum.

    Sprint 1 : Agent A  vs  RandomAgent  (baseline)
    Sprint 2 : Agent B  vs  Agent A       (frozen)
    Sprint 3+: alternate roles until a target number of iterations.
    """

    def __init__(
        self,
        output_dir: str | Path = "./training_output",
        game_config: dict | None = None,
        es_config: dict | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.game_config = {**DEFAULT_GAME_CONFIG, **(game_config or {})}
        es_cfg = {
            "population_size": 40,
            "sigma": 0.04,
            "alpha": 0.02,
        }
        es_cfg.update(es_config or {})
        self.es_config = es_cfg

        self.history: list[dict] = []  # training logs per generation

    # ------------------------------------------------------------------
    # Sprint helpers
    # ------------------------------------------------------------------

    def _run_sprint(
        self,
        name: str,
        agent_nn: SmallNN,
        opponent_fn: Callable,
        generations: int = 30,
        hands_per_eval: int = 200,
        patience: int = 15,
        parallel: int = 1,
        rng: np.random.Generator | None = None,
    ) -> tuple[SmallNN, list[dict]]:
        """Run ES for *generations* (or until patience runs out).

        Returns (trained_nn, per_generation_logs).
        """

        rng = rng or np.random.default_rng()
        es = EvolutionStrategies(
            population_size=self.es_config["population_size"],
            sigma=self.es_config["sigma"],
            alpha=self.es_config["alpha"],
            rng=rng,
        )

        gen_logs = []
        best_fitness = -float("inf")
        patience_counter = 0
        best_indiv_wr = -float("inf")
        best_indiv_weights = None

        for gen in range(generations):
            t0 = time.perf_counter()
            agent_nn, stats = es.train_generation(
                agent_nn,
                opponent_fn,
                self.game_config,
                hands_per_eval,
                parallel=parallel,
            )
            elapsed = time.perf_counter() - t0

            stats["generation"] = gen
            stats["sprint"] = name
            stats["elapsed_s"] = elapsed
            # Keep best_weights for in-memory tracking but strip for JSON
            if stats["best_winrate"] > best_indiv_wr:
                best_indiv_wr = stats["best_winrate"]
                best_indiv_weights = stats["best_weights"]
            log_entry = {k: v for k, v in stats.items() if k != "best_weights"}
            gen_logs.append(log_entry)

            print(
                f"  [{name}] gen {gen:3d} | "
                f"profit={stats['profit_mean']:+.2f}  "
                f"max={stats['profit_max']:+.2f}  "
                f"wr={stats['winrate_mean']:.1%}  "
                f"||u||={stats['update_norm']:.4f}  "
                f"t={elapsed:.1f}s"
            )

            # Early stopping on win-rate plateau (less noisy than profit)
            if stats["winrate_mean"] > best_fitness:
                best_fitness = stats["winrate_mean"]
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"  [{name}] Early stop at gen {gen} (patience={patience})")
                break

        # Save best individual found in this sprint
        if best_indiv_weights is not None:
            best_nn = SmallNN()
            best_nn.set_weights(best_indiv_weights)
            path = self.output_dir / f"best_{name}.npz"
            _save_weights(best_nn, path)
            print(f"  [{name}] Best individual saved: wr={best_indiv_wr:.1%} → {path}")

        return agent_nn, gen_logs

    # ------------------------------------------------------------------
    # Full curriculum
    # ------------------------------------------------------------------

    def run(
        self,
        generations_per_sprint: int = 30,
        hands_per_eval: int = 200,
        parallel: int = 1,
    ):
        """Execute the full training protocol."""

        rng = np.random.default_rng(42)
        start_credits = self.game_config["start_credits"]
        big_blind = self.game_config["big_blind"]

        # ── Sprint 1: Agent A vs Random ───────────────────────────────
        print("=== Sprint 1: Agent A  vs  Random ===")
        nn_A = SmallNN(rng)
        random_agent = RandomAgent()

        nn_A, logs_A = self._run_sprint(
            "A",
            nn_A,
            random_agent,
            generations=generations_per_sprint,
            hands_per_eval=hands_per_eval,
            parallel=parallel,
            rng=rng,
        )
        self.history.extend(logs_A)
        _save_weights(nn_A, self.output_dir / "agent_A.npz")

        # ── Sprint 2: Agent B vs Agent A (frozen) ─────────────────────
        print("\n=== Sprint 2: Agent B  vs  Agent A ===")
        nn_B = SmallNN(rng)
        frozen_A = ESAgent(nn_A, start_credits, big_blind)

        nn_B, logs_B = self._run_sprint(
            "B",
            nn_B,
            frozen_A,
            generations=generations_per_sprint,
            hands_per_eval=hands_per_eval,
            parallel=parallel,
            rng=rng,
        )
        self.history.extend(logs_B)
        _save_weights(nn_B, self.output_dir / "agent_B.npz")

        # ── Sprint 3: Agent A' vs Agent B (self-play iteration) ───────
        print("\n=== Sprint 3: Agent A'  vs  Agent B ===")
        nn_A2 = SmallNN(rng)
        frozen_B = ESAgent(nn_B, start_credits, big_blind)

        nn_A2, logs_A2 = self._run_sprint(
            "A2",
            nn_A2,
            frozen_B,
            generations=generations_per_sprint,
            hands_per_eval=hands_per_eval,
            parallel=parallel,
            rng=rng,
        )
        self.history.extend(logs_A2)
        _save_weights(nn_A2, self.output_dir / "agent_A2.npz")

        # ── Final evaluation ──────────────────────────────────────────
        print("\n=== Final Evaluation ===")
        final_agent = ESAgent(nn_A2, start_credits, big_blind)

        vs_random, wr_random = evaluate_agent(
            final_agent, RandomAgent(), self.game_config, num_hands=500
        )
        vs_A, wr_A = evaluate_agent(
            final_agent,
            ESAgent(nn_A, start_credits, big_blind),
            self.game_config,
            num_hands=500,
        )
        vs_B, wr_B = evaluate_agent(
            final_agent,
            ESAgent(nn_B, start_credits, big_blind),
            self.game_config,
            num_hands=500,
        )

        summary = {
            "vs_random_profit": vs_random,
            "vs_random_winrate": wr_random,
            "vs_A_profit": vs_A,
            "vs_A_winrate": wr_A,
            "vs_B_profit": vs_B,
            "vs_B_winrate": wr_B,
        }
        print(f"\nFinal Agent A' performance:")
        for opponent, key in [
            ("Random", "vs_random"),
            ("Agent A", "vs_A"),
            ("Agent B", "vs_B"),
        ]:
            p = summary[f"{key}_profit"]
            w = summary[f"{key}_winrate"]
            print(f"  vs {opponent}: {p:+.2f} chips/hand  (win rate: {w:.1%})")

        # Save summary
        with open(self.output_dir / "final_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Save full history
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        return summary

    # ------------------------------------------------------------------
    # Quick evaluation helper (used by V&V scripts)
    # ------------------------------------------------------------------

    @staticmethod
    def load_agent(
        weights_path: str | Path,
        start_credits: float = 1000.0,
        big_blind: float = 20.0,
    ) -> ESAgent:
        nn = SmallNN()
        _load_weights(nn, Path(weights_path))
        return ESAgent(nn, start_credits, big_blind)
