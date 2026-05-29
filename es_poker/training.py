# protocollo di self-play a 3 sprint (Prof. Tronci)
# sprint A: random init vs random
# sprint B: warm-start da A vs A congelato
# sprint A': warm-start da B vs B congelato

import csv
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .agent import ESAgent, RandomAgent
from .evolution import EvolutionStrategies, valuta_agente
from .network import SmallNN


def _salva_pesi(network: SmallNN, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, weights=network.get_weights())


def _carica_pesi(network: SmallNN, path: Path):
    data = np.load(path)
    network.set_weights(data["weights"])


class TrainingProtocol:
    """orchestra il curriculum di self-play a 3 sprint

    attributi di configurazione (modificabili prima di chiamare .run()):
      output_dir, num_players, start_credits, big_blind, small_blind,
      pop_size, sigma, alpha, generations, hands_per_eval,
      patience, parallel, seed
    """

    # defaults configurabili
    output_dir: str = "./training_output"
    num_players: int = 2
    start_credits: int = 1000
    big_blind: int = 20
    small_blind: int = 10
    pop_size: int = 40
    sigma: float = 0.04
    alpha: float = 0.02
    generations: int = 25
    hands_per_eval: int = 200
    patience: int = 15
    parallel: bool = False
    seed: int = 42

    def __init__(self):
        self.history: list[dict] = []

    @property
    def _game_config(self) -> dict:
        return {
            "num_players": self.num_players,
            "start_credits": self.start_credits,
            "big_blind": self.big_blind,
            "small_blind": self.small_blind,
        }

    def _run_sprint(
        self,
        nome: str,
        agent_nn: SmallNN,
        opponent_fn: Callable,
        rng: np.random.Generator,
    ) -> tuple[SmallNN, list[dict]]:
        """allena un agente per self.generations (o fino a esaurimento pazienza)"""
        es = EvolutionStrategies(
            population_size=self.pop_size,
            sigma=self.sigma,
            alpha=self.alpha,
            rng=rng,
        )

        gen_logs = []
        best_fitness = -float("inf")
        pazienza = 0
        best_wr = -float("inf")
        best_weights = None

        for gen in range(self.generations):
            t0 = time.perf_counter()
            agent_nn, stats = es.train_generation(
                agent_nn,
                opponent_fn,
                self._game_config,
                self.hands_per_eval,
                parallel=self.parallel,
            )
            elapsed = time.perf_counter() - t0

            stats["generation"] = gen
            stats["sprint"] = nome
            stats["elapsed_s"] = elapsed

            if stats["best_winrate"] > best_wr:
                best_wr = stats["best_winrate"]
                best_weights = stats["best_weights"]

            log_entry = {k: v for k, v in stats.items() if k != "best_weights"}
            gen_logs.append(log_entry)

            print(
                f"  [{nome}] gen {gen:3d} | "
                f"profit={stats['profit_mean']:+.2f}  "
                f"max={stats['profit_max']:+.2f}  "
                f"wr={stats['winrate_mean']:.1%}  "
                f"||u||={stats['update_norm']:.4f}  "
                f"t={elapsed:.1f}s"
            )

            if stats["winrate_mean"] > best_fitness:
                best_fitness = stats["winrate_mean"]
                pazienza = 0
            else:
                pazienza += 1

            if pazienza >= self.patience:
                print(f"  [{nome}] early stop a gen {gen} (patience={self.patience})")
                break

        if best_weights is not None:
            best_nn = SmallNN()
            best_nn.set_weights(best_weights)
            path = Path(self.output_dir) / f"best_{nome}.npz"
            _salva_pesi(best_nn, path)
            print(f"  [{nome}] miglior individuo salvato: wr={best_wr:.1%} → {path}")

        return agent_nn, gen_logs

    def run(self):
        """esegue il protocollo completo"""
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(self.seed)

        # sprint 1: agente A vs random
        print("=== Sprint 1: Agente A vs Random ===")
        nn_A = SmallNN(rng)
        nn_A, logs_A = self._run_sprint("A", nn_A, RandomAgent(), rng)
        self.history.extend(logs_A)
        _salva_pesi(nn_A, out / "agent_A.npz")

        # sprint 2: agente B vs A (warm-start da A, A congelato)
        print("\n=== Sprint 2: Agente B vs Agente A ===")
        nn_B = SmallNN()
        nn_B.set_weights(nn_A.get_weights().copy())
        frozen_A = ESAgent(nn_A, self.start_credits, self.big_blind)
        nn_B, logs_B = self._run_sprint("B", nn_B, frozen_A, rng)
        self.history.extend(logs_B)
        _salva_pesi(nn_B, out / "agent_B.npz")

        # sprint 3: agente A' vs B (warm-start da B, iterazione self-play)
        print("\n=== Sprint 3: Agente A' vs Agente B ===")
        nn_A2 = SmallNN()
        nn_A2.set_weights(nn_B.get_weights().copy())
        frozen_B = ESAgent(nn_B, self.start_credits, self.big_blind)
        nn_A2, logs_A2 = self._run_sprint("A2", nn_A2, frozen_B, rng)
        self.history.extend(logs_A2)
        _salva_pesi(nn_A2, out / "agent_A2.npz")

        # valutazione finale
        print("\n=== Valutazione Finale ===")
        final = ESAgent(nn_A2, self.start_credits, self.big_blind)
        frozen_A_agent = ESAgent(nn_A, self.start_credits, self.big_blind)
        frozen_B_agent = ESAgent(nn_B, self.start_credits, self.big_blind)

        vs_random, wr_random = valuta_agente(final, RandomAgent(), self._game_config, num_hands=500)
        vs_A, wr_A = valuta_agente(final, frozen_A_agent, self._game_config, num_hands=500)
        vs_B, wr_B = valuta_agente(final, frozen_B_agent, self._game_config, num_hands=500)

        summary = {
            "vs_random_profit": vs_random,
            "vs_random_winrate": wr_random,
            "vs_A_profit": vs_A,
            "vs_A_winrate": wr_A,
            "vs_B_profit": vs_B,
            "vs_B_winrate": wr_B,
        }

        print("\nPerformance Agente A':")
        for opp, key in [("Random", "vs_random"), ("Agente A", "vs_A"), ("Agente B", "vs_B")]:
            print(f"  vs {opp}: {summary[f'{key}_profit']:+.2f} chips/mano  (wr: {summary[f'{key}_winrate']:.1%})")

        # salva riepilogo JSON
        with open(out / "final_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # salva storico JSON
        with open(out / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        # salva storico CSV e grafico PNG
        csv_path = out / "training_history.csv"
        self._salva_csv(csv_path)
        from .plotter import salva_grafico
        salva_grafico(csv_path)
        print(f"grafico salvato → {csv_path.with_suffix('.png')}")

        return summary

    def _salva_csv(self, path: Path):
        """scrive lo storico in formato CSV per plotting"""
        if not self.history:
            return
        colonne = [
            "generation", "sprint", "profit_mean", "profit_std",
            "profit_max", "profit_min", "winrate_mean", "winrate_max",
            "update_norm", "elapsed_s",
        ]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=colonne, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.history)

    @staticmethod
    def carica_agente(
        weights_path: str | Path,
        start_credits: float = 1000.0,
        big_blind: float = 20.0,
    ) -> ESAgent:
        """carica un agente da file .npz"""
        nn = SmallNN()
        _carica_pesi(nn, Path(weights_path))
        return ESAgent(nn, start_credits, big_blind)
