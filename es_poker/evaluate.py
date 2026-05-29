# verifica e validazione: test unitari, invarianti, report performance

import json
from pathlib import Path
from typing import Callable

import numpy as np
from pokerl.agents.agent import PokerAgent
from pokerl.enums import PokerMoves
from pokerl.game import Game

from .agent import ESAgent, RandomAgent, masked_argmax
from .evolution import valuta_agente
from .features import FEATURE_DIM, extract_features
from .network import SmallNN


def _rete_random(rng: np.random.Generator | None = None) -> SmallNN:
    return SmallNN(rng or np.random.default_rng())


# ---------------------------------------------------------------------------
# test unitari: action masking
# ---------------------------------------------------------------------------


def test_action_masking_all_valid():
    """tutte le azioni valide → maschera non altera argmax"""

    logits = np.array([1.0, 2.0, 0.5, -0.3, 4.0, 3.0, -1.0], dtype=np.float32)
    valid = np.ones(7, dtype=bool)
    action = masked_argmax(logits, valid)
    assert action == 4, f"attesa azione 4 (max logit=4.0), ottenuta {action}"


def test_action_masking_single_valid():
    """una sola azione valida → sempre scelta"""
    logits = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for i in range(7):
        valid = np.zeros(7, dtype=bool)
        valid[i] = True
        action = masked_argmax(logits, valid)
        assert action == i, f"solo azione {i} valida ma ottenuta {action}"


def test_action_masking_never_picks_invalid():
    """test randomizzato: mai azione invalida"""
    rng = np.random.default_rng(1234)
    for _ in range(1000):
        logits = rng.standard_normal(7).astype(np.float32)
        valid = rng.choice([True, False], size=7)
        if not valid.any():
            valid[rng.integers(0, 7)] = True
        action = masked_argmax(logits, valid)
        assert valid[action], (
            f"azione invalida {action} con logits={logits} mask={valid}"
        )


def test_action_masking_inf_handling():
    """logits con -inf non rompono la maschera"""
    logits = np.array(
        [-np.inf, 1.0, 2.0, -np.inf, 0.0, -1.0, -np.inf], dtype=np.float32
    )
    valid = np.array([True, True, True, False, True, True, False])
    action = masked_argmax(logits, valid)
    assert action == 2


# ---------------------------------------------------------------------------
# controllo invarianti
# ---------------------------------------------------------------------------


def test_invariant_no_value_error(game_config: dict | None = None):
    """l'agente non deve mai sollevare ValueError con pesi NN casuali"""
    cfg = game_config or {
        "num_players": 4,
        "start_credits": 1000,
        "big_blind": 20,
        "small_blind": 10,
    }
    rng = np.random.default_rng(42)

    for trial in range(50):
        nn = _rete_random(rng)
        agent = ESAgent(nn, cfg["start_credits"], cfg["big_blind"])
        players = [agent, RandomAgent(), RandomAgent(), RandomAgent()]
        game = Game(**cfg)
        game.reset()

        steps = 0
        error = False
        while steps < 500:
            state = game.active_state
            try:
                action = players[game.active_player](state)
                done, hand_over, _ = game.step(action)
            except ValueError:
                error = True
                break
            if done:
                game.reset()
            steps += 1

        assert not error, f"ValueError al trial {trial} dopo {steps} passi"


# ---------------------------------------------------------------------------
# validazione: report performance
# ---------------------------------------------------------------------------


def stampa_report(
    agent_fn: Callable,
    opponent_fn: Callable,
    game_config: dict,
    num_hands: int = 500,
    label: str = "Agente",
) -> dict:
    """esegue match di validazione speculare e stampa report"""
    profit, winrate = valuta_agente(agent_fn, opponent_fn, game_config, num_hands)

    print(f"\n--- Validazione: {label} ---")
    print(f"  mani giocate     : {num_hands}")
    print(f"  profit medio     : {profit:+.3f}")
    print(f"  win rate         : {winrate:.1%}")

    return {
        "label": label,
        "num_hands": num_hands,
        "mean_profit": profit,
        "win_rate": winrate,
    }


def validation_suite(
    agent_weights_path: str | Path,
    game_config: dict | None = None,
    output_dir: str | Path = "./training_output",
):
    """validazione completa: agente vs Random, agente vs Tight"""
    from .training import TrainingProtocol

    cfg = game_config or {
        "num_players": 4,
        "start_credits": 1000,
        "big_blind": 20,
        "small_blind": 10,
    }

    agent = TrainingProtocol.carica_agente(
        agent_weights_path, cfg["start_credits"], cfg["big_blind"]
    )

    class TightAgent(PokerAgent):
        """gioca solo mani premium (coppia JJ+ o suited A)"""

        def __call__(self, state: Game.StateView) -> int:
            vu = state.valid_actions
            cards = state.player_cards
            high_ranks = {11, 12, 13}
            suited = cards[0].suit == cards[1].suit
            strong = (
                cards[0].rank in high_ranks
                and cards[1].rank in high_ranks
                and (cards[0].rank >= 11 and cards[1].rank >= 11 or suited)
            )
            if strong and vu[PokerMoves.RAISE_TEN]:
                return PokerMoves.RAISE_TEN
            if vu[PokerMoves.CALL]:
                return PokerMoves.CALL
            if vu[PokerMoves.CHECK]:
                return PokerMoves.CHECK
            return PokerMoves.FOLD

    reports = {}
    reports["vs_random"] = stampa_report(
        agent, RandomAgent(), cfg, num_hands=500, label="Agente vs Random"
    )
    reports["vs_tight"] = stampa_report(
        agent, TightAgent(), cfg, num_hands=500, label="Agente vs Tight"
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "validation_report.json", "w") as f:
        json.dump(reports, f, indent=2)

    return reports
