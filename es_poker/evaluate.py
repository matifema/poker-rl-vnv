"""Verification & Validation: unit tests, invariant checking, and performance graphs."""

import json
from pathlib import Path
from typing import Callable

import numpy as np

from pokerl.agents.agent import PokerAgent
from pokerl.game import Game
from pokerl.enums import PokerMoves

from .agent import ESAgent, RandomAgent, masked_argmax
from .network import SmallNN
from .features import extract_features, FEATURE_DIM
from .evolution import evaluate_agent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_random_nn(rng: np.random.Generator | None = None) -> SmallNN:
    return SmallNN(rng or np.random.default_rng())


# ---------------------------------------------------------------------------
# Verification: Action Masking Unit Tests
# ---------------------------------------------------------------------------


def test_action_masking_all_valid():
    """When all actions are valid, the mask must not alter the argmax."""
    logits = np.array([1.0, 2.0, 0.5, -0.3, 4.0, 3.0, -1.0], dtype=np.float32)
    valid = np.ones(7, dtype=bool)
    action = masked_argmax(logits, valid)
    assert action == 4, f"Expected action 4 (max logit=4.0), got {action}"


def test_action_masking_single_valid():
    """When only one action is valid, it must always be chosen."""
    logits = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for i in range(7):
        valid = np.zeros(7, dtype=bool)
        valid[i] = True
        action = masked_argmax(logits, valid)
        assert action == i, f"Only action {i} is valid but got {action}"


def test_action_masking_never_picks_invalid():
    """Randomised test: the agent must never output an invalid action."""
    rng = np.random.default_rng(1234)
    for _ in range(1000):
        logits = rng.standard_normal(7).astype(np.float32)
        valid = rng.choice([True, False], size=7)
        if not valid.any():
            valid[rng.integers(0, 7)] = True
        action = masked_argmax(logits, valid)
        assert valid[action], (
            f"Picked invalid action {action} with logits={logits} mask={valid}"
        )


def test_action_masking_inf_handling():
    """Logits with -inf values should not break the masking."""
    logits = np.array([-np.inf, 1.0, 2.0, -np.inf, 0.0, -1.0, -np.inf],
                      dtype=np.float32)
    valid = np.array([True, True, True, False, True, True, False])
    action = masked_argmax(logits, valid)
    assert action == 2


# ---------------------------------------------------------------------------
# Verification: Invariant Checking
# ---------------------------------------------------------------------------


def test_invariant_no_value_error(game_config: dict | None = None):
    """The agent must NEVER raise a ValueError, regardless of NN weights."""
    cfg = game_config or {
        "num_players": 4,
        "start_credits": 1000,
        "big_blind": 20,
        "small_blind": 10,
    }
    rng = np.random.default_rng(42)

    for trial in range(50):
        nn = _make_random_nn(rng)
        agent = ESAgent(nn, cfg["start_credits"], cfg["big_blind"])
        all_players = [agent, RandomAgent(), RandomAgent(), RandomAgent()]
        game = Game(**cfg)
        game.reset()

        steps = 0
        error = False
        while steps < 500:
            state = game.active_state
            try:
                action = all_players[game.active_player](state)
                done, hand_over, _ = game.step(action)
            except ValueError:
                error = True
                break
            if done:
                game.reset()
            steps += 1

        assert not error, f"ValueError on trial {trial} after {steps} steps"


# ---------------------------------------------------------------------------
# Validation: Performance Stats & Graphs
# ---------------------------------------------------------------------------


def print_validation_report(
    agent_fn: Callable,
    opponent_fn: Callable,
    game_config: dict,
    num_hands: int = 500,
    label: str = "Agent",
) -> dict:
    """Run a mirrored validation match and print a concise report."""
    profit, winrate = evaluate_agent(agent_fn, opponent_fn, game_config, num_hands)

    print(f"\n--- Validation: {label} ---")
    print(f"  Hands played     : {num_hands}")
    print(f"  Mean profit/hand : {profit:+.3f}")
    print(f"  Win rate         : {winrate:.1%}")

    return {
        "label": label,
        "num_hands": num_hands,
        "mean_profit": profit,
        "win_rate": winrate,
    }


def run_validation_suite(
    agent_weights_path: str | Path,
    game_config: dict | None = None,
    output_dir: str | Path = "./training_output",
):
    """Full validation suite: agent vs Random, agent vs Tight baseline."""
    from .training import TrainingProtocol

    cfg = game_config or {
        "num_players": 4,
        "start_credits": 1000,
        "big_blind": 20,
        "small_blind": 10,
    }

    agent = TrainingProtocol.load_agent(
        agent_weights_path, cfg["start_credits"], cfg["big_blind"]
    )

    class TightAgent(PokerAgent):
        """Baseline: only plays premium hands (pair JJ+ or suited A)."""

        def __call__(self, state: Game.StateView) -> int:
            vu = state.valid_actions
            cards = state.player_cards
            high_ranks = {11, 12, 13}  # J, Q, K, A
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
    reports["vs_random"] = print_validation_report(
        agent, RandomAgent(), cfg, num_hands=500, label="Agent vs Random"
    )
    reports["vs_tight"] = print_validation_report(
        agent, TightAgent(), cfg, num_hands=500, label="Agent vs Tight"
    )

    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "validation_report.json", "w") as f:
        json.dump(reports, f, indent=2)

    return reports
