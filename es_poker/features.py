"""Feature extraction: converts a poker StateView into a normalized vector."""

import numpy as np
from pokerl.game import Game
from pokerl.judger import eval_hand
from pokerl.enums import HandRanking

FEATURE_DIM = 26


def extract_features(
    state: Game.StateView,
    start_credits: float = 1000.0,
    big_blind: float = 20.0,
) -> np.ndarray:
    """Convert a StateView into a normalized feature vector of shape (FEATURE_DIM,).

    Features (26 total):
      Card features (19):
        0-3   : hole cards (rank, suit) x 2
        4-13  : community cards (rank, suit) x 5, zero-padded
        14    : hand strength (1.0=straight flush ... 0.0=nothing)
        15    : is pocket pair (1.0/0.0)
        16    : is suited (1.0/0.0)
        17    : high card rank / 13
        18    : low card rank / 13

      Situation features (7):
        19    : active players / num_players
        20    : stack in big blinds (clipped)
        21    : pot / (pot + my_credits)  ->  pot commitment
        22    : facing raise (1.0 if call needed, else 0.0)
        23    : pot odds: call_needed / (pot + call_needed + 1)
        24    : position (player_index / num_players)
        25    : turn / 4.0
    """

    f = np.zeros(FEATURE_DIM, dtype=np.float32)

    c0, c1 = state.player_cards

    # --- hole cards (indices 0–3) ---
    f[0] = c0.rank / 13.0
    f[1] = c0.suit / 4.0
    f[2] = c1.rank / 13.0
    f[3] = c1.suit / 4.0

    # --- community cards (indices 4–13) ---
    idx = 4
    for i in range(5):
        if i < len(state.community_cards):
            f[idx] = state.community_cards[i].rank / 13.0
            f[idx + 1] = state.community_cards[i].suit / 4.0
        idx += 2

    # --- hand strength (index 14) ---
    if state.player_hand:
        ranking, _ = eval_hand(state.player_hand)
        f[14] = max(0.0, (HandRanking.NONE - ranking) / 9.0)

    # --- preflop hand categories (indices 15–18) ---
    r0, r1 = c0.rank, c1.rank
    f[15] = 1.0 if r0 == r1 else 0.0                     # pocket pair
    f[16] = 1.0 if c0.suit == c1.suit else 0.0           # suited
    f[17] = max(r0, r1) / 13.0                            # high card
    f[18] = min(r0, r1) / 13.0                            # low card

    # --- situation features (indices 19–25) ---
    active = int(np.sum(state.credits > 0))
    f[19] = active / max(state.num_players, 1)

    stack_bb = state.credit / max(big_blind, 1.0)
    f[20] = np.clip(stack_bb / 100.0, 0.0, 1.0)

    f[21] = state.pot / max(state.pot + state.credit, 1.0)

    my_pending = state.pending_bets[state.player]
    f[22] = 1.0 if state.high_bet > my_pending + 0.01 else 0.0

    call_needed = state.high_bet - my_pending
    f[23] = call_needed / max(state.pot + call_needed, 1.0)

    f[24] = state.player / max(state.num_players, 1)
    f[25] = state.turn / 4.0

    return f
