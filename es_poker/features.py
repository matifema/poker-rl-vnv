# estrazione feature: converte StateView in vettore normalizzato (26 dim)

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
    """converte StateView in vettore normalizzato (26 dim)

    feature carte (19):
      0-3   : hole cards (rank, suit) x 2
      4-13  : community cards (rank, suit) x 5, zero-padded
      14    : forza mano (1.0=scala colore ... 0.0=niente)
      15    : coppia in mano (1.0/0.0)
      16    : suited (1.0/0.0)
      17    : rank carta alta / 13
      18    : rank carta bassa / 13

    feature situazione (7):
      19    : giocatori attivi / totale
      20    : stack in big blind (clippato)
      21    : pot / (pot + miei crediti)  →  pot commitment
      22    : affronta rilancio (1.0 se call necessario)
      23    : pot odds: call / (pot + call + 1)
      24    : posizione (indice / num_players)
      25    : turn / 4.0
    """
    f = np.zeros(FEATURE_DIM, dtype=np.float32)
    c0, c1 = state.player_cards

    # hole cards (0-3)
    f[0] = c0.rank / 13.0
    f[1] = c0.suit / 4.0
    f[2] = c1.rank / 13.0
    f[3] = c1.suit / 4.0

    # community cards (4-13)
    idx = 4
    for i in range(5):
        if i < len(state.community_cards):
            f[idx] = state.community_cards[i].rank / 13.0
            f[idx + 1] = state.community_cards[i].suit / 4.0
        idx += 2

    # forza mano (14)
    if state.player_hand:
        ranking, _ = eval_hand(state.player_hand)
        f[14] = max(0.0, (HandRanking.NONE - ranking) / 9.0)

    # categorie preflop (15-18)
    r0, r1 = c0.rank, c1.rank
    f[15] = 1.0 if r0 == r1 else 0.0
    f[16] = 1.0 if c0.suit == c1.suit else 0.0
    f[17] = max(r0, r1) / 13.0
    f[18] = min(r0, r1) / 13.0

    # situazione (19-25)
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
