"""ES-trained poker agent and random baseline."""

import numpy as np
from pokerl.agents.agent import PokerAgent
from pokerl.game import Game

from .features import extract_features
from .network import SmallNN


def masked_argmax(logits: np.ndarray, valid_mask: np.ndarray) -> int:
    """Return the action with the highest logit that is also valid.

    Invalid actions are set to -inf before argmax, guaranteeing that
    the agent NEVER picks a move the environment would reject.
    """
    masked = np.where(valid_mask, logits, -np.inf)
    return int(np.argmax(masked))


class ESAgent(PokerAgent):
    """Poker agent whose policy is a small NN trained by Evolution Strategies.

    Parameters
    ----------
    network : SmallNN
        The neural network that maps state features to action logits.
    start_credits : float
        Used to normalise financial features.
    big_blind : float
        Used to compute stack depth in BB.
    """

    def __init__(self, network: SmallNN, start_credits: float = 1000.0, big_blind: float = 20.0):
        self.network = network
        self.start_credits = start_credits
        self.big_blind = big_blind

    def __call__(self, state: Game.StateView) -> int:
        features = extract_features(state, self.start_credits, self.big_blind)
        logits = self.network.forward(features)
        return masked_argmax(logits, state.valid_actions)


class RandomAgent(PokerAgent):
    """Baseline agent: picks uniformly among valid actions."""

    def __call__(self, state: Game.StateView) -> int:
        vu = state.valid_actions
        return int(np.random.choice(len(vu), p=vu / np.sum(vu)))
