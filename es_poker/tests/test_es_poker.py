"""Unit tests for action masking, invariant checking, and feature extraction."""

import numpy as np
import pytest

from es_poker.agent import masked_argmax
from es_poker.features import extract_features, FEATURE_DIM
from es_poker.network import SmallNN
from es_poker.evaluate import (
    test_action_masking_all_valid,
    test_action_masking_single_valid,
    test_action_masking_never_picks_invalid,
    test_action_masking_inf_handling,
    test_invariant_no_value_error,
)


class TestActionMasking:
    def test_all_valid_unchanged(self):
        test_action_masking_all_valid()

    def test_single_valid_forced(self):
        test_action_masking_single_valid()

    def test_never_invalid(self):
        test_action_masking_never_picks_invalid()

    def test_inf_handling(self):
        test_action_masking_inf_handling()

    def test_masked_argmax_deterministic(self):
        logits = np.array([0.1, 0.5, 0.9, 0.3, 1.0, -0.2, 0.7], dtype=np.float32)
        valid = np.array([True, True, False, True, True, False, True])
        a1 = masked_argmax(logits, valid)
        a2 = masked_argmax(logits, valid)
        assert a1 == a2

    def test_full_game_action_masking(self):
        from pokerl.game import Game
        from es_poker.agent import ESAgent

        rng = np.random.default_rng(42)
        nn = SmallNN(rng)
        agent = ESAgent(nn, start_credits=1000, big_blind=20)

        game = Game(num_players=4, start_credits=1000)
        game.reset()

        for _ in range(200):
            state = game.active_state
            action = agent(state)
            done, hand_over, _ = game.step(action)
            assert state.valid_actions[action], (
                f"Invalid action {action}, valid={state.valid_actions}"
            )
            if done:
                game.reset()


class TestInvariants:
    def test_no_value_error_with_random_nn(self):
        test_invariant_no_value_error()

    def test_feature_shape(self):
        from pokerl.game import Game

        game = Game(num_players=4, start_credits=1000)
        game.reset()

        for _ in range(20):
            state = game.active_state
            f = extract_features(state, start_credits=1000, big_blind=20)
            assert f.shape == (FEATURE_DIM,), f"Expected ({FEATURE_DIM},), got {f.shape}"
            assert f.dtype == np.float32
            assert np.all(np.isfinite(f)), f"Non-finite values in features: {f}"
            _, valid = game.get_valid_actions()
            action = next(iter(valid))
            done, _, _ = game.step(action)
            if done:
                game.reset()

    def test_network_output_shape(self):
        nn = SmallNN()
        x = np.zeros(FEATURE_DIM, dtype=np.float32)
        out = nn.forward(x)
        assert out.shape == (7,)
        assert out.dtype == np.float32

    def test_weight_serialization_roundtrip(self):
        rng = np.random.default_rng(123)
        nn1 = SmallNN(rng)
        w1 = nn1.get_weights()

        nn2 = SmallNN(rng)
        nn2.set_weights(w1)
        w2 = nn2.get_weights()

        assert np.allclose(w1, w2)

        x = rng.standard_normal(FEATURE_DIM).astype(np.float32)
        assert np.allclose(nn1.forward(x), nn2.forward(x))


class TestFeatures:
    def test_features_in_range(self):
        from pokerl.game import Game

        game = Game(num_players=4, start_credits=1000)
        game.reset()

        for _ in range(30):
            state = game.active_state
            f = extract_features(state, start_credits=1000, big_blind=20)
            assert np.all(f >= 0.0), f"Negative feature: {f}"
            _, valid = game.get_valid_actions()
            action = next(iter(valid))
            done, _, _ = game.step(action)
            if done:
                game.reset()


class TestEvolution:
    def test_train_generation_changes_weights(self):
        from es_poker.evolution import EvolutionStrategies
        from es_poker.agent import RandomAgent

        rng = np.random.default_rng(42)
        es = EvolutionStrategies(population_size=10, sigma=0.1, alpha=0.01, rng=rng)
        nn = SmallNN(rng)

        old_weights = nn.get_weights().copy()

        game_config = {
            "num_players": 4,
            "start_credits": 100,
            "big_blind": 10,
            "small_blind": 5,
        }

        nn, stats = es.train_generation(nn, RandomAgent(), game_config, hands_per_eval=20)

        new_weights = nn.get_weights()
        assert not np.allclose(old_weights, new_weights), "Weights did not change"
        assert "profit_mean" in stats
