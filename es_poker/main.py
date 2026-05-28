"""Entry point: train or evaluate the ES poker agent."""

import argparse
import sys
from pathlib import Path

# Ensure pokerl is importable from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pokerl"))

from es_poker.evaluate import run_validation_suite
from es_poker.training import TrainingProtocol


def main():
    parser = argparse.ArgumentParser(
        description="ES Self-Play Poker Agent (Prof. Tronci protocol)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    p_train = sub.add_parser("train", help="Run the full training curriculum")
    p_train.add_argument("--gens", type=int, default=25, help="Generations per sprint")
    p_train.add_argument("--hands", type=int, default=200, help="Hands per evaluation")
    p_train.add_argument(
        "--pop", type=int, default=40, help="Population size (must be even)"
    )
    p_train.add_argument("--sigma", type=float, default=0.05, help="Noise std dev")
    p_train.add_argument("--alpha", type=float, default=0.02, help="Learning rate")
    p_train.add_argument("--credits", type=int, default=1000, help="Starting credits")
    p_train.add_argument("--bb", type=int, default=20, help="Big blind")
    p_train.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of worker processes (1=sequential)",
    )
    p_train.add_argument(
        "-o", "--output", default="./training_output", help="Output dir"
    )

    # --- evaluate ---
    p_eval = sub.add_parser("evaluate", help="Run validation on a trained agent")
    p_eval.add_argument("weights", help="Path to .npz weights file")
    p_eval.add_argument("--credits", type=int, default=1000, help="Starting credits")
    p_eval.add_argument(
        "-o", "--output", default="./training_output", help="Output dir"
    )

    # --- verify ---
    p_verify = sub.add_parser(
        "verify", help="Run verification tests (action masking, invariants)"
    )
    p_verify.add_argument("--credits", type=int, default=1000, help="Starting credits")

    args = parser.parse_args()

    if args.command == "train":
        game_config = {
            "num_players": 2,
            "start_credits": args.credits,
            "big_blind": args.bb,
            "small_blind": args.bb // 2,
        }
        es_config = {
            "population_size": args.pop,
            "sigma": args.sigma,
            "alpha": args.alpha,
        }
        protocol = TrainingProtocol(
            output_dir=args.output,
            game_config=game_config,
            es_config=es_config,
        )
        protocol.run(
            generations_per_sprint=args.gens,
            hands_per_eval=args.hands,
            parallel=args.parallel,
        )

        # Auto-run validation
        print("\n" + "=" * 50)
        print("Running validation suite on final agent...")
        best_weights = Path(args.output) / "agent_A2.npz"
        if best_weights.exists():
            run_validation_suite(best_weights, game_config, args.output)

    elif args.command == "evaluate":
        game_config = {
            "num_players": 2,
            "start_credits": args.credits,
            "big_blind": 20,
            "small_blind": 10,
        }
        run_validation_suite(args.weights, game_config, args.output)

    elif args.command == "verify":
        from es_poker.evaluate import (
            test_action_masking_all_valid,
            test_action_masking_inf_handling,
            test_action_masking_never_picks_invalid,
            test_action_masking_single_valid,
            test_invariant_no_value_error,
        )

        print("=== Verification: Action Masking Tests ===")
        tests = [
            ("all actions valid → argmax unchanged", test_action_masking_all_valid),
            ("single valid action → always chosen", test_action_masking_single_valid),
            (
                "random masks → never picks invalid",
                test_action_masking_never_picks_invalid,
            ),
            ("-inf logits → handled correctly", test_action_masking_inf_handling),
        ]
        for name, test_fn in tests:
            try:
                test_fn()
                print(f"  PASS  {name}")
            except AssertionError as e:
                print(f"  FAIL  {name}: {e}")

        print("\n=== Verification: Invariant Check (no ValueError) ===")
        game_config = {
            "num_players": 4,
            "start_credits": args.credits,
            "big_blind": 20,
            "small_blind": 10,
        }
        try:
            test_invariant_no_value_error(game_config)
            print("  PASS  Agent never raises ValueError (50 trials x 500 steps)")
        except AssertionError as e:
            print(f"  FAIL  {e}")


if __name__ == "__main__":
    main()
