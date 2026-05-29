# entry point: train, evaluate, verify

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pokerl"))

from es_poker.evaluate import validation_suite
from es_poker.training import TrainingProtocol


def main():
    parser = argparse.ArgumentParser(description="ES Self-Play Poker Agent")
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p = sub.add_parser("train", help="esegue il training completo")
    p.add_argument("--gens", type=int, default=25, help="generazioni per sprint")
    p.add_argument("--hands", type=int, default=500, help="mani per valutazione")
    p.add_argument("--pop", type=int, default=100, help="dimensione popolazione")
    p.add_argument(
        "--sigma", type=float, default=0.04, help="deviazione standard rumore"
    )
    p.add_argument("--alpha", type=float, default=0.001, help="learning rate")
    p.add_argument("--parallel", action="store_true", help="usa multiprocessing")
    p.add_argument(
        "-o", "--output", default="./training_output", help="directory output"
    )
    p.add_argument("--seed", type=int, default=42, help="seed random")
    p.add_argument("--sequential", action="store_true",
                   help="usa self-play sequenziale (default: co-evoluzione simultanea)")

    # evaluate
    e = sub.add_parser("evaluate", help="valida un agente salvato")
    e.add_argument("weights", help="percorso file .npz")

    # verify
    sub.add_parser("verify", help="test di verifica (action masking, invarianti)")

    args = parser.parse_args()

    if args.command == "train":
        proto = TrainingProtocol()
        proto.output_dir = args.output
        proto.generations = args.gens
        proto.hands_per_eval = args.hands
        proto.pop_size = args.pop
        proto.sigma = args.sigma
        proto.alpha = args.alpha
        proto.parallel = args.parallel
        proto.seed = args.seed
        proto.simultaneous = not args.sequential
        proto.run()

        # validazione automatica
        print("\n" + "=" * 50)
        print("Esecuzione validazione su agente finale...")
        best = Path(args.output) / "agent_A2.npz"
        if best.exists():
            validation_suite(best, proto._game_config, args.output)

    elif args.command == "evaluate":
        cfg = {
            "num_players": 2,
            "start_credits": 1000,
            "big_blind": 20,
            "small_blind": 10,
        }
        validation_suite(args.weights, cfg, "./training_output")

    elif args.command == "verify":
        from es_poker.evaluate import (
            test_action_masking_all_valid,
            test_action_masking_inf_handling,
            test_action_masking_never_picks_invalid,
            test_action_masking_single_valid,
            test_invariant_no_value_error,
        )

        print("=== Action Masking Tests ===")
        tests = [
            ("tutte valide → argmax invariato", test_action_masking_all_valid),
            ("una sola valida → sempre scelta", test_action_masking_single_valid),
            ("maschere random → mai invalida", test_action_masking_never_picks_invalid),
            ("logits -inf → gestiti correttamente", test_action_masking_inf_handling),
        ]
        for name, fn in tests:
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as e:
                print(f"  FAIL  {name}: {e}")

        print("\n=== Controllo Invarianti (no ValueError) ===")
        cfg = {
            "num_players": 2,
            "start_credits": 1000,
            "big_blind": 20,
            "small_blind": 10,
        }
        try:
            test_invariant_no_value_error(cfg)
            print("  PASS  agente non solleva mai ValueError (50 trial x 500 passi)")
        except AssertionError as e:
            print(f"  FAIL  {e}")


if __name__ == "__main__":
    main()
