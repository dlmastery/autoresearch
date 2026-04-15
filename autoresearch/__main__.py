"""Allow running autoresearch as ``python -m autoresearch``."""

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="autoresearch",
        description="Autonomous FX return prediction system",
    )
    sub = parser.add_subparsers(dest="command")

    # baseline
    p_base = sub.add_parser("baseline", help="Run walk-forward baseline evaluation")
    p_base.add_argument("--backbone", default="lfm2-350m")
    p_base.add_argument("--epochs", type=int, default=5)
    p_base.add_argument("--seq-len", type=int, default=None,
                        help="Override per-backbone default (60 for LFM, 10 for others)")

    # ablation
    p_abl = sub.add_parser("ablation", help="Run multi-backbone ablation study")
    p_abl.add_argument("--backbones", nargs="+", default=None)
    p_abl.add_argument("--epochs", type=int, default=5)

    # sweep
    p_sw = sub.add_parser("sweep", help="Run Optuna hyperparameter sweep")
    p_sw.add_argument("--backbone", required=True)
    p_sw.add_argument("--n-trials", type=int, default=50)
    p_sw.add_argument("--mode", choices=["quick", "full"], default="quick")

    # optimizer
    p_opt = sub.add_parser("optimizer", help="Run Claude-powered autonomous optimizer")
    p_opt.add_argument("--max-experiments", type=int, default=12)
    p_opt.add_argument("--baseline-only", action="store_true")
    p_opt.add_argument("--model", default="claude-sonnet-4-20250514")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "baseline":
        from .baseline import run_baseline
        result = run_baseline(
            epochs=args.epochs, seq_len=args.seq_len, backbone=args.backbone,
        )
        print(f"\nFinal avg Sharpe: {result['avg_sharpe']:.4f}")

    elif args.command == "ablation":
        from .run_ablation import run_ablation, generate_report
        from .model.backbone import BACKBONE_REGISTRY
        from datetime import datetime
        from pathlib import Path

        backbones = args.backbones or list(BACKBONE_REGISTRY.keys())
        results = run_ablation(backbones, epochs=args.epochs)
        report = generate_report(results)
        report_dir = Path(__file__).resolve().parent / "reports"
        report_dir.mkdir(exist_ok=True)
        report_path = report_dir / f"ablation_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        report_path.write_text(report)
        print(f"\nReport saved to {report_path}")

    elif args.command == "sweep":
        from .run_sweep import main as sweep_main
        sys.argv = [
            "autoresearch-sweep",
            "--backbone", args.backbone,
            "--n-trials", str(args.n_trials),
            "--mode", args.mode,
        ]
        sweep_main()

    elif args.command == "optimizer":
        from .run_optimizer import main as opt_main
        sys.argv = [
            "autoresearch-optimizer",
            "--max-experiments", str(args.max_experiments),
            "--model", args.model,
        ] + (["--baseline-only"] if args.baseline_only else [])
        opt_main()


if __name__ == "__main__":
    main()
