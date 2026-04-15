"""CLI entry point for the autoresearch optimizer loop."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autoresearch optimizer -- autonomous experiment loop",
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=12,
        help="Maximum number of experiments to run (default: 12)",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Only run the baseline evaluation, then exit",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Anthropic model to use (default: claude-sonnet-4-20250514)",
    )
    args = parser.parse_args()

    if args.baseline_only:
        from .baseline import run_baseline  # type: ignore[import-untyped]
        result = run_baseline()
        print(f"Baseline result: {result}")
    else:
        from .optimizer.agent_loop import run_optimizer
        run_optimizer(
            max_experiments=args.max_experiments,
            model_name=args.model,
        )


if __name__ == "__main__":
    main()
