"""Overnight run: baseline + 10 optimizer experiments.

Usage: python run_overnight.py
"""
import os
import sys
import time
import json
from datetime import datetime

# Load API key from .env file or environment
from pathlib import Path
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1)
            os.environ[k] = v

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("ERROR: Set ANTHROPIC_API_KEY in .env or environment")
    sys.exit(1)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print(f"Started at: {datetime.now()}")
print("=" * 60)

# --- Phase 1: Baseline (2 epochs for CPU speed) ---
print("\n=== PHASE 1: BASELINE EVALUATION ===\n")
t0 = time.time()

from .baseline import run_baseline
baseline_result = run_baseline(epochs=2)

baseline_time = time.time() - t0
print(f"\nBaseline completed in {baseline_time/60:.1f} minutes")
print(f"Baseline avg Sharpe: {baseline_result['avg_sharpe']:.4f}")

# Checkpoint baseline
with open("checkpoint_baseline.json", "w") as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "duration_minutes": round(baseline_time / 60, 1),
        **baseline_result,
    }, f, indent=2, default=str)
print("Baseline checkpoint saved to checkpoint_baseline.json")

# --- Phase 2: Optimizer (10 experiments) ---
print(f"\n{'='*60}")
print("=== PHASE 2: OPTIMIZER (10 experiments) ===")
print(f"{'='*60}\n")

from .optimizer.agent_loop import run_optimizer
t0 = time.time()

try:
    run_optimizer(max_experiments=10, model_name="claude-sonnet-4-20250514")
except Exception as e:
    print(f"\nOptimizer error: {e}")
    import traceback
    traceback.print_exc()

optimizer_time = time.time() - t0
print(f"\nOptimizer completed in {optimizer_time/60:.1f} minutes")

# --- Final report ---
print(f"\n{'='*60}")
print(f"OVERNIGHT RUN COMPLETE")
print(f"{'='*60}")
print(f"  Baseline time:   {baseline_time/60:.1f} min")
print(f"  Optimizer time:  {optimizer_time/60:.1f} min")
print(f"  Total time:      {(baseline_time + optimizer_time)/60:.1f} min")
print(f"  Finished at:     {datetime.now()}")

# Load final state
if os.path.exists("optimizer_state.json"):
    with open("optimizer_state.json") as f:
        state = json.load(f)
    print(f"  Baseline Sharpe: {baseline_result['avg_sharpe']:.4f}")
    print(f"  Best Sharpe:     {state.get('best_sharpe', 'N/A')}")
    print(f"  Experiments run: {len(state.get('experiments', []))}")
    kept = sum(1 for e in state.get("experiments", []) if e.get("kept"))
    print(f"  Kept:            {kept}")
    print(f"  Reverted:        {len(state.get('experiments', [])) - kept}")
