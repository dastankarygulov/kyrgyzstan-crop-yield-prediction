"""
run_all.py
==========
Master pipeline — runs all four steps in sequence.
Equivalent to running each step script individually.

Usage:
    python run_all.py              # full pipeline
    python run_all.py --steps 1,3  # only steps 1 and 3
    python run_all.py --steps 2-4  # steps 2 through 4
"""

import sys
import time
import argparse


def parse_steps(raw):
    """Parse '1,3' or '2-4' into a list of ints."""
    steps = set()
    for token in raw.split(","):
        if "-" in token:
            a, b = token.split("-")
            steps.update(range(int(a), int(b) + 1))
        else:
            steps.add(int(token))
    return sorted(steps)


def run_step(n, name, fn):
    print(f"\n{'='*60}")
    print(f"  STEP {n}: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    fn()
    elapsed = time.time() - t0
    print(f"\n  Step {n} finished in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Run Kyrgyzstan Crop Yield Prediction pipeline")
    parser.add_argument("--steps", default="1,2,3,4",
                        help="Steps to run, e.g. '1,2,3,4' or '1-4' or '3,4'")
    args = parser.parse_args()

    requested = parse_steps(args.steps)

    STEPS = {
        1: ("Data Cleaning & Preprocessing",          "step1_data_cleaning.main"),
        2: ("Exploratory Data Analysis",              "step2_eda.main"),
        3: ("ML Pipeline: Training + SHAP",           "step3_ml_pipeline.main"),
        4: ("Early Warning + Forecasting",            "step4_early_warning.main"),
    }

    import importlib
    start_total = time.time()

    for n in requested:
        if n not in STEPS:
            print(f"  WARNING: Step {n} not defined, skipping")
            continue
        label, dotted = STEPS[n]
        module_name, fn_name = dotted.rsplit(".", 1)
        mod = importlib.import_module(module_name)
        fn  = getattr(mod, fn_name)
        run_step(n, label, fn)

    total = time.time() - start_total
    print(f"\n{'='*60}")
    print(f"  All requested steps complete in {total:.1f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
