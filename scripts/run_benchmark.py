#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark import run_benchmark


def main():
    parser = argparse.ArgumentParser(description="Run the local RAG benchmark sweep.")
    parser.add_argument("--config", default="configs/benchmark.yaml", help="Path to config YAML")
    args = parser.parse_args()

    results_df, summary_df = run_benchmark(args.config)
    print(f"\nWrote {len(results_df)} per-question rows.")
    print("\nSummary (mean per combination):")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
