#!/usr/bin/env python3
"""
Evaluation entry point for BugAgent.

Usage:
    python evaluation/run.py \
        --predictions data/results/batch_report_0.json \
        --groundtruth groundtruth.json \
        --api-base http://localhost:8001/v1 \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --output data/results/eval_0.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.evaluator import Evaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate BugAgent batch report against ground truth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--predictions", required=True,
        help="Path to batch_report JSON file (output of BugAgent batch run)"
    )
    parser.add_argument(
        "--groundtruth", required=True,
        help="Path to ground truth JSON file"
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save detailed evaluation results (optional)"
    )
    parser.add_argument("--api-key", default="EMPTY", help="LLM API key")
    parser.add_argument("--api-base", default="http://localhost:8000/v1", help="LLM API base URL")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Model for scoring")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-video output")
    return parser.parse_args()


def main():
    args = parse_args()

    pred_file = Path(args.predictions)
    gt_file = Path(args.groundtruth)

    if not pred_file.exists():
        print(f"Error: predictions file not found: {pred_file}", file=sys.stderr)
        sys.exit(1)
    if not gt_file.exists():
        print(f"Error: ground truth file not found: {gt_file}", file=sys.stderr)
        sys.exit(1)

    evaluator = Evaluator(
        api_key=args.api_key,
        api_base=args.api_base,
        model=args.model,
        verbose=not args.quiet,
    )

    result = evaluator.evaluate(
        gt_file=gt_file,
        pred_file=pred_file,
        output_file=Path(args.output) if args.output else None,
    )

    evaluator.print_results(result)


if __name__ == "__main__":
    main()
