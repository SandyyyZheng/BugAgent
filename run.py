#!/usr/bin/env python3
"""
BugAgent entry point.

Usage:
    python run.py --video path/to/video.mp4 [options]

Examples:
    # Local vLLM server (default)
    python run.py --video data/videos/haj831.mp4

    # OpenAI API
    python run.py --video data/videos/haj831.mp4 \
        --api-key $OPENAI_API_KEY \
        --api-base https://api.openai.com/v1 \
        --model gpt-4o

    # Custom output directory
    python run.py --video data/videos/haj831.mp4 --output-dir /tmp/bugagent_out

    # Adjust analysis sensitivity
    python run.py --video data/videos/haj831.mp4 --max-iterations 3 --confidence 0.8
"""

import argparse
import json
import sys
import time
import requests as _requests
from pathlib import Path

# Ensure project root is on sys.path when running directly
sys.path.insert(0, str(Path(__file__).parent))

from config import BugAgentConfig, LLMConfig, AnalyzerConfig
from graph import run_pipeline

_defaults = BugAgentConfig()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BugAgent: LLM-powered video game glitch detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--game-name", default="Unknown", help="Game title for the report")

    # LLM
    parser.add_argument("--api-key", default=_defaults.llm.api_key, help="LLM API key")
    parser.add_argument("--api-base", default=_defaults.llm.api_base, help="LLM API base URL")
    parser.add_argument("--model", default=_defaults.llm.model, help="Model name")

    # Preprocessing
    parser.add_argument("--fps", type=float, default=_defaults.preprocess.target_fps, help="Frame extraction FPS")
    parser.add_argument("--window-size", type=int, default=_defaults.preprocess.window_size, help="Frames per window")

    # Analyzer
    parser.add_argument("--max-iterations", type=int, default=_defaults.analyzer.max_iterations, help="Max analyzer iterations per window")
    parser.add_argument("--confidence", type=float, default=_defaults.analyzer.confidence_threshold, help="Confidence threshold to conclude")
    parser.add_argument(
        "--sam3-gpus", type=int, nargs="+", default=_defaults.analyzer.sam3_gpus, metavar="GPU",
        help="GPU ID(s) for SAM3 object tracker"
    )

    # Output
    parser.add_argument("--output-dir", default=_defaults.output_dir, help="Base output directory")
    parser.add_argument("--no-intermediate", action="store_true", help="Skip saving intermediate results")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    return parser.parse_args()


def _wait_for_vllm(api_base: str, timeout: int = 300, poll: int = 5) -> None:
    """
    Block until the vLLM server is responsive.

    Polls GET /v1/models every `poll` seconds for up to `timeout` seconds.
    Exits immediately if api_base points to a non-local server.
    """
    if "localhost" not in api_base and "127.0.0.1" not in api_base:
        return  # Remote APIs (OpenAI etc.) are assumed to be up

    health_url = f"{api_base.rstrip('/')}/models"
    deadline = time.time() + timeout
    attempt = 0
    while time.time() < deadline:
        try:
            r = _requests.get(health_url, timeout=5)
            if r.status_code == 200:
                if attempt > 0:
                    print(f"  vLLM ready after {attempt * poll}s")
                return
        except Exception:
            pass
        attempt += 1
        print(f"  Waiting for vLLM ({attempt * poll}s)…", end="\r")
        time.sleep(poll)
    print(f"\nWarning: vLLM not ready after {timeout}s — proceeding anyway")


def main():
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    # Build config
    cfg = BugAgentConfig()
    cfg.llm.api_key = args.api_key
    cfg.llm.api_base = args.api_base
    cfg.llm.model = args.model
    cfg.preprocess.target_fps = args.fps
    cfg.preprocess.window_size = args.window_size
    cfg.analyzer.max_iterations = args.max_iterations
    cfg.analyzer.confidence_threshold = args.confidence
    cfg.analyzer.sam3_gpus = args.sam3_gpus
    cfg.summarizer.fps = args.fps
    cfg.output_dir = args.output_dir
    cfg.save_intermediate = not args.no_intermediate
    cfg.verbose = not args.quiet

    print(f"\nBugAgent")
    print(f"{'=' * 60}")
    print(f"Video:   {video_path}")
    print(f"Game:    {args.game_name}")
    print(f"Model:   {args.model}")
    print(f"API:     {args.api_base}")
    print(f"Output:  {args.output_dir}")
    print(f"Logs:    {args.output_dir}/logs/")
    print(f"SAM3:    GPU(s) {args.sam3_gpus}")
    print(f"{'=' * 60}\n")

    _wait_for_vllm(args.api_base)

    final_state = run_pipeline(
        video_path=str(video_path),
        config_dict=cfg.to_dict(),
        game_name=args.game_name,
        log_dir=f"{args.output_dir}/logs",
    )

    report = final_state.get("final_report", {})
    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"Bugs found: {len(report.get('bugs', []))}")
    if report.get("bugs"):
        for i, bug in enumerate(report["bugs"], 1):
            time_nodes = report["time_nodes"][i - 1] if i <= len(report.get("time_nodes", [])) else []
            print(f"\n  Bug #{i}: {bug[:120]}...")
            print(f"  Time: {time_nodes}")
    else:
        print("  No bugs detected.")
    print(f"{'=' * 60}\n")

    return final_state


if __name__ == "__main__":
    main()
