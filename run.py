#!/usr/bin/env python3
"""
BugAgent entry point.

Usage:
    # Single video
    python run.py --video path/to/video.mp4 [options]

    # Batch: all videos in a folder
    python run.py --video-dir path/to/folder/ [options]

Examples:
    # Local vLLM server (default)
    python run.py --video data/videos/video_name.mp4

    # OpenAI API
    python run.py --video data/videos/video_name.mp4 \
        --api-key $OPENAI_API_KEY \
        --api-base https://api.openai.com/v1 \
        --model gpt-4o

    # Batch processing
    python run.py --video-dir data/videos/ --game-name "GTA V"
"""

import argparse
import json
import sys
import time
import requests as _requests
from pathlib import Path

# Ensure project root is on sys.path when running directly
sys.path.insert(0, str(Path(__file__).parent))

from config import BugAgentConfig
from graph import run_pipeline

_defaults = BugAgentConfig()

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BugAgent: LLM-powered video game glitch detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input (mutually exclusive, one required)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video", help="Path to a single input video file")
    input_group.add_argument("--video-dir", help="Path to a folder — process all videos inside")
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


def _build_config(args: argparse.Namespace) -> BugAgentConfig:
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
    return cfg


def _wait_for_vllm(api_base: str, timeout: int = 300, poll: int = 5) -> None:
    """
    Block until the vLLM server is responsive.

    Polls GET /v1/models every `poll` seconds for up to `timeout` seconds.
    Exits immediately if api_base points to a non-local server.
    """
    if "localhost" not in api_base and "127.0.0.1" not in api_base:
        return

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


def _print_report(report: dict) -> None:
    print(f"Bugs found: {len(report.get('bugs', []))}")
    if report.get("bugs"):
        for i, bug in enumerate(report["bugs"], 1):
            time_nodes = report["time_nodes"][i - 1] if i <= len(report.get("time_nodes", [])) else []
            print(f"\n  Bug #{i}: {bug[:120]}...")
            print(f"  Time: {time_nodes}")
    else:
        print("  No bugs detected.")


def run_single(video_path: Path, cfg: BugAgentConfig, game_name: str) -> dict:
    """Run pipeline on one video. Returns the final_report dict."""
    final_state = run_pipeline(
        video_path=str(video_path),
        config_dict=cfg.to_dict(),
        game_name=game_name,
        log_dir=f"{cfg.output_dir}/logs",
    )
    return final_state.get("final_report", {})


def run_batch(video_dir: Path, cfg: BugAgentConfig, game_name: str) -> None:
    """
    Process every video in video_dir and write a consolidated batch report.

    Per-video JSON reports and logs are written as usual.
    A merged batch_report.json is saved to {output_dir}/results/.
    """
    videos = sorted(
        p for p in video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _VIDEO_EXTENSIONS
    )

    if not videos:
        print(f"No video files found in {video_dir}")
        sys.exit(1)

    print(f"\nBugAgent — Batch Mode")
    print(f"{'=' * 60}")
    print(f"Folder:  {video_dir}")
    print(f"Videos:  {len(videos)}")
    print(f"Game:    {game_name}")
    print(f"Model:   {cfg.llm.model}")
    print(f"API:     {cfg.llm.api_base}")
    print(f"Output:  {cfg.output_dir}")
    print(f"{'=' * 60}\n")

    all_reports = []
    failed = []

    for idx, video_path in enumerate(videos, 1):
        print(f"\n[{idx}/{len(videos)}] {video_path.name}")
        print(f"{'-' * 40}")
        try:
            report = run_single(video_path, cfg, game_name)
            all_reports.append(report)
            _print_report(report)
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append({"video": video_path.name, "error": str(e)})

    # Write consolidated batch report
    batch_output = Path(cfg.output_dir) / "results" / "batch_report.json"
    batch_output.parent.mkdir(parents=True, exist_ok=True)
    with open(batch_output, "w") as f:
        json.dump(all_reports, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Batch complete: {len(all_reports)} succeeded, {len(failed)} failed")
    if failed:
        print("Failed videos:")
        for entry in failed:
            print(f"  {entry['video']}: {entry['error']}")
    print(f"Batch report → {batch_output}")
    print(f"{'=' * 60}\n")


def main():
    args = parse_args()
    cfg = _build_config(args)

    _wait_for_vllm(args.api_base)

    if args.video_dir:
        video_dir = Path(args.video_dir)
        if not video_dir.is_dir():
            print(f"Error: Not a directory: {video_dir}", file=sys.stderr)
            sys.exit(1)
        run_batch(video_dir, cfg, args.game_name)
    else:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"Error: Video not found: {video_path}", file=sys.stderr)
            sys.exit(1)

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

        report = run_single(video_path, cfg, args.game_name)

        print(f"\n{'=' * 60}")
        print("Done!")
        _print_report(report)
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
