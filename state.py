"""
BugAgent state definition for LangGraph workflow.

BugAgentState flows through all graph nodes, accumulating results
at each stage of the pipeline.
"""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class BugAgentState(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────────────────────
    video_path: str           # Absolute path to the input video
    video_name: str           # Stem name (e.g. "haj831")
    game_name: str            # Game title for the final report
    config: Dict[str, Any]   # Runtime configuration dict (see config.py)

    # ── Stage 1: Preprocess ────────────────────────────────────────────────
    frames_dir: str           # Directory of extracted frames
    windows_dir: str          # Directory of stitched window images
    num_frames: int           # Total extracted frames
    num_windows: int          # Total stitched windows
    windows_metadata: List[Dict]  # Per-window metadata from preprocessor

    # ── Stage 2: Scanner ───────────────────────────────────────────────────
    scan_results: List[Dict]  # Per-window scan results (includes game_context)
    game_context: str         # Aggregated game context (used as RAG knowledge base)

    # ── Stage 3: Analyzer ─────────────────────────────────────────────────
    analysis_results: List[Dict]  # Detailed analysis for flagged windows

    # ── Stage 4: Grounder ─────────────────────────────────────────────────
    grounded_results: Dict    # Consolidated glitch records after temporal grounding

    # ── Stage 5: Summarizer ───────────────────────────────────────────────
    final_report: Dict        # Final output report

    # ── Diagnostics ───────────────────────────────────────────────────────
    errors: List[str]         # Non-fatal errors accumulated during processing
