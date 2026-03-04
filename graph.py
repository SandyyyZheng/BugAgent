"""
BugAgent LangGraph workflow.

Pipeline:
  preprocess → scanner → (conditional) → analyzer → grounder → summarizer
                                      ↘ (no glitches found) → summarizer

Each node is a pure function that receives the full BugAgentState and returns
a dict of state updates.  LangGraph merges updates automatically.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Literal

# Ensure the project root is on sys.path so submodule imports (e.g. `from llm
# import LLMClient` inside grounder.py) work regardless of CWD.
_PROJECT_ROOT = Path(__file__).parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from langgraph.graph import StateGraph, END

from state import BugAgentState
from preprocess import VideoPreprocessor
from scanner import GlitchScanner
from analyzer import GlitchAnalyzer
from grounder import TemporalGrounder
from summarizer import Summarizer
from llm import LLMClient
from logger import get_logger, setup_logging

_log = get_logger(__name__)


# ── Helper: build LLMClient from config ───────────────────────────────────────

def _make_client(cfg: Dict[str, Any]) -> LLMClient:
    llm = cfg["llm"]
    return LLMClient(
        api_key=llm["api_key"],
        api_base=llm["api_base"],
        model=llm["model"],
        temperature=llm["temperature"],
        max_tokens=llm["max_tokens"],
        timeout=llm["timeout"],
        max_retries=llm["max_retries"],
    )


# ── Helper: aggregate game_context across scan results ────────────────────────

def _aggregate_game_context(scan_results: List[Dict]) -> str:
    """
    Collect game_context strings from scanner results and return a
    representative description for the whole video.

    Strategy: use the first non-empty game_context (the game type is
    consistent throughout a video, so the first description is sufficient).
    Collect up to 3 unique values and join them only if they differ significantly.
    """
    seen: list = []
    for r in scan_results:
        ctx = r.get("game_context", "").strip()
        if ctx and ctx not in seen:
            seen.append(ctx)
        if len(seen) >= 3:
            break

    if not seen:
        return ""
    # Return the most informative one (longest, up to a reasonable length)
    return max(seen, key=len)


# ── Node 1: Preprocess ────────────────────────────────────────────────────────

def preprocess_node(state: BugAgentState) -> Dict[str, Any]:
    """Extract frames and segment video into stitched windows."""
    cfg = state["config"]
    pp_cfg = cfg["preprocess"]

    _log.info("━━━ Stage 1: Preprocess ━━━")
    t0 = time.time()

    preprocessor = VideoPreprocessor(
        output_path=Path(cfg["output_dir"]),
        target_fps=pp_cfg["target_fps"],
        window_size=pp_cfg["window_size"],
        window_overlap=pp_cfg["window_overlap"],
    )

    result = preprocessor.process_video(Path(state["video_path"]))

    elapsed = time.time() - t0
    _log.info(
        f"Preprocess done | frames={result['num_frames']} | "
        f"windows={result['num_windows']} | elapsed={elapsed:.1f}s"
    )

    return {
        "video_name": result["video_name"],
        "frames_dir": result["frames_dir"],
        "windows_dir": result["windows_dir"],
        "num_frames": result["num_frames"],
        "num_windows": result["num_windows"],
        "windows_metadata": result["windows_metadata"],
    }


# ── Node 2: Scanner ───────────────────────────────────────────────────────────

def scanner_node(state: BugAgentState) -> Dict[str, Any]:
    """Scan all windows for potential glitches (initial screening)."""
    cfg = state["config"]
    sc_cfg = cfg["scanner"]
    verbose = cfg.get("verbose", True)

    _log.info("━━━ Stage 2: Scanner ━━━")
    t0 = time.time()

    scanner = GlitchScanner(
        llm_client=LLMClient(
            api_key=cfg["llm"]["api_key"],
            api_base=cfg["llm"]["api_base"],
            model=cfg["llm"]["model"],
            temperature=sc_cfg["temperature"],
            max_tokens=sc_cfg["max_tokens"],
            timeout=cfg["llm"]["timeout"],
            max_retries=cfg["llm"]["max_retries"],
        ),
        verbose=verbose,
    )

    windows_dir = Path(state["windows_dir"])
    window_paths = sorted(windows_dir.glob("window_*_stitched.jpg"))

    output_file = None
    if cfg.get("save_intermediate"):
        output_file = Path(cfg["output_dir"]) / "intermediate" / state["video_name"] / "scan_results.json"

    scan_results = scanner.scan_windows_batch(window_paths, output_file=output_file)
    game_context = _aggregate_game_context(scan_results)

    if game_context:
        _log.debug(f"Aggregated game_context: {game_context}")

    glitch_count = sum(1 for r in scan_results if r.get("has_glitch"))
    elapsed = time.time() - t0
    _log.info(
        f"Scanner done | windows={len(scan_results)} | glitches={glitch_count} | elapsed={elapsed:.1f}s"
    )

    return {
        "scan_results": scan_results,
        "game_context": game_context,
    }


# ── Conditional edge after scanner ────────────────────────────────────────────

def route_after_scanner(state: BugAgentState) -> Literal["analyzer", "summarizer"]:
    """
    Route to analyzer if any glitches were found; skip directly to summarizer
    (producing a 'no bugs' report) if the video is clean.
    """
    has_any_glitch = any(r.get("has_glitch", False) for r in state.get("scan_results", []))
    route = "analyzer" if has_any_glitch else "summarizer"
    _log.info(f"Routing after scanner → {route}")
    return route


# ── Node 3: Analyzer ──────────────────────────────────────────────────────────

def analyzer_node(state: BugAgentState) -> Dict[str, Any]:
    """Detailed analysis of flagged windows using Memory-Planner-Executor-Reflector."""
    cfg = state["config"]
    an_cfg = cfg["analyzer"]
    verbose = cfg.get("verbose", True)

    _log.info("━━━ Stage 3: Analyzer ━━━")
    t0 = time.time()

    analyzer = GlitchAnalyzer(
        llm_client=LLMClient(
            api_key=cfg["llm"]["api_key"],
            api_base=cfg["llm"]["api_base"],
            model=cfg["llm"]["model"],
            temperature=an_cfg["temperature"],
            max_tokens=an_cfg["max_tokens"],
            timeout=cfg["llm"]["timeout"],
            max_retries=cfg["llm"]["max_retries"],
        ),
        max_iterations=an_cfg["max_iterations"],
        confidence_threshold=an_cfg["confidence_threshold"],
        verbose=verbose,
        frames_dir=Path(state["frames_dir"]),
        target_fps=cfg["preprocess"]["target_fps"],
        gpus=an_cfg.get("sam3_gpus", [1]),
    )

    output_file = None
    if cfg.get("save_intermediate"):
        output_file = (
            Path(cfg["output_dir"]) / "intermediate" / state["video_name"] / "analysis_results.json"
        )

    analysis_results = analyzer.analyze_windows_batch(
        scanner_results=state["scan_results"],
        windows_dir=Path(state["windows_dir"]),
        game_context=state.get("game_context", ""),
        frames_dir=Path(state["frames_dir"]),
        output_file=output_file,
    )

    confirmed = sum(1 for r in analysis_results if r.get("has_glitch"))
    elapsed = time.time() - t0
    _log.info(
        f"Analyzer done | analyzed={len(analysis_results)} | confirmed={confirmed} | elapsed={elapsed:.1f}s"
    )

    return {"analysis_results": analysis_results}


# ── Node 4: Grounder ──────────────────────────────────────────────────────────

def grounder_node(state: BugAgentState) -> Dict[str, Any]:
    """Temporal grounding: cluster adjacent similar glitches and find boundaries."""
    cfg = state["config"]
    gr_cfg = cfg["grounder"]
    verbose = cfg.get("verbose", True)

    _log.info("━━━ Stage 4: Grounder ━━━")
    t0 = time.time()

    grounder = TemporalGrounder(
        llm_client=LLMClient(
            api_key=cfg["llm"]["api_key"],
            api_base=cfg["llm"]["api_base"],
            model=cfg["llm"]["model"],
            temperature=cfg["llm"]["temperature"],
            max_tokens=cfg["llm"]["max_tokens"],
            timeout=cfg["llm"]["timeout"],
            max_retries=gr_cfg["max_retries"],
        ),
        # Must match preprocess.window_size so frame→time conversion is consistent.
        frames_per_window=cfg["preprocess"]["window_size"],
        verbose=verbose,
    )

    analysis_results = state.get("analysis_results", [])

    # Build window image map for bidirectional propagation
    windows_dir = Path(state["windows_dir"])
    window_images: Dict[int, str] = {}
    for img_path in windows_dir.glob("window_*_stitched.jpg"):
        try:
            wid = int(img_path.stem.split("_")[1])
            window_images[wid] = str(img_path)
        except (IndexError, ValueError):
            pass

    total_windows = state.get("num_windows", len(window_images))

    records = grounder.ground(
        plan_adjust_results=analysis_results,
        window_images=window_images if window_images else None,
        total_windows=total_windows,
    )

    grounded_results = {
        "num_glitches": len(records),
        "glitches": [r.to_dict() for r in records],
    }

    elapsed = time.time() - t0
    _log.info(f"Grounder done | glitches={grounded_results['num_glitches']} | elapsed={elapsed:.1f}s")

    if cfg.get("save_intermediate"):
        out = Path(cfg["output_dir"]) / "intermediate" / state["video_name"] / "grounded_results.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(grounded_results, f, indent=2)
        _log.debug(f"Grounded results saved → {out}")

    return {"grounded_results": grounded_results}


# ── Node 5: Summarizer ────────────────────────────────────────────────────────

def summarizer_node(state: BugAgentState) -> Dict[str, Any]:
    """Generate the final report from grounded glitch records."""
    cfg = state["config"]
    sm_cfg = cfg["summarizer"]

    _log.info("━━━ Stage 5: Summarizer ━━━")
    t0 = time.time()

    summarizer = Summarizer(
        llm_client=LLMClient(
            api_key=cfg["llm"]["api_key"],
            api_base=cfg["llm"]["api_base"],
            model=cfg["llm"]["model"],
            temperature=cfg["llm"]["temperature"],
            max_tokens=sm_cfg["max_tokens"],
            timeout=cfg["llm"]["timeout"],
            max_retries=cfg["llm"]["max_retries"],
        ),
        fps=sm_cfg["fps"],
    )

    # When scanner found no glitches (skipped analyzer+grounder), grounded_results
    # may be absent — produce an empty report in that case.
    grounded_results = state.get("grounded_results", {"glitches": []})

    output_file = Path(cfg["output_dir"]) / "results" / f"{state['video_name']}_report.json"

    report = summarizer.summarize_and_save(
        grounded_results=grounded_results,
        output_file=output_file,
        video_name=state.get("video_name", "unknown"),
        game_name=state.get("game_name", "Unknown"),
    )

    elapsed = time.time() - t0
    bugs = report.to_dict().get("bugs", [])
    _log.info(
        f"Summarizer done | bugs={len(bugs)} | no_bugs={report.no_bugs} | elapsed={elapsed:.1f}s"
    )
    _log.info(f"Final report saved → {output_file}")

    return {"final_report": report.to_dict()}


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_graph():
    """
    Build and compile the BugAgent LangGraph workflow.

    Returns a compiled LangGraph app that can be invoked with:
        app.invoke(initial_state)
    """
    workflow = StateGraph(BugAgentState)

    # Register nodes
    workflow.add_node("preprocess", preprocess_node)
    workflow.add_node("scanner", scanner_node)
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("grounder", grounder_node)
    workflow.add_node("summarizer", summarizer_node)

    # Entry point
    workflow.set_entry_point("preprocess")

    # Edges
    workflow.add_edge("preprocess", "scanner")
    workflow.add_conditional_edges(
        "scanner",
        route_after_scanner,
        {
            "analyzer": "analyzer",
            "summarizer": "summarizer",   # fast path: no glitches found
        },
    )
    workflow.add_edge("analyzer", "grounder")
    workflow.add_edge("grounder", "summarizer")
    workflow.add_edge("summarizer", END)

    return workflow.compile()


# ── Convenience function ──────────────────────────────────────────────────────

def run_pipeline(
    video_path: str,
    config_dict: Dict[str, Any],
    game_name: str = "Unknown",
    log_dir: str = "data/logs",
) -> Dict[str, Any]:
    """
    Run the full BugAgent pipeline on a single video.

    Args:
        video_path:   Absolute path to the input video file.
        config_dict:  Config dict from BugAgentConfig.to_dict().
        game_name:    Game title for the final report.
        log_dir:      Directory to write the log file into.

    Returns:
        Final BugAgentState after all stages complete.
    """
    video_name = Path(video_path).stem
    verbose = config_dict.get("verbose", True)

    log_file = setup_logging(
        log_dir=log_dir,
        video_name=video_name,
        verbose=verbose,
    )

    _log.info(f"BugAgent pipeline started | video={video_path} | game={game_name}")
    _log.info(f"Model: {config_dict['llm']['model']} @ {config_dict['llm']['api_base']}")
    _log.debug(f"Full config: {config_dict}")

    app = build_graph()
    t_start = time.time()

    initial_state: BugAgentState = {
        "video_path": video_path,
        "video_name": video_name,
        "game_name": game_name,
        "config": config_dict,
        "errors": [],
    }

    final_state = app.invoke(initial_state)

    total_elapsed = time.time() - t_start
    report = final_state.get("final_report", {})
    _log.info(
        f"Pipeline complete | bugs={len(report.get('bugs', []))} | "
        f"total={total_elapsed:.1f}s | log={log_file}"
    )

    return final_state
