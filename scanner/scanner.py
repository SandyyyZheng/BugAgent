"""
Scanner module for initial glitch detection and classification.

Renamed from 'categorizer' — performs initial screening of stitched window
images and produces per-window scan results including a game_context field.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from llm import LLMClient
from logger import get_logger

_log = get_logger(__name__)


class GlitchScanner:
    """
    Initial glitch scanner using a multimodal LLM.

    Analyzes stitched window images to detect and categorize potential glitches.
    Also produces a `game_context` field per window used as a RAG knowledge base
    in the Analyzer stage.
    """

    def __init__(
        self,
        api_key: str = "EMPTY",
        api_base: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        temperature: float = 0.3,
        max_tokens: int = 512,
        timeout: int = 60,
        verbose: bool = True,
        llm_client: Optional[LLMClient] = None,
    ):
        self.model = model
        self.verbose = verbose

        self.client = llm_client or LLMClient(
            api_key=api_key,
            api_base=api_base,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        prompt_path = Path(__file__).parent / "prompt.txt"
        with open(prompt_path, "r") as f:
            self.system_prompt = f.read()

        _log.debug(f"GlitchScanner initialized | model={model} | api_base={api_base}")

    # ── Public API ────────────────────────────────────────────────────────

    def scan_window(
        self,
        image_path: Union[str, Path],
        window_id: Optional[int] = None,
    ) -> Dict:
        """
        Scan a single stitched window image for glitches.

        Returns a dict with: window_id, image_path, timestamp, has_glitch,
        category, visual_cues, reasoning, confidence, frame_range (optional),
        game_context, raw_response.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        _log.debug(f"[Window {window_id}] Scanning {image_path.name}")
        t0 = time.time()

        try:
            raw_content = self.client.chat(
                system_msg=self.system_prompt,
                user_msg="Analyze this window for glitches:",
                images=[image_path],
            )
            elapsed = time.time() - t0

            _log.debug(f"[Window {window_id}] LLM response ({elapsed:.1f}s):\n{raw_content}")

            parsed = self._parse_json_response(raw_content)

            # ── Log result summary ──
            if parsed.get("has_glitch"):
                _log.info(
                    f"[Window {window_id}] ✗ GLITCH | "
                    f"category={parsed.get('category','?')} | "
                    f"confidence={parsed.get('confidence', 0):.2f} | "
                    f"elapsed={elapsed:.1f}s"
                )
                _log.debug(f"[Window {window_id}] visual_cues: {parsed.get('visual_cues','')}")
                _log.debug(f"[Window {window_id}] reasoning: {parsed.get('reasoning','')}")
                if "frame_range" in parsed:
                    _log.debug(f"[Window {window_id}] frame_range: {parsed['frame_range']}")
            else:
                _log.info(
                    f"[Window {window_id}] ✓ No glitch | "
                    f"confidence={parsed.get('confidence', 0):.2f} | "
                    f"elapsed={elapsed:.1f}s"
                )
                _log.debug(f"[Window {window_id}] reasoning: {parsed.get('reasoning','')}")

            if parsed.get("game_context"):
                _log.debug(f"[Window {window_id}] game_context: {parsed['game_context']}")

            return {
                "window_id": window_id,
                "image_path": str(image_path),
                "timestamp": time.time(),
                "raw_response": raw_content,
                **parsed,
            }

        except Exception as e:
            _log.error(f"[Window {window_id}] Scan failed: {e}")
            raise RuntimeError(f"Failed to scan window: {e}")

    def scan_windows_batch(
        self,
        window_paths: List[Path],
        output_file: Optional[Path] = None,
        save_interval: int = 5,
    ) -> List[Dict]:
        """
        Scan multiple windows in batch.

        Args:
            window_paths:  Sorted list of stitched window image paths.
            output_file:   Optional path to save results incrementally.
            save_interval: Save every N windows.

        Returns:
            List of scan result dicts (one per window).
        """
        results = []
        total = len(window_paths)
        t_batch = time.time()

        _log.info(f"Scanning {total} windows | model={self.model}")

        for idx, window_path in enumerate(window_paths):
            _log.debug(f"[{idx + 1}/{total}] Processing {window_path.name}")

            try:
                window_id = int(window_path.stem.split("_")[1])
                result = self.scan_window(window_path, window_id=window_id)
                results.append(result)

                if output_file and (idx + 1) % save_interval == 0:
                    self._save_results(results, output_file)
                    _log.debug(f"Incremental save → {output_file} ({len(results)} results)")

            except Exception as e:
                _log.error(f"[{idx + 1}/{total}] Error on {window_path.name}: {e}")
                results.append({
                    "window_id": idx,
                    "image_path": str(window_path),
                    "error": str(e),
                    "has_glitch": False,
                    "confidence": 0.0,
                    "game_context": "",
                })

        if output_file:
            self._save_results(results, output_file)

        # ── Batch summary ──
        glitch_count = sum(1 for r in results if r.get("has_glitch", False))
        elapsed = time.time() - t_batch
        _log.info(
            f"Scan complete | {total} windows | "
            f"{glitch_count} potential glitches | "
            f"{total - glitch_count} clean | "
            f"total={elapsed:.1f}s"
        )
        if output_file:
            _log.info(f"Scan results saved → {output_file}")

        return results

    # ── Private helpers ───────────────────────────────────────────────────

    def _parse_json_response(self, content: str) -> Dict:
        result = LLMClient._parse_json_from_text(content)
        if result is None:
            _log.warning("JSON parse failed — returning default no-glitch result")
            return {
                "has_glitch": False,
                "reasoning": "Failed to parse response",
                "confidence": 0.0,
                "game_context": "",
                "parse_error": "no json found",
                "raw_content": content,
            }

        if "has_glitch" not in result:
            result["has_glitch"] = False
        if "confidence" not in result:
            result["confidence"] = 0.5
        if "game_context" not in result:
            result["game_context"] = ""

        return result

    def _save_results(self, results: List[Dict], output_file: Path) -> None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
