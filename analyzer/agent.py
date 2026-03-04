"""
Analyzer Agent

Orchestrates the Memory → Planner → Executor → Reflector investigation loop
for detailed glitch analysis of windows flagged by the Scanner.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from llm import LLMClient
from logger import get_logger
from .memory import Memory
from .tools import ToolRegistry, VQATool, ObjectTrackingTool, ZoomInTool, SAM3_AVAILABLE
from .planner import Planner
from .reflector import Reflector

_log = get_logger(__name__)


class GlitchAnalyzer:
    """
    Detailed glitch analyzer using an iterative tool-based investigation loop.

    Pipeline per window:
      Planner → Executor → [Advocate → Skeptic → Judge] × N → Final Output
    """

    def __init__(
        self,
        api_key: str = "EMPTY",
        api_base: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        max_iterations: int = 5,
        confidence_threshold: float = 0.80,
        timeout: int = 60,
        verbose: bool = True,
        llm_client: Optional[LLMClient] = None,
        frames_dir: Optional[Path] = None,
        target_fps: float = 4.0,
        gpus: Optional[List[int]] = None,
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose

        self.client = llm_client or LLMClient(
            api_key=api_key,
            api_base=api_base,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        # Load prompts and create subagents
        _dir = Path(__file__).parent
        self.planner = Planner(
            client=self.client,
            prompt=(_dir / "prompt_planner.txt").read_text(),
        )
        self.reflector = Reflector(
            client=self.client,
            advocate_prompt=(_dir / "prompt_advocate.txt").read_text(),
            skeptic_prompt=(_dir / "prompt_skeptic.txt").read_text(),
            judge_prompt=(_dir / "prompt_judge.txt").read_text(),
        )

        # Tool registration
        self.tool_registry = ToolRegistry()

        self.vqa_tool = VQATool(llm_client=self.client)
        self.tool_registry.register(self.vqa_tool)

        self.tracking_tool = ObjectTrackingTool(
            frames_dir=frames_dir,
            fps=target_fps,
            gpus=gpus,
        )
        self.tool_registry.register(self.tracking_tool)

        self.zoom_tool = ZoomInTool(
            frames_dir=frames_dir,
            llm_client=self.client,
        )
        self.tool_registry.register(self.zoom_tool)

        if frames_dir:
            _log.debug(f"ObjectTrackingTool / ZoomInTool frames_dir: {frames_dir}")

        # Per-window state
        self._current_image_path: Optional[Path] = None
        self._current_frame_range: Optional[List[int]] = None
        self._current_visual_cues: str = ""

        _log.debug(
            f"GlitchAnalyzer initialized | model={model} | "
            f"max_iterations={max_iterations} | threshold={confidence_threshold} | "
            f"sam3={'available (lazy load)' if SAM3_AVAILABLE else 'unavailable (not installed)'}"
        )

    # ── Public API ────────────────────────────────────────────────────────

    def analyze_window(
        self,
        scanner_result: Dict,
        window_image_path: Path,
        window_info: Optional[Dict] = None,
        game_context: Optional[str] = None,
        use_adversarial: bool = True,
    ) -> Dict:
        """Perform detailed analysis on a window flagged by the Scanner."""
        self._current_image_path = Path(window_image_path)
        window_id = (window_info or {}).get("window_id", scanner_result.get("window_id", "?"))

        time_nodes = scanner_result.get("time_nodes", [])
        if time_nodes and all(isinstance(f, int) for f in time_nodes):
            self._current_frame_range = [min(time_nodes), max(time_nodes)]
        else:
            self._current_frame_range = None
        self._current_visual_cues = scanner_result.get("visual_cues", "")
        t_window = time.time()

        memory = Memory()
        memory.set_hypothesis(
            hypothesis=scanner_result,
            window_info=window_info,
            game_context=game_context,
        )

        _log.info(
            f"══ Window {window_id} ══ "
            f"Hypothesis: {scanner_result.get('category', 'Unknown')} | "
            f"confidence={scanner_result.get('confidence', 0.0):.2f} | "
            f"{'Adversarial' if use_adversarial else 'Legacy'} mode"
        )
        _log.debug(f"[Window {window_id}] visual_cues: {scanner_result.get('visual_cues', '')}")
        if memory.game_context:
            _log.debug(f"[Window {window_id}] game_context: {memory.game_context}")

        iteration = 0
        final_judge_ruling = None

        while iteration < self.max_iterations:
            iteration += 1
            _log.info(f"[Window {window_id}] ── Iteration {iteration}/{self.max_iterations} ──")

            # ── Step 1: Planner ──
            plan = self.planner.run(memory, iteration, self.tool_registry)
            _log.debug(
                f"[Window {window_id}] [Planner] raw result: "
                f"{json.dumps({k: v for k, v in plan.items() if k != 'question'}, ensure_ascii=False)}"
            )

            if plan.get("tool") == "conclude":
                if iteration == 1:
                    _log.info(f"[Window {window_id}] [Planner] First iteration → forcing VQA verification")
                    visual_cues = scanner_result.get("visual_cues", "the anomaly")
                    plan = {
                        "tool": "vqa",
                        "question": (
                            f"Examine this image carefully. The initial scan detected: "
                            f"'{visual_cues}'. Is this observation accurate? Describe what "
                            f"you see in detail, including the specific visual appearance "
                            f"(color, shape, size, type) of any affected objects or characters."
                        ),
                        "reasoning": "Forced verification of initial hypothesis",
                    }
                else:
                    _log.info(f"[Window {window_id}] [Planner] → conclude | reason: {plan.get('reasoning','')}")
                    break
            else:
                _log.info(
                    f"[Window {window_id}] [Planner] → {plan.get('tool')} | "
                    f"reason: {plan.get('reasoning','')[:80]}"
                )
                _log.debug(f"[Window {window_id}] [Planner] question: {plan.get('question','')}")

            # ── Step 2: Executor ──
            t_exec = time.time()
            tool_result = self._run_executor(plan, memory)
            elapsed_exec = time.time() - t_exec

            if tool_result.get("success"):
                _log.info(
                    f"[Window {window_id}] [Executor/{plan.get('tool')}] success | "
                    f"elapsed={elapsed_exec:.1f}s"
                )
                if plan.get("tool") == "zoom_in":
                    ans = tool_result.get("answer", "")
                    _log.info(
                        f"[Window {window_id}] [ZoomIn] "
                        f"frames={tool_result.get('frame_index')} | "
                        f"region={tool_result.get('region')} | "
                        f"answer: {ans[:120]}"
                    )
                    _log.debug(f"[Window {window_id}] [ZoomIn] full answer:\n{ans}")
                elif plan.get("tool") == "vqa" and "answer" in tool_result:
                    ans = tool_result["answer"]
                    _log.info(f"[Window {window_id}] [VQA] answer (first 120 chars): {ans[:120]}")
                    _log.debug(f"[Window {window_id}] [VQA] full answer:\n{ans}")
                elif plan.get("tool") == "object_tracking":
                    _log.info(
                        f"[Window {window_id}] [Tracking] "
                        f"frames={tool_result.get('num_frames', 0)} | "
                        f"range={tool_result.get('frame_range')} | "
                        f"obj='{tool_result.get('object_description', '')}'"
                    )
                    physics = tool_result.get("physics", {})
                    if physics.get("has_anomaly"):
                        for anomaly in physics.get("anomalies", []):
                            _log.info(
                                f"[Window {window_id}] [Tracking] anomaly: "
                                f"{anomaly.get('description', anomaly.get('type'))}"
                            )
                    else:
                        _log.info(f"[Window {window_id}] [Tracking] no physics anomalies detected")
                    _log.debug(
                        f"[Window {window_id}] [Tracking] physics detail: "
                        f"avg_speed={physics.get('avg_speed_px_s', 0):.1f}px/s | "
                        f"max_speed={physics.get('max_speed_px_s', 0):.1f}px/s"
                    )
            else:
                _log.warning(
                    f"[Window {window_id}] [Executor/{plan.get('tool')}] FAILED: "
                    f"{tool_result.get('error','')}"
                )

            memory.add_tool_call(
                tool_name=plan.get("tool", "unknown"),
                query=plan,
                result=tool_result,
            )

            # ── Step 3: Reflection ──
            if use_adversarial and self.reflector.has_adversarial_mode:
                judge_ruling = self.reflector.run_debate(memory, tool_result, window_id)
                if judge_ruling.final_confidence >= self.confidence_threshold:
                    _log.info(
                        f"[Window {window_id}] Confidence threshold "
                        f"({self.confidence_threshold}) reached"
                    )
                    final_judge_ruling = judge_ruling
                    break
            else:
                reflection = self.reflector.run_legacy(memory, tool_result)
                _log.info(
                    f"[Window {window_id}] [Reflector] "
                    f"confidence={reflection.get('updated_confidence', 0):.2f} | "
                    f"continue={reflection.get('should_continue', True)}"
                )
                _log.debug(f"[Window {window_id}] [Reflector] observation: {reflection.get('observation','')}")
                memory.add_reflection(
                    observation=reflection.get("observation", ""),
                    confidence=reflection.get("updated_confidence", 0.5),
                    should_continue=reflection.get("should_continue", True),
                    has_glitch=reflection.get("has_glitch"),
                    adjustment_suggestion=reflection.get("adjustment_suggestion"),
                )
                if not reflection.get("should_continue", True):
                    break
                if reflection.get("updated_confidence", 0) >= self.confidence_threshold:
                    break

        output = self._generate_final_output(memory, final_judge_ruling, iteration)
        elapsed_window = time.time() - t_window

        _log.info(
            f"[Window {window_id}] ═══ FINAL: ruling={output.get('ruling')} | "
            f"category={output.get('category')} | "
            f"subtype={output.get('subtype','?')} | "
            f"confidence={output.get('confidence', 0):.2f} | "
            f"iterations={iteration} | "
            f"total={elapsed_window:.1f}s ═══"
        )
        if output.get("has_glitch") and output.get("description"):
            _log.debug(f"[Window {window_id}] description: {output['description']}")
        if output.get("supporting_evidence"):
            for ev in output["supporting_evidence"]:
                _log.debug(f"[Window {window_id}]   evidence: {ev}")
        if output.get("rejected_explanations"):
            for ex in output["rejected_explanations"]:
                _log.debug(f"[Window {window_id}]   rejected: {ex}")

        return output

    def analyze_windows_batch(
        self,
        scanner_results: List[Dict],
        windows_dir: Path,
        game_context: Optional[str] = None,
        frames_dir: Optional[Path] = None,
        output_file: Optional[Path] = None,
        save_interval: int = 3,
    ) -> List[Dict]:
        """Analyze all windows flagged as glitches by the Scanner."""
        results = []
        windows_dir = Path(windows_dir)
        glitch_windows = [r for r in scanner_results if r.get("has_glitch", False)]
        total = len(glitch_windows)
        t_batch = time.time()

        _log.info(f"Analyzer: {total} glitch window(s) to analyze")
        if game_context:
            _log.info(f"Game context (RAG): {game_context[:100]}{'...' if len(game_context) > 100 else ''}")

        if frames_dir and Path(frames_dir).exists():
            self.tracking_tool.set_frames_dir(Path(frames_dir))
            self.zoom_tool.set_frames_dir(Path(frames_dir))
            _log.debug(f"ObjectTrackingTool / ZoomInTool frames_dir set to {frames_dir}")
        else:
            if frames_dir:
                _log.warning(f"frames_dir not found: {frames_dir} — object_tracking/zoom_in disabled")

        try:
            for idx, scan_result in enumerate(glitch_windows):
                window_id = scan_result.get("window_id", idx)
                image_path = windows_dir / f"window_{window_id:04d}_stitched.jpg"

                _log.info(f"[{idx + 1}/{total}] Analyzing window {window_id}")

                if not image_path.exists():
                    _log.warning(f"Image not found: {image_path}")
                    continue

                try:
                    result = self.analyze_window(
                        scanner_result=scan_result,
                        window_image_path=image_path,
                        window_info={"window_id": window_id},
                        game_context=game_context,
                    )
                    results.append(result)

                    if output_file and (idx + 1) % save_interval == 0:
                        self._save_results(results, output_file)
                        _log.debug(f"Incremental save → {output_file}")

                except Exception as e:
                    _log.error(f"Error analyzing window {window_id}: {e}", exc_info=True)
                    results.append({
                        "window_id": window_id,
                        "error": str(e),
                        "has_glitch": scan_result.get("has_glitch", False),
                        "category": scan_result.get("category", "Unknown"),
                        "confidence": 0.0,
                    })

        finally:
            # Always close the SAM3 session after the batch, even on error
            self.tracking_tool.close()

        if output_file:
            self._save_results(results, output_file)

        elapsed = time.time() - t_batch
        confirmed = sum(1 for r in results if r.get("has_glitch") and r.get("ruling") == "glitch")
        rejected = sum(1 for r in results if r.get("ruling") == "normal")
        _log.info(
            f"Analyzer complete | {total} analyzed | "
            f"{confirmed} confirmed | {rejected} rejected as normal | "
            f"total={elapsed:.1f}s"
        )
        if output_file:
            _log.info(f"Analysis results saved → {output_file}")

        return results

    # ── Executor ──────────────────────────────────────────────────────────

    def _run_executor(self, plan: Dict, memory: Memory) -> Dict:
        tool_name = plan.get("tool")
        if tool_name == "conclude":
            return {"success": True, "message": "Concluding analysis"}

        tool = self.tool_registry.get(tool_name)
        if tool is None:
            return {"error": f"Unknown tool: {tool_name}", "success": False}

        params: Dict = {}
        if tool_name == "vqa":
            params["question"] = plan.get("question", "Describe what you see.")
            params["image_path"] = str(self._current_image_path)

        elif tool_name == "zoom_in":
            fi = plan.get("frame_index")
            if fi is None:
                if self._current_frame_range:
                    fi = (self._current_frame_range[0] + self._current_frame_range[1]) // 2
                else:
                    fi = 0
                _log.warning(f"zoom_in missing frame_index; defaulting to frame {fi}")
            params["frame_index"] = fi
            params["region"] = plan.get("region", "center")
            params["question"] = plan.get("question", "Describe what you see in detail.")

        elif tool_name == "object_tracking":
            raw_desc = plan.get("object_description", "").strip()
            obj_desc = self.planner.sanitize_object_description(
                raw_desc, memory, self._current_visual_cues
            )
            params["object_description"] = obj_desc
            params["frame_range"] = self._current_frame_range

        try:
            return tool.execute(**params)
        except Exception as e:
            return {"error": str(e), "success": False}

    # ── Final output assembly ─────────────────────────────────────────────

    def _generate_final_output(
        self,
        memory: Memory,
        final_judge_ruling,
        iterations: int,
    ) -> Dict:
        hypothesis = memory.hypothesis or {}
        window_info = memory.window_info or {}
        window_id = window_info.get("window_id", hypothesis.get("window_id"))

        last_judge = final_judge_ruling or memory.get_last_judge_ruling()
        last_reflection = memory.get_last_reflection()

        if last_judge:
            has_glitch = (
                True if last_judge.ruling == "glitch"
                else False if last_judge.ruling == "normal"
                else hypothesis.get("has_glitch", True)
            )
            output = {
                "window_id": window_id,
                "has_glitch": has_glitch,
                "ruling": last_judge.ruling,
                "category": last_judge.category or memory.current_category or hypothesis.get("category", "Unknown"),
                "category_corrected": last_judge.category_corrected,
                "subtype": last_judge.subtype or "Unknown",
                "description": last_judge.description or hypothesis.get("visual_cues", ""),
                "time_nodes": hypothesis.get("frame_range", []),
                "confidence": last_judge.final_confidence,
                "supporting_evidence": last_judge.supporting_evidence or [],
                "rejected_explanations": last_judge.rejected_explanations or [],
                "reasoning": last_judge.reasoning,
                "iterations": iterations,
            }
        elif last_reflection:
            has_glitch = (
                last_reflection.has_glitch
                if last_reflection.has_glitch is not None
                else hypothesis.get("has_glitch", True)
            )
            output = {
                "window_id": window_id,
                "has_glitch": has_glitch,
                "ruling": "glitch" if has_glitch else "normal",
                "category": hypothesis.get("category", "Unknown"),
                "category_corrected": False,
                "subtype": "Unknown",
                "description": hypothesis.get("visual_cues", ""),
                "time_nodes": hypothesis.get("frame_range", []),
                "confidence": last_reflection.confidence,
                "supporting_evidence": [],
                "rejected_explanations": [],
                "reasoning": last_reflection.observation,
                "iterations": iterations,
            }
        else:
            output = {
                "window_id": window_id,
                "has_glitch": hypothesis.get("has_glitch", True),
                "ruling": "glitch" if hypothesis.get("has_glitch", True) else "normal",
                "category": hypothesis.get("category", "Unknown"),
                "category_corrected": False,
                "subtype": "Unknown",
                "description": hypothesis.get("visual_cues", ""),
                "time_nodes": hypothesis.get("frame_range", []),
                "confidence": hypothesis.get("confidence", 0.5),
                "supporting_evidence": [],
                "rejected_explanations": [],
                "reasoning": "",
                "iterations": iterations,
            }

        output["memory"] = memory.to_dict()
        return output

    # ── IO ────────────────────────────────────────────────────────────────

    def _save_results(self, results: List[Dict], output_file: Path) -> None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        save_results = [{k: v for k, v in r.items() if k != "memory"} for r in results]
        with open(output_file, "w") as f:
            json.dump(save_results, f, indent=2, default=str)
