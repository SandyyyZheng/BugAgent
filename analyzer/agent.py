"""
Analyzer Agent (renamed from plan_adjust)

Implements the Memory → Planner → Executor → Reflector loop for detailed
glitch analysis. The Reflector uses an adversarial debate: Advocate argues
FOR a glitch, Skeptic argues AGAINST, and the Judge arbitrates.

Key addition: accepts `game_context` from the Scanner and stores it in
Memory so all agents see the game-type/content description as a knowledge base.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from llm import LLMClient
from logger import get_logger
from .memory import (
    Memory, AdvocateReflection, SkepticReflection, JudgeRuling,
)
from .tools import ToolRegistry, VQATool, ObjectTrackingTool, MathCalculationTool, ZoomInTool, SAM3_AVAILABLE

_log = get_logger(__name__)


# ── Function-calling schemas ───────────────────────────────────────────────────

PLANNER_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "use_tool",
            "description": (
                "Use a tool to gather evidence about the potential glitch. "
                "Iteration 1: always use 'vqa' to establish scene context. "
                "Iteration 2+: choose proactively — "
                "Physics glitches → prefer 'object_tracking' for quantitative position/velocity proof; "
                "Visual/texture glitches → prefer 'zoom_in' on the affected region; "
                "Animation/Game Logic → prefer 'zoom_in' on the character. "
                "Do NOT default back to 'vqa' on iteration 2 unless the other tools are clearly unsuitable."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tool": {
                        "type": "string",
                        "enum": ["vqa", "zoom_in", "object_tracking"],
                        "description": (
                            "'vqa': visual question on the full stitched window image. Use on iteration 1, or when other tools are unsuitable. "
                            "'zoom_in': crop+magnify a specific region, then VQA. PREFERRED for Visual/Animation glitches on iteration 2+. "
                            "'object_tracking': frame-by-frame SAM3 tracking + physics analysis. PREFERRED for Physics glitches (floating, clipping, jittering, teleportation) on iteration 2+."
                        ),
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this tool will help verify the hypothesis",
                    },
                    "question": {
                        "type": "string",
                        "description": (
                            "Visual question to ask. Required for 'vqa' and 'zoom_in'."
                        ),
                    },
                    "frame_index": {
                        "description": (
                            "REQUIRED for zoom_in. "
                            "Single int for one frame (e.g. 12), or list of ints for a "
                            "multi-frame grid (e.g. [10, 12, 14]). "
                            "Pick frames where the glitch is most visible."
                        ),
                    },
                    "region": {
                        "description": (
                            "REQUIRED for zoom_in. Where to zoom: "
                            "spatial name (top_left/top_center/top_right/center_left/center/"
                            "center_right/bottom_left/bottom_center/bottom_right/full) "
                            "or normalized bbox [x1, y1, x2, y2] in [0, 1]."
                        ),
                    },
                    "object_description": {
                        "type": "string",
                        "description": (
                            "REQUIRED when tool='object_tracking'. "
                            "Short, visually distinctive description of the object to track "
                            "(e.g. 'player character', 'red sports car', 'white NPC'). "
                            "2-5 words max. Describe what it looks like, NOT the glitch itself."
                        ),
                    },
                },
                "required": ["tool", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "conclude",
            "description": (
                "Stop investigation and provide final conclusion. "
                "Only use after asking at least one VQA question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Why we have enough evidence to conclude",
                    }
                },
                "required": ["reasoning"],
            },
        },
    },
]

ADVOCATE_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "advocate",
            "description": "Argue that this IS a glitch - find supporting evidence",
            "parameters": {
                "type": "object",
                "properties": {
                    "supporting_evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Evidence supporting the glitch hypothesis (include appearance details).",
                    },
                    "affected_object_appearance": {
                        "type": "string",
                        "description": "Brief visual description of the affected object/character.",
                    },
                    "argument": {
                        "type": "string",
                        "description": "Your argument for why this is a glitch",
                    },
                    "violated_rules": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Physics/visual/logic rules being violated",
                    },
                    "confidence_for_glitch": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Your confidence that this IS a glitch (0.0-1.0)",
                    },
                },
                "required": ["supporting_evidence", "argument", "violated_rules", "confidence_for_glitch"],
            },
        },
    }
]

SKEPTIC_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "skeptic",
            "description": "Argue that this is NOT a glitch - find alternative explanations",
            "parameters": {
                "type": "object",
                "properties": {
                    "alternative_explanations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Possible normal explanations for this behavior",
                    },
                    "argument": {
                        "type": "string",
                        "description": "Your argument for why this is normal game behavior",
                    },
                    "missing_context": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Information needed to rule out normal behavior",
                    },
                    "confidence_for_normal": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Your confidence that this is normal (0.0-1.0)",
                    },
                },
                "required": ["alternative_explanations", "argument", "missing_context", "confidence_for_normal"],
            },
        },
    }
]

JUDGE_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "judge",
            "description": "Make a ruling based on Advocate and Skeptic arguments",
            "parameters": {
                "type": "object",
                "properties": {
                    "advocate_summary": {"type": "string"},
                    "skeptic_summary": {"type": "string"},
                    "ruling": {
                        "type": "string",
                        "enum": ["glitch", "normal", "needs_more_evidence"],
                    },
                    "reasoning": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": ["Visual", "Physics", "Game Logic", "Other"],
                    },
                    "category_corrected": {"type": "boolean"},
                    "correction_reason": {"type": "string"},
                    "subtype": {"type": "string"},
                    "final_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "should_continue": {"type": "boolean"},
                    "next_questions": {"type": "array", "items": {"type": "string"}},
                    "description": {"type": "string"},
                    "supporting_evidence": {"type": "array", "items": {"type": "string"}},
                    "rejected_explanations": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["advocate_summary", "skeptic_summary", "ruling", "reasoning",
                             "category", "category_corrected", "final_confidence", "should_continue"],
            },
        },
    }
]

REFLECTOR_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "reflect",
            "description": "Evaluate the tool result and decide next steps",
            "parameters": {
                "type": "object",
                "properties": {
                    "observation": {"type": "string"},
                    "evidence_strength": {"type": "string", "enum": ["strong", "moderate", "weak"]},
                    "updated_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "should_continue": {"type": "boolean"},
                    "adjustment_suggestion": {"type": "string"},
                    "has_glitch": {"type": "boolean"},
                    "category": {"type": "string", "enum": ["Visual", "Physics", "Game Logic", "Other"]},
                    "subtype": {"type": "string"},
                    "description": {"type": "string"},
                    "supporting_evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["observation", "evidence_strength", "updated_confidence", "should_continue"],
            },
        },
    }
]


# ── Analyzer Agent ─────────────────────────────────────────────────────────────

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

        prompt_path = Path(__file__).parent / "prompt.txt"
        self._load_prompts(prompt_path)

        # ── Tool registration ──────────────────────────────────────────────
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

        # ── Per-window state ───────────────────────────────────────────────
        self._current_image_path: Optional[Path] = None
        self._current_frame_range: Optional[List[int]] = None  # [start, end]
        self._current_visual_cues: str = ""  # scanner's visual_cues, for tracking fallback

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

        # Derive frame range and visual cues for object_tracking
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
            t_plan = time.time()
            plan = self._run_planner(memory, iteration)
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
            if use_adversarial and self.advocate_prompt and self.skeptic_prompt and self.judge_prompt:
                judge_ruling = self._run_adversarial_reflection(memory, tool_result, window_id)
                if judge_ruling.final_confidence >= self.confidence_threshold:
                    _log.info(
                        f"[Window {window_id}] Confidence threshold "
                        f"({self.confidence_threshold}) reached"
                    )
                    final_judge_ruling = judge_ruling
                    break
            else:
                reflection = self._run_reflector(memory, tool_result)
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

        # Set frames_dir on the tracking tool so it can start a SAM3 session
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

    # ── Planner ───────────────────────────────────────────────────────────

    def _run_planner(self, memory: Memory, iteration: int) -> Dict:
        context = memory.get_context_for_planner()
        tools_desc = self.tool_registry.get_tools_description()

        system_msg = f"{self.planner_prompt}\n\n{tools_desc}"
        if iteration == 1:
            system_msg += (
                "\n\n**IMPORTANT**: This is iteration 1. "
                "You MUST use 'vqa' to establish scene context. Do NOT use conclude."
            )
        elif iteration == 2:
            category = memory.current_category or "Unknown"
            if category == "Physics":
                tool_hint = "'object_tracking' (quantitative position/velocity proof)"
            elif category in ("Visual", "Animation", "Game Logic"):
                tool_hint = "'zoom_in' (magnify the affected region for a closer look)"
            else:
                tool_hint = "'zoom_in' or 'object_tracking' depending on the glitch type"
            system_msg += (
                f"\n\n**IMPORTANT**: This is iteration 2. "
                f"The glitch category is '{category}'. "
                f"You MUST use {tool_hint} — do NOT call 'vqa' again unless the other tools are clearly unsuitable."
            )

        user_msg = (
            f"## Current Context\n\n{context}\n\n"
            "Select the most appropriate tool to investigate this potential glitch."
        )

        _log.debug(f"[Planner] Calling LLM (iteration={iteration})")
        return self._call_llm_with_functions(system_msg, user_msg, PLANNER_FUNCTIONS)

    # ── Object-description extraction for SAM3 ────────────────────────────

    def _get_last_vqa_answer(self, memory: Memory) -> str:
        """Return the answer text from the most recent successful VQA/zoom_in call."""
        for tc in reversed(memory.tool_calls):
            if tc.tool_name in ("vqa", "zoom_in") and tc.result.get("success"):
                return tc.result.get("answer", "")
        return ""

    def _sanitize_object_description(self, raw: str, memory: Memory) -> str:
        """
        Return a SAM3-safe object description by asking the LLM to extract one
        from all available context: Planner hint, last VQA answer, scanner cues.
        Falls back to a generic label only if the LLM call fails entirely.
        """
        vqa_answer = self._get_last_vqa_answer(memory)
        visual_cues = self._current_visual_cues.strip()
        category = memory.current_category or "Unknown"

        # Build a context block from whatever is available
        context_parts = []
        if vqa_answer:
            context_parts.append(f"Latest scene observation: {vqa_answer[:400]}")
        if visual_cues:
            context_parts.append(f"Initial scan cues: {visual_cues[:150]}")
        if raw:
            context_parts.append(f"Planner hint (may be imprecise): {raw[:100]}")
        context = "\n".join(context_parts) or "No scene description available."

        prompt = (
            f"{context}\n"
            f"Suspected glitch category: {category}\n\n"
            "Task: From the information above, extract a 2-4 word visual description "
            "of the main object or character involved in the suspected glitch. "
            "Describe only its appearance — NOT the glitch behavior itself. "
            "Output ONLY the description, nothing else.\n"
            "Good examples: 'red sports car'  'player in blue armor'  "
            "'white NPC on left'  'dark motorcycle'  'wooden crate'"
        )
        try:
            raw_out = self.client.chat(
                system_msg=(
                    "You extract concise visual object descriptions (2-4 words) "
                    "from scene text. Output only the description."
                ),
                user_msg=prompt,
                images=[],
            )
            desc = raw_out.strip().split("\n")[0].strip("\"'.,;:")
            desc = " ".join(desc.split()[:5])
            _log.info(f"[ObjDesc] LLM extracted: '{desc}'")
            if desc:
                return desc
        except Exception as e:
            _log.warning(f"[ObjDesc] LLM extraction failed: {e}")

        fallback = (
            "main character"
            if any(w in visual_cues.lower() for w in ("character", "player", "npc"))
            else "main object"
        )
        _log.warning(f"[ObjDesc] using generic fallback: '{fallback}'")
        return fallback

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
                # Default: middle frame of the current window range
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
            obj_desc = self._sanitize_object_description(raw_desc, memory)
            params["object_description"] = obj_desc
            params["frame_range"] = self._current_frame_range

        try:
            return tool.execute(**params)
        except Exception as e:
            return {"error": str(e), "success": False}

    # ── Adversarial Reflection ─────────────────────────────────────────────

    def _run_adversarial_reflection(
        self, memory: Memory, tool_result: Dict, window_id
    ) -> JudgeRuling:
        # Advocate
        _log.debug(f"[Window {window_id}] [Advocate] Building case for glitch...")
        t0 = time.time()
        advocate = self._run_advocate(memory, tool_result)
        _log.info(
            f"[Window {window_id}] [Advocate] confidence_for_glitch={advocate.confidence_for_glitch:.2f} | "
            f"elapsed={time.time()-t0:.1f}s"
        )
        _log.debug(f"[Window {window_id}] [Advocate] argument: {advocate.argument}")
        _log.debug(f"[Window {window_id}] [Advocate] violated_rules: {advocate.violated_rules}")
        _log.debug(f"[Window {window_id}] [Advocate] evidence: {advocate.supporting_evidence}")
        if advocate.affected_object_appearance:
            _log.debug(f"[Window {window_id}] [Advocate] object appearance: {advocate.affected_object_appearance}")

        # Skeptic
        _log.debug(f"[Window {window_id}] [Skeptic] Building case against glitch...")
        t0 = time.time()
        skeptic = self._run_skeptic(memory, tool_result)
        _log.info(
            f"[Window {window_id}] [Skeptic] confidence_for_normal={skeptic.confidence_for_normal:.2f} | "
            f"elapsed={time.time()-t0:.1f}s"
        )
        _log.debug(f"[Window {window_id}] [Skeptic] argument: {skeptic.argument}")
        _log.debug(f"[Window {window_id}] [Skeptic] alternatives: {skeptic.alternative_explanations}")
        _log.debug(f"[Window {window_id}] [Skeptic] missing_context: {skeptic.missing_context}")

        # Judge
        _log.debug(f"[Window {window_id}] [Judge] Arbitrating...")
        t0 = time.time()
        judge = self._run_judge(memory, tool_result, advocate, skeptic)
        _log.info(
            f"[Window {window_id}] [Judge] ruling={judge.ruling} | "
            f"confidence={judge.final_confidence:.2f} | "
            f"continue={judge.should_continue} | "
            f"elapsed={time.time()-t0:.1f}s"
        )
        _log.debug(f"[Window {window_id}] [Judge] reasoning: {judge.reasoning}")
        if judge.category_corrected:
            _log.info(
                f"[Window {window_id}] [Judge] Category corrected → {judge.category} "
                f"(was: {memory.hypothesis.get('category','?')}) | reason: {judge.correction_reason}"
            )
        if judge.subtype:
            _log.debug(f"[Window {window_id}] [Judge] subtype: {judge.subtype}")
        if judge.next_questions:
            _log.debug(f"[Window {window_id}] [Judge] suggested next questions: {judge.next_questions}")

        memory.add_debate_round(tool_result, advocate, skeptic, judge)
        return judge

    def _run_advocate(self, memory: Memory, tool_result: Dict) -> AdvocateReflection:
        context = memory.get_context_for_advocate(tool_result)
        result = self._call_llm_with_functions(
            self.advocate_prompt,
            f"## Evidence to Analyze\n\n{context}\n\nBuild your case for why this IS a glitch.",
            ADVOCATE_FUNCTIONS,
        )
        return AdvocateReflection(
            supporting_evidence=result.get("supporting_evidence", []),
            argument=result.get("argument", ""),
            violated_rules=result.get("violated_rules", []),
            confidence_for_glitch=result.get("confidence_for_glitch", 0.5),
            affected_object_appearance=result.get("affected_object_appearance"),
        )

    def _run_skeptic(self, memory: Memory, tool_result: Dict) -> SkepticReflection:
        context = memory.get_context_for_skeptic(tool_result)
        result = self._call_llm_with_functions(
            self.skeptic_prompt,
            f"## Evidence to Analyze\n\n{context}\n\nBuild your case for why this is NORMAL game behavior.",
            SKEPTIC_FUNCTIONS,
        )
        return SkepticReflection(
            alternative_explanations=result.get("alternative_explanations", []),
            argument=result.get("argument", ""),
            missing_context=result.get("missing_context", []),
            confidence_for_normal=result.get("confidence_for_normal", 0.5),
        )

    def _run_judge(
        self,
        memory: Memory,
        tool_result: Dict,
        advocate: AdvocateReflection,
        skeptic: SkepticReflection,
    ) -> JudgeRuling:
        context = memory.get_context_for_judge(tool_result, advocate, skeptic)
        result = self._call_llm_with_functions(
            self.judge_prompt,
            f"## Debate Context\n\n{context}\n\nMake your ruling based on both arguments.",
            JUDGE_FUNCTIONS,
        )
        return JudgeRuling(
            advocate_summary=result.get("advocate_summary", ""),
            skeptic_summary=result.get("skeptic_summary", ""),
            ruling=result.get("ruling", "needs_more_evidence"),
            reasoning=result.get("reasoning", ""),
            category=result.get("category", memory.current_category or "Unknown"),
            category_corrected=result.get("category_corrected", False),
            correction_reason=result.get("correction_reason"),
            subtype=result.get("subtype"),
            final_confidence=result.get("final_confidence", 0.5),
            should_continue=result.get("should_continue", True),
            next_questions=result.get("next_questions", []),
            description=result.get("description"),
            supporting_evidence=result.get("supporting_evidence"),
            rejected_explanations=result.get("rejected_explanations"),
        )

    def _run_reflector(self, memory: Memory, tool_result: Dict) -> Dict:
        """Legacy single-reflector (used when adversarial prompts are missing)."""
        context = memory.get_context_for_reflector(tool_result)
        guidance = ""
        if not tool_result.get("success", True):
            guidance = (
                "\n\n**NOTE**: Tool execution FAILED. "
                "Set should_continue=true and try a different approach."
            )
        return self._call_llm_with_functions(
            self.reflector_prompt + guidance,
            f"## Analysis Context\n\n{context}\n\nEvaluate this result and provide your reflection.",
            REFLECTOR_FUNCTIONS,
        )

    # ── LLM call ─────────────────────────────────────────────────────────

    def _call_llm_with_functions(
        self,
        system_message: str,
        user_message: str,
        functions: List[Dict],
    ) -> Dict:
        try:
            result = self.client.chat_with_functions(
                system_msg=system_message,
                user_msg=user_message,
                functions=functions,
            )
            _log.debug(f"[LLM] function call result: {json.dumps(result, ensure_ascii=False)[:300]}")
            return result
        except Exception as e:
            _log.error(f"[LLM] API call failed: {e}", exc_info=True)
            return {"error": str(e), "tool": "conclude"}

    # ── Final output assembly ─────────────────────────────────────────────

    def _generate_final_output(
        self,
        memory: Memory,
        final_judge_ruling: Optional[JudgeRuling],
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
            # Prefer the Reflector's explicit verdict; fall back to Scanner's hypothesis.
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

    # ── Prompt loading ────────────────────────────────────────────────────

    def _load_prompts(self, prompt_path: Path) -> None:
        with open(prompt_path, "r") as f:
            content = f.read()

        parts = content.split(
            "################################################################################"
        )

        self.planner_prompt = ""
        self.reflector_prompt = ""
        self.advocate_prompt = ""
        self.skeptic_prompt = ""
        self.judge_prompt = ""

        # Each section is bracketed by two delimiter lines:
        #   ###...
        #   # SECTION TITLE
        #   ###...
        #   [actual prompt content]
        # After splitting by the delimiter, title and content are in consecutive parts.
        # We look for the title in parts[i] and read the content from parts[i+1].
        for i, part in enumerate(parts):
            stripped = part.strip()
            if stripped == "# PLANNER PROMPT" and i + 1 < len(parts):
                self.planner_prompt = parts[i + 1].strip()
            elif stripped == "# ADVOCATE REFLECTOR PROMPT" and i + 1 < len(parts):
                self.advocate_prompt = parts[i + 1].strip()
            elif stripped == "# SKEPTIC REFLECTOR PROMPT" and i + 1 < len(parts):
                self.skeptic_prompt = parts[i + 1].strip()
            elif stripped == "# JUDGE REFLECTOR PROMPT" and i + 1 < len(parts):
                self.judge_prompt = parts[i + 1].strip()
            elif stripped == "# REFLECTOR PROMPT" and i + 1 < len(parts):
                self.reflector_prompt = parts[i + 1].strip()

        _log.debug(
            f"Prompts loaded | planner={'✓' if self.planner_prompt else '✗'} | "
            f"advocate={'✓' if self.advocate_prompt else '✗'} | "
            f"skeptic={'✓' if self.skeptic_prompt else '✗'} | "
            f"judge={'✓' if self.judge_prompt else '✗'}"
        )

    # ── IO ─────────────────────────────────────────────────────────────────

    def _save_results(self, results: List[Dict], output_file: Path) -> None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        save_results = [{k: v for k, v in r.items() if k != "memory"} for r in results]
        with open(output_file, "w") as f:
            json.dump(save_results, f, indent=2, default=str)
