"""
Planner subagent for the Analyzer investigation loop.

Selects which tool to use next based on the current memory state, and provides
a helper to extract a clean SAM3-compatible object description from scene context.
"""

import json
from typing import Dict, List, Optional

from llm import LLMClient
from logger import get_logger
from .memory import Memory
from .tools import ToolRegistry

_log = get_logger(__name__)


# ── Function-calling schema ────────────────────────────────────────────────────

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
                        "description": "Visual question to ask. Required for 'vqa' and 'zoom_in'.",
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


# ── Planner class ──────────────────────────────────────────────────────────────

class Planner:
    """
    Decides which investigation tool to call next.

    Reads memory state and returns a tool-call plan (dict with 'tool' key).
    Also provides sanitize_object_description() for cleaning up SAM3 queries.
    """

    def __init__(self, client: LLMClient, prompt: str):
        self.client = client
        self.prompt = prompt

    def run(self, memory: Memory, iteration: int, tool_registry: ToolRegistry) -> Dict:
        """Select the next tool to use for this investigation iteration."""
        context = memory.get_context_for_planner()
        tools_desc = tool_registry.get_tools_description()

        system_msg = f"{self.prompt}\n\n{tools_desc}"
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
        return self._call_llm(system_msg, user_msg)

    def sanitize_object_description(
        self, raw: str, memory: Memory, current_visual_cues: str
    ) -> str:
        """
        Return a SAM3-safe object description by asking the LLM to extract one
        from all available context: Planner hint, last VQA answer, scanner cues.
        Falls back to a generic label only if the LLM call fails entirely.
        """
        vqa_answer = self.get_last_vqa_answer(memory)
        visual_cues = current_visual_cues.strip()
        category = memory.current_category or "Unknown"

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

    @staticmethod
    def get_last_vqa_answer(memory: Memory) -> str:
        """Return the answer text from the most recent successful VQA/zoom_in call."""
        for tc in reversed(memory.tool_calls):
            if tc.tool_name in ("vqa", "zoom_in") and tc.result.get("success"):
                return tc.result.get("answer", "")
        return ""

    def _call_llm(self, system_msg: str, user_msg: str) -> Dict:
        try:
            result = self.client.chat_with_functions(
                system_msg=system_msg,
                user_msg=user_msg,
                functions=PLANNER_FUNCTIONS,
            )
            _log.debug(f"[LLM] Planner result: {json.dumps(result, ensure_ascii=False)[:300]}")
            return result
        except Exception as e:
            _log.error(f"[LLM] Planner call failed: {e}", exc_info=True)
            return {"error": str(e), "tool": "conclude"}
