"""
Reflector subagents for the Analyzer debate loop.

Implements the adversarial debate panel: Advocate (argues for glitch),
Skeptic (argues against), and Judge (arbitrates). Also contains the legacy
single-reflector fallback used when adversarial prompts are unavailable.
"""

import json
import time
from typing import Dict, List

from llm import LLMClient
from logger import get_logger
from .memory import Memory, AdvocateReflection, SkepticReflection, JudgeRuling

_log = get_logger(__name__)


# ── Function-calling schemas ───────────────────────────────────────────────────

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


# ── Reflector class ────────────────────────────────────────────────────────────

class Reflector:
    """
    Adversarial debate panel: Advocate → Skeptic → Judge.

    run_debate() runs the full three-way debate for one tool result.
    run_legacy() is a fallback single-reflector used when adversarial prompts
    are unavailable.
    """

    def __init__(
        self,
        client: LLMClient,
        advocate_prompt: str,
        skeptic_prompt: str,
        judge_prompt: str,
        reflector_prompt: str = "",
    ):
        self.client = client
        self.advocate_prompt = advocate_prompt
        self.skeptic_prompt = skeptic_prompt
        self.judge_prompt = judge_prompt
        self.reflector_prompt = reflector_prompt

    @property
    def has_adversarial_mode(self) -> bool:
        return bool(self.advocate_prompt and self.skeptic_prompt and self.judge_prompt)

    def run_debate(self, memory: Memory, tool_result: Dict, window_id) -> JudgeRuling:
        """Run the full adversarial debate: Advocate → Skeptic → Judge."""
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

    def run_legacy(self, memory: Memory, tool_result: Dict) -> Dict:
        """Legacy single-reflector (fallback when adversarial prompts are missing)."""
        context = memory.get_context_for_reflector(tool_result)
        guidance = ""
        if not tool_result.get("success", True):
            guidance = (
                "\n\n**NOTE**: Tool execution FAILED. "
                "Set should_continue=true and try a different approach."
            )
        return self._call_llm(
            self.reflector_prompt + guidance,
            f"## Analysis Context\n\n{context}\n\nEvaluate this result and provide your reflection.",
            REFLECTOR_FUNCTIONS,
        )

    # ── Private debate methods ─────────────────────────────────────────────

    def _run_advocate(self, memory: Memory, tool_result: Dict) -> AdvocateReflection:
        context = memory.get_context_for_advocate(tool_result)
        result = self._call_llm(
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
        result = self._call_llm(
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
        result = self._call_llm(
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

    def _call_llm(self, system_msg: str, user_msg: str, functions: List[Dict]) -> Dict:
        try:
            result = self.client.chat_with_functions(
                system_msg=system_msg,
                user_msg=user_msg,
                functions=functions,
            )
            _log.debug(f"[LLM] Reflector result: {json.dumps(result, ensure_ascii=False)[:300]}")
            return result
        except Exception as e:
            _log.error(f"[LLM] Reflector call failed: {e}", exc_info=True)
            return {"error": str(e), "tool": "conclude"}
