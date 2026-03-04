"""
Memory module for storing context across Analyzer iterations.

Extends the original plan_adjust Memory with a `game_context` field
that holds the game-type/content description produced by the Scanner.
This context is injected into every prompt as a RAG-style knowledge base,
helping the Planner, Advocate, Skeptic, and Judge understand the game's
normal behavior before reasoning about potential glitches.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    tool_name: str
    query: Dict[str, Any]
    result: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class Reflection:
    """Legacy single-reflector record (kept for compatibility)."""
    observation: str
    confidence: float
    should_continue: bool
    has_glitch: Optional[bool] = None   # Reflector's explicit glitch verdict
    adjustment_suggestion: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class AdvocateReflection:
    supporting_evidence: List[str]
    argument: str
    violated_rules: List[str]
    confidence_for_glitch: float
    affected_object_appearance: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class SkepticReflection:
    alternative_explanations: List[str]
    argument: str
    missing_context: List[str]
    confidence_for_normal: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class JudgeRuling:
    advocate_summary: str
    skeptic_summary: str
    ruling: str           # "glitch" | "normal" | "needs_more_evidence"
    reasoning: str
    category: str
    category_corrected: bool
    correction_reason: Optional[str]
    subtype: Optional[str]
    final_confidence: float
    should_continue: bool
    next_questions: List[str]
    description: Optional[str] = None
    supporting_evidence: Optional[List[str]] = None
    rejected_explanations: Optional[List[str]] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class DebateRound:
    tool_result: Dict[str, Any]
    advocate: AdvocateReflection
    skeptic: SkepticReflection
    judge: JudgeRuling
    timestamp: float = field(default_factory=time.time)


class Memory:
    """
    Memory storage for the Analyzer's iterative investigation loop.

    In addition to the original plan_adjust Memory fields, this class
    stores `game_context` — a concise description of the game type and
    content produced by the Scanner.  It is prepended to every context
    string so that the LLM agents have a stable reference when judging
    whether observed behavior is a glitch or normal game mechanics.
    """

    def __init__(self):
        self.hypothesis: Optional[Dict] = None
        self.window_info: Optional[Dict] = None
        self.game_context: str = ""          # ← RAG knowledge base from Scanner
        self.tool_calls: List[ToolCall] = []
        self.reflections: List[Reflection] = []   # legacy
        self.debate_rounds: List[DebateRound] = []
        self.current_category: Optional[str] = None

    def set_hypothesis(
        self,
        hypothesis: Dict,
        window_info: Optional[Dict] = None,
        game_context: Optional[str] = None,
    ) -> None:
        """
        Set initial hypothesis from scanner result.

        Args:
            hypothesis:   Scanner output dict (has_glitch, category, …).
            window_info:  Optional window metadata.
            game_context: Game-type/content description from the Scanner
                          (used as a RAG knowledge base in all prompts).
        """
        self.hypothesis = hypothesis
        self.window_info = window_info
        self.current_category = hypothesis.get("category", "Unknown")
        # Prefer explicitly passed game_context; fall back to the field
        # embedded in the hypothesis dict (scanner always adds it).
        self.game_context = (
            game_context
            or hypothesis.get("game_context", "")
        )

    def add_tool_call(self, tool_name: str, query: Dict, result: Dict) -> None:
        self.tool_calls.append(ToolCall(tool_name=tool_name, query=query, result=result))

    def add_reflection(
        self,
        observation: str,
        confidence: float,
        should_continue: bool,
        has_glitch: Optional[bool] = None,
        adjustment_suggestion: Optional[str] = None,
    ) -> None:
        self.reflections.append(Reflection(
            observation=observation,
            confidence=confidence,
            should_continue=should_continue,
            has_glitch=has_glitch,
            adjustment_suggestion=adjustment_suggestion,
        ))

    def add_debate_round(
        self,
        tool_result: Dict,
        advocate: AdvocateReflection,
        skeptic: SkepticReflection,
        judge: JudgeRuling,
    ) -> None:
        self.debate_rounds.append(DebateRound(
            tool_result=tool_result,
            advocate=advocate,
            skeptic=skeptic,
            judge=judge,
        ))
        if judge.category_corrected:
            self.current_category = judge.category

    # ── Accessors ─────────────────────────────────────────────────────────

    def get_last_reflection(self) -> Optional[Reflection]:
        return self.reflections[-1] if self.reflections else None

    def get_last_debate(self) -> Optional[DebateRound]:
        return self.debate_rounds[-1] if self.debate_rounds else None

    def get_last_judge_ruling(self) -> Optional[JudgeRuling]:
        last = self.get_last_debate()
        return last.judge if last else None

    def get_iteration_count(self) -> int:
        return len(self.tool_calls)

    # ── Context builders ──────────────────────────────────────────────────

    def _game_context_section(self) -> List[str]:
        """Return formatted game context block (prepended to all contexts)."""
        if not self.game_context:
            return []
        return [
            "## Game Context (Knowledge Base)",
            self.game_context,
            "",
        ]

    def get_context_for_planner(self) -> str:
        lines: List[str] = []

        # Game context first — acts as a knowledge base for the planner
        lines.extend(self._game_context_section())

        if self.hypothesis:
            lines.append("## Initial Hypothesis (from Scanner)")
            lines.append(f"- Has Glitch: {self.hypothesis.get('has_glitch', 'Unknown')}")
            lines.append(f"- Category: {self.hypothesis.get('category', 'Unknown')}")
            lines.append(f"- Visual Cues: {self.hypothesis.get('visual_cues', 'None')}")
            lines.append(f"- Reasoning: {self.hypothesis.get('reasoning', 'None')}")
            lines.append(f"- Initial Confidence: {self.hypothesis.get('confidence', 0.0)}")
            if "frame_range" in self.hypothesis:
                lines.append(f"- Frame Range: {self.hypothesis['frame_range']}")
            lines.append("")

        if self.current_category and self.hypothesis and \
                self.current_category != self.hypothesis.get("category"):
            lines.append(f"## Category Corrected To: {self.current_category}")
            lines.append("")

        if self.window_info:
            lines.append("## Window Information")
            for k, v in self.window_info.items():
                lines.append(f"- {k}: {v}")
            lines.append("")

        if self.debate_rounds:
            lines.append("## Previous Debate Rounds")
            for i, debate in enumerate(self.debate_rounds, 1):
                lines.append(f"\n### Round {i}")
                lines.append(f"Tool Result: {self._summarize_result(debate.tool_result)}")
                lines.append(f"Advocate: {debate.advocate.argument[:100]}...")
                lines.append(f"Skeptic: {debate.skeptic.argument[:100]}...")
                lines.append(
                    f"Judge Ruling: {debate.judge.ruling} "
                    f"(confidence: {debate.judge.final_confidence})"
                )
                if debate.judge.category_corrected:
                    lines.append(
                        f"Category Corrected: {debate.judge.category} "
                        f"({debate.judge.correction_reason})"
                    )
            lines.append("")

            last_judge = self.get_last_judge_ruling()
            if last_judge and last_judge.next_questions:
                lines.append("## Suggested Next Questions (from Judge)")
                for q in last_judge.next_questions:
                    lines.append(f"- {q}")
                lines.append("")
        elif self.tool_calls:
            lines.append("## Previous Tool Calls")
            for i, call in enumerate(self.tool_calls, 1):
                lines.append(f"\n### Iteration {i}: {call.tool_name}")
                lines.append(f"Query: {call.query}")
                lines.append(f"Result Summary: {self._summarize_result(call.result)}")
                if i <= len(self.reflections):
                    ref = self.reflections[i - 1]
                    lines.append(f"Observation: {ref.observation}")
                    lines.append(f"Confidence: {ref.confidence}")
                    if ref.adjustment_suggestion:
                        lines.append(f"Adjustment: {ref.adjustment_suggestion}")
            lines.append("")

        return "\n".join(lines)

    def get_context_for_advocate(self, current_tool_result: Dict) -> str:
        lines: List[str] = []
        lines.extend(self._game_context_section())

        if self.hypothesis:
            lines.append("## Hypothesis to SUPPORT")
            lines.append(f"- Category: {self.current_category}")
            lines.append(f"- Visual Cues: {self.hypothesis.get('visual_cues', 'None')}")
            lines.append("")

        lines.append("## Latest Evidence")
        lines.append(self._format_result(current_tool_result))
        lines.append("")

        if self.debate_rounds:
            lines.append("## Your Previous Arguments")
            for i, debate in enumerate(self.debate_rounds, 1):
                lines.append(f"- Round {i}: {debate.advocate.argument[:80]}...")
            lines.append("")

        return "\n".join(lines)

    def get_context_for_skeptic(self, current_tool_result: Dict) -> str:
        lines: List[str] = []
        lines.extend(self._game_context_section())

        if self.hypothesis:
            lines.append("## Hypothesis to REFUTE")
            lines.append(f"- Category: {self.current_category}")
            lines.append(f"- Visual Cues: {self.hypothesis.get('visual_cues', 'None')}")
            lines.append("")

        lines.append("## Latest Evidence")
        lines.append(self._format_result(current_tool_result))
        lines.append("")

        if self.debate_rounds:
            lines.append("## Your Previous Arguments")
            for i, debate in enumerate(self.debate_rounds, 1):
                lines.append(f"- Round {i}: {debate.skeptic.argument[:80]}...")
            lines.append("")

        return "\n".join(lines)

    def get_context_for_judge(
        self,
        current_tool_result: Dict,
        advocate: AdvocateReflection,
        skeptic: SkepticReflection,
    ) -> str:
        lines: List[str] = []
        lines.extend(self._game_context_section())

        if self.hypothesis:
            lines.append("## Current Hypothesis")
            lines.append(f"- Original Category: {self.hypothesis.get('category', 'Unknown')}")
            lines.append(f"- Current Category: {self.current_category}")
            lines.append(f"- Visual Cues: {self.hypothesis.get('visual_cues', 'None')}")
            lines.append("")

        lines.append(f"## Debate Round: {len(self.debate_rounds) + 1}")
        lines.append("")
        lines.append("## Evidence Being Debated")
        lines.append(self._format_result(current_tool_result))
        lines.append("")

        lines.append("## Advocate's Argument (THIS IS A GLITCH)")
        lines.append(f"Argument: {advocate.argument}")
        lines.append(f"Supporting Evidence: {advocate.supporting_evidence}")
        if advocate.affected_object_appearance:
            lines.append(f"Affected Object Appearance: {advocate.affected_object_appearance}")
        lines.append(f"Violated Rules: {advocate.violated_rules}")
        lines.append(f"Confidence for Glitch: {advocate.confidence_for_glitch}")
        lines.append("")

        lines.append("## Skeptic's Argument (THIS IS NORMAL)")
        lines.append(f"Argument: {skeptic.argument}")
        lines.append(f"Alternative Explanations: {skeptic.alternative_explanations}")
        lines.append(f"Missing Context: {skeptic.missing_context}")
        lines.append(f"Confidence for Normal: {skeptic.confidence_for_normal}")
        lines.append("")

        if self.debate_rounds:
            lines.append("## Previous Rulings")
            for i, debate in enumerate(self.debate_rounds, 1):
                lines.append(
                    f"- Round {i}: {debate.judge.ruling} "
                    f"({debate.judge.reasoning[:60]}...)"
                )
            lines.append("")

        return "\n".join(lines)

    def get_context_for_reflector(self, current_tool_result: Dict) -> str:
        """Legacy single-reflector context."""
        lines: List[str] = []
        lines.extend(self._game_context_section())

        if self.hypothesis:
            lines.append("## Initial Hypothesis")
            lines.append(f"- Category: {self.hypothesis.get('category', 'Unknown')}")
            lines.append(f"- Visual Cues: {self.hypothesis.get('visual_cues', 'None')}")
            lines.append(f"- Initial Confidence: {self.hypothesis.get('confidence', 0.0)}")
            lines.append("")

        lines.append(f"## Current Iteration: {self.get_iteration_count()}")
        lines.append("")
        lines.append("## Latest Tool Result")
        lines.append(self._format_result(current_tool_result))
        lines.append("")

        if self.reflections:
            lines.append("## Previous Observations")
            for i, ref in enumerate(self.reflections, 1):
                lines.append(
                    f"- Iteration {i}: {ref.observation} (confidence: {ref.confidence})"
                )
            lines.append("")

        return "\n".join(lines)

    # ── Serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        return {
            "hypothesis": self.hypothesis,
            "window_info": self.window_info,
            "game_context": self.game_context,
            "current_category": self.current_category,
            "tool_calls": [
                {"tool_name": tc.tool_name, "query": tc.query,
                 "result": tc.result, "timestamp": tc.timestamp}
                for tc in self.tool_calls
            ],
            "reflections": [
                {"observation": r.observation, "confidence": r.confidence,
                 "should_continue": r.should_continue,
                 "adjustment_suggestion": r.adjustment_suggestion,
                 "timestamp": r.timestamp}
                for r in self.reflections
            ],
            "debate_rounds": [
                {
                    "tool_result": dr.tool_result,
                    "advocate": {
                        "supporting_evidence": dr.advocate.supporting_evidence,
                        "argument": dr.advocate.argument,
                        "violated_rules": dr.advocate.violated_rules,
                        "confidence_for_glitch": dr.advocate.confidence_for_glitch,
                        "affected_object_appearance": dr.advocate.affected_object_appearance,
                        "timestamp": dr.advocate.timestamp,
                    },
                    "skeptic": {
                        "alternative_explanations": dr.skeptic.alternative_explanations,
                        "argument": dr.skeptic.argument,
                        "missing_context": dr.skeptic.missing_context,
                        "confidence_for_normal": dr.skeptic.confidence_for_normal,
                        "timestamp": dr.skeptic.timestamp,
                    },
                    "judge": {
                        "advocate_summary": dr.judge.advocate_summary,
                        "skeptic_summary": dr.judge.skeptic_summary,
                        "ruling": dr.judge.ruling,
                        "reasoning": dr.judge.reasoning,
                        "category": dr.judge.category,
                        "category_corrected": dr.judge.category_corrected,
                        "correction_reason": dr.judge.correction_reason,
                        "subtype": dr.judge.subtype,
                        "final_confidence": dr.judge.final_confidence,
                        "should_continue": dr.judge.should_continue,
                        "next_questions": dr.judge.next_questions,
                        "description": dr.judge.description,
                        "supporting_evidence": dr.judge.supporting_evidence,
                        "rejected_explanations": dr.judge.rejected_explanations,
                        "timestamp": dr.judge.timestamp,
                    },
                    "timestamp": dr.timestamp,
                }
                for dr in self.debate_rounds
            ],
        }

    # ── Helpers ───────────────────────────────────────────────────────────

    def _summarize_result(self, result: Dict) -> str:
        if "error" in result:
            return f"Error: {result['error']}"
        parts = []
        if "answer" in result:
            parts.append(f"VQA answer: {result['answer'][:100]}...")
        if "num_objects" in result:
            parts.append(f"{result['num_objects']} objects tracked")
        if "physics_analysis" in result:
            parts.append("physics analyzed")
        return ", ".join(parts) if parts else str(result)[:200]

    def _format_result(self, result: Dict) -> str:
        return json.dumps(result, indent=2, default=str)
