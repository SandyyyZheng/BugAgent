"""
Summarizer module for generating final glitch detection reports.

Converts grounded glitch records into the final output format with
time nodes in seconds. Uses MLLM to summarize fragmented descriptions
into clean, coherent glitch descriptions.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from llm import LLMClient
from logger import get_logger

_log = get_logger(__name__)


@dataclass
class SummaryReport:
    """Final summary report for a video."""
    video_name: str
    game_name: str
    bugs: List[str]
    time_nodes: List[List[List[int]]]  # [bug_idx][occurrence_idx][start, end] in seconds
    no_bugs: bool
    id: Optional[int] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary matching expected output format."""
        result = {
            "video_name": self.video_name,
            "game_name": self.game_name,
            "bugs": self.bugs,
            "time_nodes": self.time_nodes,
            "no_bugs": self.no_bugs
        }
        if self.id is not None:
            result["id"] = self.id
        return result


class Summarizer:
    """
    Summarizer for converting grounded glitch records to final report.

    Converts frame-based occurrences to time-based (seconds) format.
    Uses MLLM to summarize fragmented descriptions into clean descriptions.
    """

    def __init__(
        self,
        api_key: str = "EMPTY",
        api_base: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
        temperature: float = 0.3,
        max_tokens: int = 512,
        timeout: int = 60,
        max_retries: int = 3,
        fps: float = 4.0,
        llm_client: Optional[LLMClient] = None,
    ):
        self.fps = fps
        self.system_prompt = (Path(__file__).parent / "system_prompt.txt").read_text()
        self.client = llm_client or LLMClient(
            api_key=api_key,
            api_base=api_base,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )

    def _call_llm(self, prompt: str) -> str:
        return self.client.chat(system_msg="", user_msg=prompt)

    def _clean_description(self, description: str) -> str:
        """Clean up a description by removing JSON formatting and code blocks."""
        # Remove markdown code blocks
        description = re.sub(r'```json\s*', '', description)
        description = re.sub(r'```\s*', '', description)

        # Try to extract from JSON if present
        try:
            data = json.loads(description)
            if isinstance(data, dict):
                if "merged_description" in data:
                    description = data["merged_description"]
                elif "description" in data:
                    description = data["description"]
        except json.JSONDecodeError:
            pass

        return description.strip()

    def _summarize_description(
        self,
        original_descriptions: List[str],
        merged_description: str,
        category: str,
        subtype: str,
        time_nodes: List[List[float]]
    ) -> str:
        """Use MLLM to summarize fragmented descriptions into a clean description."""
        # If only one description and it's clean, just clean and return it
        if len(original_descriptions) == 1:
            cleaned = self._clean_description(original_descriptions[0])
            # Remove frame references (various patterns)
            cleaned = re.sub(r'\bin frames?\s*#?\d+[-–]?\d*,?\s*', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'\bframes?\s*#?\d+[-–]?\d*,?\s*', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'\s*\(frames?\s*#?\d+[-–]?\d*\)', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'\bacross frames?\s*#?\d+[-–]?\d*', '', cleaned, flags=re.IGNORECASE)
            # Clean up leading/trailing punctuation and whitespace
            cleaned = re.sub(r'^[\s,.:;]+', '', cleaned)
            cleaned = re.sub(r'[\s,.:;]+$', '', cleaned)
            # Capitalize first letter
            if cleaned:
                cleaned = cleaned[0].upper() + cleaned[1:]
            return cleaned.strip()

        # Format descriptions for the prompt
        descriptions_text = "\n".join([
            f"- {self._clean_description(desc)}"
            for desc in original_descriptions
        ])

        # Format time range
        time_range_text = ", ".join([
            f"{t[0]:.1f}s - {t[1]:.1f}s" for t in time_nodes
        ])

        prompt = self.system_prompt.format(
            descriptions=descriptions_text,
            category=category,
            subtype=subtype,
            time_range=time_range_text
        )

        try:
            summarized = self._call_llm(prompt)
            return self._clean_description(summarized)
        except Exception as e:
            _log.warning(f"Failed to summarize description: {e}")
            return self._clean_description(merged_description)

    def summarize(
        self,
        grounded_results: Dict,
        video_name: str,
        game_name: str = "Unknown",
        video_id: Optional[int] = None
    ) -> SummaryReport:
        """
        Generate summary report from grounded results.

        Args:
            grounded_results: Dict containing grounded glitch records.
                Should have 'glitches' key with list of glitch records.
            video_name: Name of the video file.
            game_name: Name of the game.
            video_id: Optional ID for the video.

        Returns:
            SummaryReport object.
        """
        glitches = grounded_results.get("glitches", [])

        if not glitches:
            _log.info("No glitches found — generating empty report.")
            return SummaryReport(
                video_name=video_name,
                game_name=game_name,
                bugs=[],
                time_nodes=[],
                no_bugs=True,
                id=video_id
            )

        _log.info(f"Summarizing {len(glitches)} glitch(es)")

        bugs = []
        time_nodes = []

        for glitch in glitches:
            glitch_id = glitch.get("glitch_id", "?")
            _log.debug(f"Processing Glitch #{glitch_id}...")

            # Convert frame-based occurrences to seconds first
            occurrences = glitch.get("occurrences", [])
            bug_time_nodes = []

            for occ in occurrences:
                start_frame = occ.get("start_frame", 0)
                end_frame = occ.get("end_frame", 0)
                bug_time_nodes.append([
                    self._frame_to_seconds(start_frame),
                    self._frame_to_seconds(end_frame),
                ])

            time_nodes.append(bug_time_nodes)

            # Summarize descriptions using MLLM
            original_descriptions = glitch.get("original_descriptions", [])
            merged_description = glitch.get("description", "")
            category = glitch.get("category", "Unknown")
            subtype = glitch.get("subtype", "Unknown")

            if not original_descriptions:
                original_descriptions = [merged_description]

            summarized_description = self._summarize_description(
                original_descriptions=original_descriptions,
                merged_description=merged_description,
                category=category,
                subtype=subtype,
                time_nodes=bug_time_nodes
            )

            bugs.append(summarized_description)

            _log.info(
                f"Glitch #{glitch_id} | category={category}/{subtype} | time_nodes={bug_time_nodes}"
            )
            _log.debug(f"Glitch #{glitch_id} description: {summarized_description}")

        report = SummaryReport(
            video_name=video_name,
            game_name=game_name,
            bugs=bugs,
            time_nodes=time_nodes,
            no_bugs=False,
            id=video_id
        )

        _log.info(f"Summary complete: {len(bugs)} bug(s) reported")
        return report

    def _frame_to_seconds(self, frame: int) -> int:
        """Convert frame number to seconds (integer) at self.fps."""
        return int(frame // self.fps)

    def summarize_and_save(
        self,
        grounded_results: Dict,
        output_file: Path,
        video_name: str,
        game_name: str = "Unknown",
        video_id: Optional[int] = None
    ) -> SummaryReport:
        """
        Generate summary and save to file.

        Args:
            grounded_results: Dict containing grounded glitch records.
            output_file: Path to save the summary report.
            video_name: Name of the video file.
            game_name: Name of the game.
            video_id: Optional ID for the video.

        Returns:
            SummaryReport object.
        """
        report = self.summarize(
            grounded_results=grounded_results,
            video_name=video_name,
            game_name=game_name,
            video_id=video_id
        )

        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        _log.info(f"Report saved → {output_file}")
        return report

    @staticmethod
    def load_grounded_results(file_path: Path) -> Dict:
        """Load grounded results from file."""
        with open(file_path, "r") as f:
            return json.load(f)
