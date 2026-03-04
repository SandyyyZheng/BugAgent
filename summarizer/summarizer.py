"""
Summarizer module for generating final glitch detection reports.

Converts grounded glitch records into the final output format with
time nodes in seconds. Uses MLLM to summarize fragmented descriptions
into clean, coherent glitch descriptions.
"""

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests

from llm import LLMClient
from logger import get_logger

_log = get_logger(__name__)


# Load prompt template from file
PROMPT_FILE = Path(__file__).parent / "prompt.txt"
SUMMARIZE_PROMPT = PROMPT_FILE.read_text() if PROMPT_FILE.exists() else ""


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
        verbose: bool = True,
        llm_client: Optional[LLMClient] = None,
    ):
        """
        Initialize the Summarizer.

        Args:
            api_key: API key for the LLM service.
            api_base: Base URL for the API.
            model: Model name to use.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries on failure.
            fps: Frames per second used during preprocessing (time-based sampling).
            verbose: Whether to print progress information.
            llm_client: Optional pre-configured LLMClient instance.
        """
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.fps = fps
        self.verbose = verbose

        # Use provided client or create one from individual params
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
        """
        Call LLM API with text prompt.

        Args:
            prompt: Text prompt.

        Returns:
            LLM response text.
        """
        return self.client.chat(system_msg="", user_msg=prompt)

    def _clean_description(self, description: str) -> str:
        """
        Clean up a description by removing JSON formatting and code blocks.

        Args:
            description: Raw description text.

        Returns:
            Cleaned description text.
        """
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

        # Clean up whitespace
        description = description.strip()
        return description

    def _summarize_description(
        self,
        original_descriptions: List[str],
        merged_description: str,
        category: str,
        subtype: str,
        time_nodes: List[List[float]]
    ) -> str:
        """
        Use MLLM to summarize fragmented descriptions into a clean description.

        Args:
            original_descriptions: List of original fragment descriptions.
            merged_description: The merged description from grounder.
            category: Glitch category.
            subtype: Glitch subtype.
            time_nodes: List of [start_sec, end_sec] time ranges.

        Returns:
            Summarized clean description.
        """
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

        prompt = SUMMARIZE_PROMPT.format(
            descriptions=descriptions_text,
            category=category,
            subtype=subtype,
            time_range=time_range_text
        )

        try:
            summarized = self._call_llm(prompt)
            # Clean any remaining formatting
            summarized = self._clean_description(summarized)
            return summarized
        except Exception as e:
            _log.warning(f"Failed to summarize description: {e}")
            # Fallback to cleaned merged description
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

                # Convert frames to seconds
                start_sec = self._frame_to_seconds(start_frame)
                end_sec = self._frame_to_seconds(end_frame)

                bug_time_nodes.append([start_sec, end_sec])

            time_nodes.append(bug_time_nodes)

            # Summarize descriptions using MLLM
            original_descriptions = glitch.get("original_descriptions", [])
            merged_description = glitch.get("description", "")
            category = glitch.get("category", "Unknown")
            subtype = glitch.get("subtype", "Unknown")

            # If no original descriptions, use the merged one
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
        """
        Convert frame number to seconds.

        With time-based sampling at fps=4:
        - frame 0 is at t=0.00s → second 0
        - frame 1 is at t=0.25s → second 0
        - frame 4 is at t=1.00s → second 1
        - frame N is at t=N/fps → second N//fps

        Args:
            frame: Frame number (extracted frame index).

        Returns:
            Time in seconds (integer).
        """
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

        # Save to file
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        _log.info(f"Report saved → {output_file}")
        return report

    @staticmethod
    def load_grounded_results(file_path: Path) -> Dict:
        """
        Load grounded results from file.

        Args:
            file_path: Path to grounded results JSON file.

        Returns:
            Dict containing grounded results.
        """
        with open(file_path, "r") as f:
            return json.load(f)
