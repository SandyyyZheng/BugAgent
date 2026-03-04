"""
Temporal Grounding module for merging adjacent glitches.

Identifies glitches that span multiple windows and merges them into
consolidated glitch records with precise temporal boundaries.

Approach:
1. Cluster detected glitches by similarity (using LLM)
2. Bidirectional propagation: search forward/backward from glitch windows
   with VISUAL VERIFICATION using window images
3. Merge windows and descriptions for each glitch cluster
"""

import json
import requests
import time
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Dict, List, Optional

from llm import LLMClient
from logger import get_logger

_log = get_logger(__name__)


@dataclass
class GlitchRecord:
    """A consolidated glitch record spanning one or more windows."""
    glitch_id: int
    category: str
    subtype: str
    description: str
    window_ids: List[int]
    occurrences: List[Dict]  # List of {start_frame, end_frame} for each continuous segment
    confidence: float
    original_descriptions: List[str] = field(default_factory=list)
    glitch_windows: List[int] = field(default_factory=list)  # Original detected windows
    explored_windows: List[int] = field(default_factory=list)  # All explored windows

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "glitch_id": self.glitch_id,
            "category": self.category,
            "subtype": self.subtype,
            "description": self.description,
            "window_ids": self.window_ids,
            "occurrences": self.occurrences,
            "confidence": self.confidence,
            "num_windows": len(self.window_ids),
            "original_descriptions": self.original_descriptions,
        }


class TemporalGrounder:
    """
    Temporal Grounding for merging adjacent glitches.

    Process:
    1. Cluster detected glitches by similarity (LLM-based)
    2. For each cluster, perform bidirectional propagation:
       - Backward search: visually verify if earlier windows contain same glitch
       - Forward search: visually verify if later windows contain same glitch
    3. Merge windows and generate consolidated descriptions
    """

    def __init__(
        self,
        api_key: str = "EMPTY",
        api_base: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        timeout: int = 60,
        max_retries: int = 3,
        frames_per_window: int = 4,
        verbose: bool = True,
        llm_client: Optional[LLMClient] = None,
    ):
        """
        Initialize the Temporal Grounder.

        Args:
            api_key: API key for the LLM service.
            api_base: Base URL for the API endpoint.
            model: Model name to use (must support vision).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            timeout: Request timeout in seconds.
            max_retries: Maximum retries for API calls.
            frames_per_window: Number of frames per window (default 4).
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
        self.frames_per_window = frames_per_window
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

        # Load prompts
        _dir = Path(__file__).parent
        self.base_prompt = (_dir / "system_prompt.txt").read_text()
        self.prompt_similarity = Template((_dir / "prompt_similarity.txt").read_text())
        self.prompt_visual = Template((_dir / "prompt_visual.txt").read_text())
        self.prompt_merge = Template((_dir / "prompt_merge.txt").read_text())

    def ground(
        self,
        plan_adjust_results: List[Dict],
        window_images: Optional[Dict[int, str]] = None,
        total_windows: Optional[int] = None
    ) -> List[GlitchRecord]:
        """
        Perform temporal grounding on Analyzer results.

        Args:
            plan_adjust_results: List of results from Plan-Adjust detection.
                Each result should have: window_id, has_glitch, category, subtype,
                description, time_nodes, confidence, etc.
            window_images: Dict mapping window_id to image path or base64 string.
                Required for bidirectional propagation with visual verification.
                If not provided, only clustering is performed.
            total_windows: Total number of windows in the video.
                Required if window_images is provided.

        Returns:
            List of consolidated GlitchRecord objects.
        """
        # Filter to only glitches
        glitches = [r for r in plan_adjust_results if r.get("has_glitch", False)]

        if not glitches:
            _log.info("No glitches to ground.")
            return []

        # Sort by window_id
        glitches = sorted(glitches, key=lambda x: x.get("window_id", 0))

        _log.info(f"Grounding {len(glitches)} detected glitches | windows={[g.get('window_id') for g in glitches]}")
        if window_images:
            _log.debug(f"Window images provided: {len(window_images)} windows")

        # Build window timing lookup from results
        window_timings = self._build_window_timings(plan_adjust_results)

        # Step 1: Cluster glitches by similarity
        glitch_clusters = self._cluster_by_similarity(glitches)

        _log.info(f"Clustering complete: {len(glitch_clusters)} cluster(s)")
        for i, cluster in enumerate(glitch_clusters):
            _log.debug(f"  Cluster {i+1}: windows={cluster['glitch_windows']}")

        # Step 2: Bidirectional propagation with visual verification
        if window_images is not None and total_windows is not None:
            _log.info("Starting bidirectional propagation with visual verification...")

            for cluster in glitch_clusters:
                self._bidirectional_propagation(
                    cluster, window_images, total_windows
                )

            _log.info("Bidirectional propagation complete")
            for i, cluster in enumerate(glitch_clusters):
                _log.debug(f"  Cluster {i+1}: final_windows={sorted(cluster['final_windows'])}")

            # Update window timings for propagated windows
            for cluster in glitch_clusters:
                for wid in cluster["final_windows"]:
                    if wid not in window_timings:
                        start = wid * self.frames_per_window
                        end = start + self.frames_per_window - 1
                        window_timings[wid] = {
                            "start_frame": start,
                            "end_frame": end
                        }
        else:
            _log.info("No window images provided — skipping bidirectional propagation.")

        # Step 3: Merge windows and create GlitchRecords
        records = []
        for i, cluster in enumerate(glitch_clusters):
            record = self._create_glitch_record(i + 1, cluster, window_timings)
            records.append(record)

            _log.info(
                f"Glitch #{record.glitch_id} | category={record.category}/{record.subtype} | "
                f"windows={record.window_ids} | confidence={record.confidence:.2f}"
            )
            _log.debug(f"Glitch #{record.glitch_id} occurrences: {record.occurrences}")

        _log.info(f"Grounding complete: {len(records)} glitch record(s)")
        return records

    def _build_window_timings(self, results: List[Dict]) -> Dict[int, Dict]:
        """
        Build window timing lookup from results.

        Frame ranges are computed purely from window IDs using the formula:
            start_frame = window_id * frames_per_window
            end_frame   = start_frame + frames_per_window - 1

        We intentionally ignore the LLM's time_nodes field here: those are
        sub-window frame predictions that use a different (and inconsistent)
        coordinate system compared to the formula used for propagated windows,
        which would cause end < start when the two are mixed.

        Args:
            results: List of window results.

        Returns:
            Dict mapping window_id to {start_frame, end_frame}.
        """
        timings = {}
        for r in results:
            wid = r.get("window_id")
            if wid is None:
                continue
            start = wid * self.frames_per_window
            timings[wid] = {
                "start_frame": start,
                "end_frame": start + self.frames_per_window - 1,
            }
        return timings

    def _cluster_by_similarity(self, glitches: List[Dict]) -> List[Dict]:
        """
        Cluster glitches by similarity using LLM.

        Args:
            glitches: List of detected glitches.

        Returns:
            List of cluster dicts.
        """
        clusters = []

        for glitch in glitches:
            window_idx = glitch.get("window_id", 0)
            description = glitch.get("description", "")

            if not clusters:
                # First glitch, create new cluster
                clusters.append({
                    "description": [description],
                    "glitch_windows": [window_idx],
                    "explored_windows": [window_idx],
                    "final_windows": [window_idx],
                    "category": glitch.get("category", "Unknown"),
                    "subtype": glitch.get("subtype", "Unknown"),
                    "confidences": [glitch.get("confidence", 0.5)],
                })
            else:
                # Check similarity with existing clusters
                added = False
                for cluster in clusters:
                    is_similar = self._judge_similarity(
                        description, cluster["description"]
                    )
                    if is_similar:
                        cluster["description"].append(description)
                        cluster["glitch_windows"].append(window_idx)
                        cluster["explored_windows"].append(window_idx)
                        cluster["final_windows"].append(window_idx)
                        cluster["confidences"].append(glitch.get("confidence", 0.5))
                        added = True
                        break

                if not added:
                    clusters.append({
                        "description": [description],
                        "glitch_windows": [window_idx],
                        "explored_windows": [window_idx],
                        "final_windows": [window_idx],
                        "category": glitch.get("category", "Unknown"),
                        "subtype": glitch.get("subtype", "Unknown"),
                        "confidences": [glitch.get("confidence", 0.5)],
                    })

        return clusters

    def _judge_similarity(self, anomaly_description: str, existing_descriptions: List[str]) -> bool:
        """
        Judge if an anomaly is similar to existing anomalies in a cluster.
        """
        prompt = self.prompt_similarity.substitute(
            anomaly_description=anomaly_description,
            existing_descriptions=json.dumps(existing_descriptions, indent=2),
        )
        try:
            response = self._call_llm(prompt)
            result = self._parse_json(response)
            judgement = result.get("judgement", "no")
            return "yes" in judgement.lower()
        except Exception as e:
            _log.warning(f"Similarity judgment failed: {e}")
            return False

    def _bidirectional_propagation(
        self,
        cluster: Dict,
        window_images: Dict[int, str],
        total_windows: int
    ):
        """
        Perform bidirectional propagation with visual verification.

        Args:
            cluster: Glitch cluster to expand.
            window_images: Dict mapping window_id to image path or base64 string.
            total_windows: Total number of windows.
        """
        # For each originally detected glitch window
        for gw_idx, glitch_window_idx in enumerate(cluster["glitch_windows"]):
            description = cluster["description"][gw_idx]

            _log.debug(f"Propagating from window {glitch_window_idx}...")

            # Backward search
            current_idx = glitch_window_idx - 1
            while current_idx >= 0 and current_idx not in cluster["explored_windows"]:
                cluster["explored_windows"].append(current_idx)

                # Get image for this window
                image_data = window_images.get(current_idx)
                if image_data is None:
                    _log.debug(f"  Backward: Window {current_idx} — no image, stopping")
                    break

                # Visual verification
                is_similar = self._find_similar_anomaly_visual(description, image_data, current_idx)

                _log.debug(f"  Backward: Window {current_idx} — {'MATCH' if is_similar else 'NO MATCH'}")

                if is_similar:
                    cluster["final_windows"].append(current_idx)
                    current_idx -= 1
                else:
                    break

            # Forward search
            current_idx = glitch_window_idx + 1
            while current_idx < total_windows and current_idx not in cluster["explored_windows"]:
                cluster["explored_windows"].append(current_idx)

                # Get image for this window
                image_data = window_images.get(current_idx)
                if image_data is None:
                    _log.debug(f"  Forward: Window {current_idx} — no image, stopping")
                    break

                # Visual verification
                is_similar = self._find_similar_anomaly_visual(description, image_data, current_idx)

                _log.debug(f"  Forward: Window {current_idx} — {'MATCH' if is_similar else 'NO MATCH'}")

                if is_similar:
                    cluster["final_windows"].append(current_idx)
                    current_idx += 1
                else:
                    break

    def _find_similar_anomaly_visual(
        self,
        anomaly_description: str,
        image_data: str,
        window_id: int
    ) -> bool:
        """
        Check if a window contains a similar anomaly using visual verification.

        Args:
            anomaly_description: Description of the anomaly to search for.
            image_data: Image path or base64 encoded image string.
            window_id: Window ID for logging.

        Returns:
            True if similar anomaly found, False otherwise.
        """
        prompt = self.prompt_visual.substitute(anomaly_description=anomaly_description)
        try:
            response = self._call_llm_with_image(prompt, image_data)
            result = self._parse_json(response)
            judgement = result.get("judgement", "no")
            return "yes" in judgement.lower()
        except Exception as e:
            _log.warning(f"Visual verification failed for window {window_id}: {e}")
            return False

    def _create_glitch_record(
        self,
        glitch_id: int,
        cluster: Dict,
        window_timings: Dict[int, Dict]
    ) -> GlitchRecord:
        """
        Create a GlitchRecord from a cluster.
        """
        # Sort final windows
        final_windows = sorted(set(cluster["final_windows"]))

        # Get category and subtype from cluster
        category = cluster.get("category", "Unknown")
        subtype = cluster.get("subtype", "Unknown")

        # Merge descriptions
        if len(cluster["description"]) == 1:
            merged_description = cluster["description"][0]
        else:
            merged_description = self._merge_descriptions(cluster["description"])

        # Calculate occurrences (merge consecutive windows into intervals)
        occurrences = self._merge_windows_to_occurrences(final_windows, window_timings)

        # Calculate confidence (max of all)
        confidences = cluster.get("confidences", [0.5])
        confidence = max(confidences)

        return GlitchRecord(
            glitch_id=glitch_id,
            category=category,
            subtype=subtype,
            description=merged_description,
            window_ids=final_windows,
            occurrences=occurrences,
            confidence=confidence,
            original_descriptions=cluster["description"],
            glitch_windows=cluster["glitch_windows"],
            explored_windows=cluster["explored_windows"],
        )

    def _merge_windows_to_occurrences(
        self,
        final_windows: List[int],
        window_timings: Dict[int, Dict]
    ) -> List[Dict]:
        """
        Merge consecutive windows into frame-based occurrences.
        """
        if not final_windows:
            return []

        final_windows = sorted(final_windows)
        occurrences = []

        # Get timing for first window
        first_wid = final_windows[0]
        if first_wid in window_timings:
            cur_start = window_timings[first_wid]["start_frame"]
            cur_end = window_timings[first_wid]["end_frame"]
        else:
            cur_start = first_wid * self.frames_per_window
            cur_end = cur_start + self.frames_per_window - 1

        for i in range(1, len(final_windows)):
            prev_wid = final_windows[i - 1]
            curr_wid = final_windows[i]

            if curr_wid == prev_wid + 1:
                # Consecutive, extend the range
                if curr_wid in window_timings:
                    cur_end = window_timings[curr_wid]["end_frame"]
                else:
                    cur_end = curr_wid * self.frames_per_window + self.frames_per_window - 1
            else:
                # Not consecutive, finish current interval
                occurrences.append({
                    "start_frame": cur_start,
                    "end_frame": cur_end
                })

                # Start new interval
                if curr_wid in window_timings:
                    cur_start = window_timings[curr_wid]["start_frame"]
                    cur_end = window_timings[curr_wid]["end_frame"]
                else:
                    cur_start = curr_wid * self.frames_per_window
                    cur_end = cur_start + self.frames_per_window - 1

        # Append the last interval
        occurrences.append({
            "start_frame": cur_start,
            "end_frame": cur_end
        })

        return occurrences

    def _merge_descriptions(self, descriptions: List[str]) -> str:
        """
        Merge multiple descriptions into one coherent description.
        """
        prompt = self.prompt_merge.substitute(descriptions=json.dumps(descriptions, indent=2))
        try:
            merged = self._call_llm(prompt)
            return merged.strip()
        except Exception as e:
            _log.warning(f"Description merge failed: {e}")
            return " | ".join(descriptions)

    def _call_llm(self, user_message: str) -> str:
        """Call LLM API (text only)."""
        return self.client.chat(
            system_msg=self.base_prompt,
            user_msg=user_message,
        )

    def _call_llm_with_image(self, user_message: str, image_data: str) -> str:
        """
        Call LLM API with image (vision).

        Args:
            user_message: Text prompt.
            image_data: Either a file path or base64 encoded image string.

        Returns:
            LLM response content.
        """
        # Determine if image_data is a path or base64
        if image_data.startswith("/") or image_data.startswith("./") or Path(image_data).exists():
            # It's a file path, pass directly
            return self.client.chat(
                system_msg=self.base_prompt,
                user_msg=user_message,
                images=[image_data],
            )
        else:
            # It's already base64 — fall back to raw requests for this case
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            content = [
                {"type": "text", "text": user_message},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    },
                },
            ]

            payload = {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "messages": [
                    {"role": "system", "content": self.base_prompt},
                    {"role": "user", "content": content},
                ],
            }

            endpoint = f"{self.api_base}/chat/completions"

            for attempt in range(self.max_retries):
                try:
                    response = requests.post(
                        endpoint,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout,
                    )
                    response.raise_for_status()
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise e
                    _log.warning(f"Vision API call failed (attempt {attempt + 1}): {e}")
                    time.sleep(1)

    def _parse_json(self, content: str) -> Dict:
        """Parse JSON from LLM response (with repair for local models)."""
        result = LLMClient._parse_json_from_text(content)
        if result is not None:
            return result
        raise json.JSONDecodeError("Failed to parse JSON from response", content, 0)

    def ground_and_save(
        self,
        plan_adjust_results: List[Dict],
        output_file: Path,
        window_images: Optional[Dict[int, str]] = None,
        total_windows: Optional[int] = None
    ) -> List[GlitchRecord]:
        """
        Ground glitches and save results to file.

        Args:
            plan_adjust_results: Results from Plan-Adjust detection.
            output_file: Path to save grounded results.
            window_images: Dict mapping window_id to image path or base64 string.
            total_windows: Total number of windows in the video.

        Returns:
            List of GlitchRecord objects.
        """
        records = self.ground(plan_adjust_results, window_images, total_windows)

        # Save to file
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "num_glitches": len(records),
            "glitches": [r.to_dict() for r in records],
            "timestamp": time.time()
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        _log.info(f"Grounded results saved → {output_file}")
        return records
