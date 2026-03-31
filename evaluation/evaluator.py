"""
Evaluator for glitch detection results.

Compares predicted glitch descriptions and time nodes against ground truth
using LLM-based scoring and temporal IoU metrics.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from scipy.optimize import linear_sum_assignment

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from llm import LLMClient


# Load prompt template
SCORE_PROMPT_TEMPLATE = (Path(__file__).parent / "prompt.txt").read_text()


@dataclass
class EvaluationResult:
    """Evaluation result for a single video or the entire dataset."""
    # Counts
    num_videos: int = 0
    num_gt_bugs: int = 0
    num_pred_bugs: int = 0
    num_matched: int = 0

    # Scores
    matched_scores: List[int] = field(default_factory=list)
    matched_ious: List[float] = field(default_factory=list)

    # Metrics
    mean_score: float = 0.0
    mean_iou: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    # IoU-weighted metrics
    precision_iou: float = 0.0
    recall_iou: float = 0.0
    f1_iou: float = 0.0

    # Per-video details
    video_results: List[Dict] = field(default_factory=list)

    def compute_metrics(self) -> None:
        """Compute aggregate metrics from matched scores and IoUs."""
        if self.matched_scores:
            self.mean_score = np.mean(self.matched_scores)
        if self.matched_ious:
            self.mean_iou = np.mean(self.matched_ious)

        # Precision/Recall based on description quality (max score = 5)
        if self.num_pred_bugs > 0:
            self.precision = sum(self.matched_scores) / (5 * self.num_pred_bugs)
        if self.num_gt_bugs > 0:
            self.recall = sum(self.matched_scores) / (5 * self.num_gt_bugs)
        if self.precision + self.recall > 0:
            self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)

        # IoU-weighted metrics
        score_x_iou = [s * i for s, i in zip(self.matched_scores, self.matched_ious)]
        if self.num_pred_bugs > 0:
            self.precision_iou = sum(score_x_iou) / (5 * self.num_pred_bugs)
        if self.num_gt_bugs > 0:
            self.recall_iou = sum(score_x_iou) / (5 * self.num_gt_bugs)
        if self.precision_iou + self.recall_iou > 0:
            self.f1_iou = 2 * self.precision_iou * self.recall_iou / (self.precision_iou + self.recall_iou)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "num_videos": self.num_videos,
            "num_gt_bugs": self.num_gt_bugs,
            "num_pred_bugs": self.num_pred_bugs,
            "num_matched": self.num_matched,
            "mean_score": self.mean_score,
            "mean_iou": self.mean_iou,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "precision_iou": self.precision_iou,
            "recall_iou": self.recall_iou,
            "f1_iou": self.f1_iou
        }


class Evaluator:
    """
    Evaluator for glitch detection results.

    Uses LLM to score description quality and computes temporal IoU.
    """

    def __init__(
        self,
        api_key: str = "EMPTY",
        api_base: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 512,
        timeout: int = 60,
        max_retries: int = 3,
        verbose: bool = True,
        llm_client: Optional[LLMClient] = None,
    ):
        self.verbose = verbose

        self.client = llm_client or LLMClient(
            api_key=api_key,
            api_base=api_base,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )

    def load_ground_truth(self, gt_file: Path) -> Dict[str, Dict]:
        """
        Load ground truth from JSON file.

        Returns:
            Dict mapping video_name (without extension) to ground truth data.
        """
        with open(gt_file, "r") as f:
            data = json.load(f)

        gt_dict = {}
        for item in data:
            video_name = item["video_name"].replace(".mp4", "")
            gt_dict[video_name] = {
                "id": item.get("id"),
                "game_name": item.get("game_name", "Unknown"),
                "bugs": item.get("bugs", []),
                "time_nodes": item.get("time_nodes", []),
                "no_bugs": item.get("no_bugs", True)
            }

        return gt_dict

    def load_predictions(self, pred_file: Path) -> Dict[str, Dict]:
        """
        Load predictions from a batch_report JSON file.

        Args:
            pred_file: Path to batch_report.json (a JSON array of per-video reports).

        Returns:
            Dict mapping video_name (without extension) to prediction data.
        """
        with open(pred_file, "r") as f:
            data = json.load(f)

        pred_dict = {}
        for item in data:
            video_name = item["video_name"].replace(".mp4", "")
            pred_dict[video_name] = {
                "video_name": item.get("video_name", ""),
                "game_name": item.get("game_name", "Unknown"),
                "bugs": item.get("bugs", []),
                "time_nodes": item.get("time_nodes", []),
                "no_bugs": item.get("no_bugs", True),
            }

        return pred_dict

    def compute_iou(
        self,
        gt_timestamps: List[List[float]],
        pred_timestamps: List[List[float]]
    ) -> float:
        """
        Compute temporal Intersection over Union (IoU).

        Args:
            gt_timestamps: List of [start, end] intervals for ground truth.
            pred_timestamps: List of [start, end] intervals for prediction.

        Returns:
            IoU value between 0 and 1.
        """
        if not gt_timestamps and not pred_timestamps:
            return 1.0  # Both empty, perfect match

        if not gt_timestamps or not pred_timestamps:
            return 0.0  # One is empty, no overlap

        def merge_intervals(intervals: List[List[float]]) -> List[List[float]]:
            """Merge overlapping intervals."""
            if not intervals:
                return []

            sorted_intervals = sorted(intervals, key=lambda x: x[0])
            merged = [list(sorted_intervals[0])]

            for current in sorted_intervals[1:]:
                last = merged[-1]
                if current[0] <= last[1]:
                    merged[-1] = [last[0], max(last[1], current[1])]
                else:
                    merged.append(list(current))

            return merged

        def total_duration(intervals: List[List[float]]) -> float:
            """Calculate total duration of intervals."""
            merged = merge_intervals(intervals)
            return sum(end - start for start, end in merged)

        def intersection_duration(
            gt_segs: List[List[float]],
            pred_segs: List[List[float]]
        ) -> float:
            """Calculate intersection duration."""
            intersections = []

            for gt_start, gt_end in gt_segs:
                for pred_start, pred_end in pred_segs:
                    overlap_start = max(gt_start, pred_start)
                    overlap_end = min(gt_end, pred_end)

                    if overlap_start < overlap_end:
                        intersections.append([overlap_start, overlap_end])

            return total_duration(intersections)

        gt_duration = total_duration(gt_timestamps)
        pred_duration = total_duration(pred_timestamps)
        inter_duration = intersection_duration(gt_timestamps, pred_timestamps)

        union_duration = gt_duration + pred_duration - inter_duration

        if union_duration == 0:
            return 1.0 if inter_duration == 0 else 0.0

        return max(0.0, min(1.0, inter_duration / union_duration))

    def score_description(
        self,
        gt_description: str,
        pred_description: str
    ) -> Tuple[int, str]:
        """
        Score a predicted description against ground truth using LLM.

        Returns:
            Tuple of (score 0-5, reasoning).
        """
        prompt = SCORE_PROMPT_TEMPLATE.format(
            gt_description=gt_description,
            pred_description=pred_description
        )

        try:
            content = self.client.chat(
                system_msg="You must respond with a JSON object containing 'rating' (int 0-5) and 'reasoning' (string).",
                user_msg=prompt,
            )

            result_dict = json.loads(content)
            score = int(result_dict.get("rating", 0))
            reasoning = result_dict.get("reasoning", "")
            return score, reasoning

        except (json.JSONDecodeError, ValueError):
            # Try to extract JSON from text response
            from llm.client import LLMClient as _LC
            result_dict = _LC._parse_json_from_text(content)
            if result_dict:
                score = int(result_dict.get("rating", 0))
                reasoning = result_dict.get("reasoning", "")
                return score, reasoning
            if self.verbose:
                print(f"    Warning: Could not parse scoring response")
            return 0, "Parse error"
        except Exception as e:
            if self.verbose:
                print(f"    Warning: Scoring failed: {e}")
            return 0, f"Error: {e}"

    def evaluate_video(
        self,
        gt_data: Dict,
        pred_data: Dict,
        video_name: str
    ) -> Dict:
        """
        Evaluate predictions for a single video.

        Returns:
            Evaluation result for the video.
        """
        gt_bugs = gt_data.get("bugs", [])
        pred_bugs = pred_data.get("bugs", [])
        gt_time_nodes = gt_data.get("time_nodes", [])
        pred_time_nodes = pred_data.get("time_nodes", [])

        gt_num = len(gt_bugs)
        pred_num = len(pred_bugs)

        result = {
            "video_name": video_name,
            "gt_num": gt_num,
            "pred_num": pred_num,
            "matched_scores": [],
            "matched_ious": [],
            "matches": []
        }

        # Handle edge cases
        if gt_num == 0 and pred_num == 0:
            return result
        elif gt_num == 0 or pred_num == 0:
            return result

        # Build score matrix using LLM
        if self.verbose:
            print(f"  Computing score matrix ({pred_num} x {gt_num})...")

        score_matrix = np.zeros((pred_num, gt_num))

        for i, pred_desc in enumerate(pred_bugs):
            for j, gt_desc in enumerate(gt_bugs):
                score, reasoning = self.score_description(gt_desc, pred_desc)
                score_matrix[i][j] = score

                if self.verbose and score >= 3:
                    print(f"    pred[{i}] vs gt[{j}]: score={score}")

        # Hungarian algorithm for optimal matching
        pred_indices, gt_indices = linear_sum_assignment(-score_matrix)

        # Extract matched pairs
        for pred_idx, gt_idx in zip(pred_indices, gt_indices):
            score = int(score_matrix[pred_idx][gt_idx])

            # Compute IoU for matched pair
            gt_times = gt_time_nodes[gt_idx] if gt_idx < len(gt_time_nodes) else []
            pred_times = pred_time_nodes[pred_idx] if pred_idx < len(pred_time_nodes) else []
            iou = self.compute_iou(gt_times, pred_times)

            result["matched_scores"].append(score)
            result["matched_ious"].append(iou)
            result["matches"].append({
                "pred_idx": pred_idx,
                "gt_idx": gt_idx,
                "score": score,
                "iou": iou,
                "pred_desc": pred_bugs[pred_idx][:100] + "..." if len(pred_bugs[pred_idx]) > 100 else pred_bugs[pred_idx],
                "gt_desc": gt_bugs[gt_idx][:100] + "..." if len(gt_bugs[gt_idx]) > 100 else gt_bugs[gt_idx]
            })

        result["score_matrix"] = score_matrix.tolist()

        return result

    def evaluate(
        self,
        gt_file: Path,
        pred_file: Path,
        output_file: Optional[Path] = None
    ) -> EvaluationResult:
        """
        Evaluate all predictions against ground truth.

        Args:
            gt_file:     Path to ground truth JSON file.
            pred_file:   Path to batch_report JSON file.
            output_file: Optional path to save detailed results.

        Returns:
            EvaluationResult with aggregate metrics.
        """
        gt_dict = self.load_ground_truth(gt_file)
        pred_dict = self.load_predictions(pred_file)

        if self.verbose:
            print(f"Loaded {len(gt_dict)} ground truth entries")
            print(f"Loaded {len(pred_dict)} prediction entries")

        common_videos = set(gt_dict.keys()) & set(pred_dict.keys())

        if self.verbose:
            print(f"Found {len(common_videos)} matching videos")

        result = EvaluationResult()
        result.num_videos = len(common_videos)

        for video_name in sorted(common_videos):
            if self.verbose:
                print(f"\nEvaluating: {video_name}")

            gt_data = gt_dict[video_name]
            pred_data = pred_dict[video_name]

            video_result = self.evaluate_video(gt_data, pred_data, video_name)

            result.num_gt_bugs += video_result["gt_num"]
            result.num_pred_bugs += video_result["pred_num"]
            result.num_matched += len(video_result["matched_scores"])

            result.matched_scores.extend(video_result["matched_scores"])
            result.matched_ious.extend(video_result["matched_ious"])

            result.video_results.append(video_result)

        result.compute_metrics()

        if output_file:
            self._save_results(result, output_file)

        return result

    def _save_results(self, result: EvaluationResult, output_file: Path) -> None:
        """Save evaluation results to file."""
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "summary": result.to_dict(),
            "video_results": result.video_results
        }

        def convert_numpy(obj):
            """Convert numpy types to Python native types for JSON serialization."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        output_data = convert_numpy(output_data)

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        if self.verbose:
            print(f"\nResults saved to: {output_file}")

    def print_results(self, result: EvaluationResult) -> None:
        """Print evaluation results in a formatted way."""
        print(f"\n{'='*60}")
        print("Evaluation Results")
        print(f"{'='*60}")
        print(f"Videos evaluated: {result.num_videos}")
        print(f"Ground truth bugs: {result.num_gt_bugs}")
        print(f"Predicted bugs: {result.num_pred_bugs}")
        print(f"Matched pairs: {result.num_matched}")
        print(f"\nDescription Quality:")
        print(f"  Mean score: {result.mean_score:.2f} / 5.0")
        print(f"\nTemporal Localization:")
        print(f"  Mean IoU: {result.mean_iou:.2f}")
        print(f"\nDetection Metrics:")
        print(f"  Precision: {result.precision*100:.2f}%")
        print(f"  Recall: {result.recall*100:.2f}%")
        print(f"  F1: {result.f1*100:.2f}%")
        print(f"\nIoU-weighted Metrics:")
        print(f"  Precision (IoU): {result.precision_iou*100:.2f}%")
        print(f"  Recall (IoU): {result.recall_iou*100:.2f}%")
        print(f"  F1 (IoU): {result.f1_iou*100:.2f}%")
        print(f"{'='*60}")
