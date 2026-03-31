"""
Tool implementations for the Analyzer investigation loop.

Active tools:
  - VQATool:           Visual question answering via MLLM
  - ObjectTrackingTool: Frame-by-frame SAM3 tracking + auto physics analysis
  - MathCalculationTool: Velocity / acceleration / anomaly analysis on trajectories
"""

import math
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from PIL import Image as _PILImage
    PIL_AVAILABLE = True
except ImportError:
    _PILImage = None
    PIL_AVAILABLE = False

from llm import LLMClient
from logger import get_logger

_log = get_logger(__name__)

# ── SAM3 optional import ───────────────────────────────────────────────────────

try:
    import sys
    _TOOLBOX = Path(__file__).parent.parent.parent / "GlitchAgentV2" / "toolbox"
    if str(_TOOLBOX) not in sys.path:
        sys.path.insert(0, str(_TOOLBOX.parent))
    from toolbox.video_tracking import SAM3VideoTracker
    SAM3_AVAILABLE = True
    _log.debug("SAM3VideoTracker imported successfully")
except ImportError:
    SAM3VideoTracker = None
    SAM3_AVAILABLE = False
    _log.debug("SAM3 not available — ObjectTrackingTool will be disabled")


# ── Abstract base ──────────────────────────────────────────────────────────────

class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, str]: ...

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]: ...


# ── VQA Tool ───────────────────────────────────────────────────────────────────

class VQATool(Tool):
    """
    Visual Question Answering tool using a multimodal LLM.

    Answers specific questions about stitched window images or video frames.
    This is the primary tool used by the Analyzer.
    """

    def __init__(
        self,
        api_key: str = "EMPTY",
        api_base: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        temperature: float = 0.3,
        max_tokens: int = 512,
        timeout: int = 60,
        llm_client: Optional[LLMClient] = None,
    ):
        self.client = llm_client or LLMClient(
            api_key=api_key,
            api_base=api_base,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    @property
    def name(self) -> str:
        return "vqa"

    @property
    def description(self) -> str:
        return (
            "Ask specific questions about video frames or images. "
            "Use this to verify visual details, check object states, "
            "or get descriptions of specific regions."
        )

    @property
    def parameters(self) -> Dict[str, str]:
        return {
            "image_path": "Path to the image to analyze",
            "question": "Specific question about the image content",
        }

    def execute(self, image_path: str, question: str, **kwargs) -> Dict[str, Any]:
        image_path = Path(image_path)
        if not image_path.exists():
            return {"error": f"Image not found: {image_path}", "success": False}

        try:
            answer = self.client.chat(
                system_msg="",
                user_msg=question,
                images=[image_path],
            )
            return {
                "success": True,
                "question": question,
                "answer": answer,
                "image_path": str(image_path),
            }
        except Exception as e:
            return {"error": str(e), "success": False}


# ── Math Calculation Tool ──────────────────────────────────────────────────────

class MathCalculationTool(Tool):
    """
    Analyze object trajectories and compute physics metrics.

    Takes centroid sequences (frame_id → (x, y), normalized 0-1) from
    ObjectTrackingTool and computes velocity, acceleration, and detects
    motion anomalies (teleportation, freezing, jittering).

    This tool is also used internally by ObjectTrackingTool to auto-chain
    physics analysis after every tracking call.
    """

    def __init__(
        self,
        fps: float = 4.0,
        frame_width: int = 1920,
        frame_height: int = 1080,
    ):
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height

    @property
    def name(self) -> str:
        return "math_calculation"

    @property
    def description(self) -> str:
        return (
            "Analyze trajectory data (centroids) to compute velocity, acceleration, "
            "and detect physics anomalies such as teleportation (position_jump), "
            "motion_freeze, velocity_spike, or jittering. "
            "Typically called automatically by object_tracking."
        )

    @property
    def parameters(self) -> Dict[str, str]:
        return {
            "centroids": "Dict mapping frame_id (int) -> [x, y] normalized centroid positions",
            "analysis_type": "One of: 'velocity', 'acceleration', 'anomaly', 'all' (default: 'all')",
        }

    def execute(
        self,
        centroids: Dict[int, Tuple[float, float]],
        analysis_type: str = "all",
        **kwargs,
    ) -> Dict[str, Any]:
        if not centroids or len(centroids) < 2:
            return {
                "error": "Need at least 2 data points for analysis.",
                "success": False,
            }

        sorted_frames = sorted(centroids.keys())
        positions = [centroids[f] for f in sorted_frames]

        # Convert normalized [0-1] to pixel coordinates
        positions_px = [
            (p[0] * self.frame_width, p[1] * self.frame_height)
            for p in positions
        ]

        results: Dict[str, Any] = {"success": True, "num_points": len(positions)}

        # ── Velocities ──
        velocities = []
        for i in range(1, len(positions_px)):
            dt = (sorted_frames[i] - sorted_frames[i - 1]) / self.fps
            if dt <= 0:
                continue
            dx = positions_px[i][0] - positions_px[i - 1][0]
            dy = positions_px[i][1] - positions_px[i - 1][1]
            speed = math.sqrt(dx ** 2 + dy ** 2) / dt
            velocities.append({
                "frame": sorted_frames[i],
                "vx": dx / dt,
                "vy": dy / dt,
                "speed": speed,
            })

        if analysis_type in ("velocity", "all"):
            results["velocities"] = velocities
            results["avg_speed_px_s"] = (
                sum(v["speed"] for v in velocities) / len(velocities)
                if velocities else 0.0
            )
            results["max_speed_px_s"] = (
                max(v["speed"] for v in velocities) if velocities else 0.0
            )

        # ── Accelerations ──
        accelerations = []
        for i in range(1, len(velocities)):
            dt = (velocities[i]["frame"] - velocities[i - 1]["frame"]) / self.fps
            if dt <= 0:
                continue
            ax = (velocities[i]["vx"] - velocities[i - 1]["vx"]) / dt
            ay = (velocities[i]["vy"] - velocities[i - 1]["vy"]) / dt
            accelerations.append({
                "frame": velocities[i]["frame"],
                "ax": ax,
                "ay": ay,
                "magnitude": math.sqrt(ax ** 2 + ay ** 2),
            })

        if analysis_type in ("acceleration", "all"):
            results["accelerations"] = accelerations
            results["max_accel_px_s2"] = (
                max(a["magnitude"] for a in accelerations) if accelerations else 0.0
            )

        # ── Anomaly Detection ──
        if analysis_type in ("anomaly", "all"):
            anomalies = []
            frame_diagonal = math.sqrt(self.frame_width ** 2 + self.frame_height ** 2)

            # Position jumps (teleportation)
            for i in range(1, len(positions_px)):
                dx = abs(positions_px[i][0] - positions_px[i - 1][0])
                dy = abs(positions_px[i][1] - positions_px[i - 1][1])
                jump = math.sqrt(dx ** 2 + dy ** 2)
                # Flag if jump > 20% of frame diagonal (~440px on 1080p)
                if jump > 0.20 * frame_diagonal:
                    anomalies.append({
                        "type": "position_jump",
                        "frame": sorted_frames[i],
                        "distance_px": round(jump, 1),
                        "pct_diagonal": round(jump / frame_diagonal * 100, 1),
                        "description": (
                            f"Position jumped {jump:.0f}px "
                            f"({jump/frame_diagonal*100:.0f}% of diagonal) "
                            f"between frames {sorted_frames[i-1]}→{sorted_frames[i]}"
                        ),
                    })

            # Velocity spikes (sudden speed change)
            for i in range(1, len(velocities)):
                speed_change = abs(velocities[i]["speed"] - velocities[i - 1]["speed"])
                if speed_change > 500:  # px/s threshold
                    anomalies.append({
                        "type": "velocity_spike",
                        "frame": velocities[i]["frame"],
                        "speed_change_px_s": round(speed_change, 1),
                        "description": (
                            f"Speed changed by {speed_change:.0f}px/s "
                            f"at frame {velocities[i]['frame']}"
                        ),
                    })

            # Motion freeze (< 1px movement for 3+ consecutive frames)
            freeze_frames = []
            for i in range(1, len(positions_px)):
                dx = abs(positions_px[i][0] - positions_px[i - 1][0])
                dy = abs(positions_px[i][1] - positions_px[i - 1][1])
                if math.sqrt(dx ** 2 + dy ** 2) < 1.0:
                    freeze_frames.append(sorted_frames[i])
            if len(freeze_frames) >= 3:
                anomalies.append({
                    "type": "motion_freeze",
                    "frames": freeze_frames,
                    "num_frames": len(freeze_frames),
                    "description": (
                        f"Object appears frozen for {len(freeze_frames)} frames "
                        f"(< 1px movement per frame)"
                    ),
                })

            # Jittering (direction reversal in ≥ 50% of steps)
            if len(positions_px) >= 4:
                direction_changes = 0
                for i in range(2, len(positions_px)):
                    dx1 = positions_px[i - 1][0] - positions_px[i - 2][0]
                    dx2 = positions_px[i][0] - positions_px[i - 1][0]
                    dy1 = positions_px[i - 1][1] - positions_px[i - 2][1]
                    dy2 = positions_px[i][1] - positions_px[i - 1][1]
                    if dx1 * dx2 < 0 or dy1 * dy2 < 0:
                        direction_changes += 1
                if direction_changes >= (len(positions_px) - 2) * 0.5:
                    anomalies.append({
                        "type": "jittering",
                        "direction_changes": direction_changes,
                        "description": (
                            f"Jittering detected: {direction_changes} direction "
                            f"reversals out of {len(positions_px)-2} steps"
                        ),
                    })

            results["anomalies"] = anomalies
            results["has_anomaly"] = len(anomalies) > 0

        return results


# ── Object Tracking Tool ───────────────────────────────────────────────────────

class ObjectTrackingTool(Tool):
    """
    Track objects across video frames using SAM3.

    Given a natural-language object description and a frames directory,
    uses SAM3VideoTracker to obtain per-frame bounding boxes and centroids.
    Physics analysis (velocity, acceleration, anomaly detection) is
    automatically chained and included in the result — the Planner only
    needs to call this once to get quantitative evidence.

    Session lifecycle (managed by GlitchAnalyzer):
      - GlitchAnalyzer calls set_frames_dir() once before the batch.
      - The first execute() call starts the SAM3 session.
      - Each subsequent execute() call resets then re-prompts the session.
      - GlitchAnalyzer calls close() after analyze_windows_batch completes.
    """

    def __init__(
        self,
        frames_dir: Optional[Path] = None,
        fps: float = 4.0,
        frame_width: int = 1920,
        frame_height: int = 1080,
        gpus: Optional[List[int]] = None,
    ):
        self.frames_dir: Optional[Path] = Path(frames_dir) if frames_dir else None
        self._gpus = gpus or [1]  # default GPU 1 to stay off the VLM's GPU 0
        self._tracker = None       # lazy — created on first execute() call
        self._session_started = False
        self._math = MathCalculationTool(
            fps=fps, frame_width=frame_width, frame_height=frame_height
        )
        # No model loading here; SAM3 is expensive and only needed if Planner calls it

    def _ensure_tracker(self) -> bool:
        """Lazily initialize SAM3VideoTracker on first use. Returns True if ready."""
        if self._tracker is not None:
            return True
        if not SAM3_AVAILABLE:
            _log.warning("SAM3 is not installed — object_tracking unavailable")
            return False
        try:
            _log.info(f"Loading SAM3VideoTracker on GPU(s) {self._gpus} ...")
            self._tracker = SAM3VideoTracker(gpus_to_use=self._gpus, verbose=False)
            _log.info(f"SAM3VideoTracker ready on GPU(s) {self._gpus}")
            return True
        except Exception as e:
            _log.warning(f"SAM3VideoTracker init failed: {e}")
            self._tracker = None
            return False

    def set_frames_dir(self, frames_dir: Path) -> None:
        """Update the frames directory (call before starting a new video batch)."""
        new_dir = Path(frames_dir)
        if self.frames_dir != new_dir:
            # Close existing session if open
            self.close()
            self.frames_dir = new_dir
            _log.debug(f"ObjectTrackingTool frames_dir set to {frames_dir}")

    @property
    def name(self) -> str:
        return "object_tracking"

    @property
    def description(self) -> str:
        status = "" if SAM3_AVAILABLE else " [NOT AVAILABLE — SAM3 not installed]"
        return (
            f"Track a specified object across individual video frames using SAM3{status}. "
            "Returns frame-by-frame bounding boxes, centroids, and automatic physics "
            "analysis (velocity, acceleration, anomaly detection). "
            "Use this when VQA evidence is ambiguous and you need quantitative proof "
            "for a Physics-type glitch (floating, teleportation, jittering)."
        )

    @property
    def parameters(self) -> Dict[str, str]:
        return {
            "object_description": (
                "Concise natural-language description of the object to track "
                "(e.g., 'player character', 'red sports car', 'white NPC'). "
                "Keep it brief and distinctive."
            ),
        }

    def execute(
        self,
        object_description: str,
        frame_range: Optional[List[int]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Track an object and return bounding boxes + physics analysis.

        Args:
            object_description: Short text to prompt SAM3 for the target object.
            frame_range:         [start_frame, end_frame] to filter results.
                                 Tracking always runs on the full video; this
                                 just narrows what is returned and analyzed.
        Returns:
            {
                success, object_description, num_frames, frame_range,
                boxes:     {frame_id: [x1,y1,x2,y2] normalized},
                centroids: {frame_id: [cx, cy] normalized},
                physics:   MathCalculationTool result dict,
            }
        """
        if not self._ensure_tracker():
            return {
                "error": (
                    "SAM3 tracker could not be initialized "
                    "(check GPU memory and SAM3 installation). Use vqa instead."
                ),
                "success": False,
            }
        if self.frames_dir is None or not self.frames_dir.exists():
            return {
                "error": f"frames_dir not set or missing: {self.frames_dir}",
                "success": False,
            }

        start_frame = frame_range[0] if frame_range else 0

        try:
            # Start session lazily (once per video); reset between windows
            if not self._session_started:
                _log.debug(f"SAM3 start_session on {self.frames_dir}")
                self._tracker.start_session(str(self.frames_dir))
                self._session_started = True
            else:
                _log.debug("SAM3 reset_session (new window)")
                self._tracker.reset_session()

            # Detect the object via text prompt
            _log.debug(
                f"SAM3 add_text_prompt: '{object_description}' @ frame {start_frame}"
            )
            detection = self._tracker.add_text_prompt(
                text=object_description,
                frame_index=start_frame,
            )

            if len(detection.get("out_obj_ids", [])) == 0:
                return {
                    "error": f"SAM3 found no objects matching '{object_description}'.",
                    "success": False,
                    "object_description": object_description,
                }

            # Propagate — limited to the window's frame range if provided
            _log.debug(f"SAM3 propagate_in_video (frame_range={frame_range}) ...")
            all_outputs = self._tracker.propagate_in_video(frame_range=frame_range)

            # Build boxes / centroids, filtering to frame_range if given
            boxes: Dict[int, List[float]] = {}
            centroids: Dict[int, List[float]] = {}

            for frame_idx, frame_out in all_outputs.items():
                if frame_range is not None:
                    if frame_idx < frame_range[0] or frame_idx > frame_range[1]:
                        continue
                boxes_xywh = frame_out.get("out_boxes_xywh")
                if boxes_xywh is None or len(boxes_xywh) == 0:
                    continue

                xywh = boxes_xywh[0]
                if hasattr(xywh, "cpu"):
                    xywh = xywh.cpu().numpy()

                cx, cy, w, h = float(xywh[0]), float(xywh[1]), float(xywh[2]), float(xywh[3])
                x1, y1 = cx - w / 2, cy - h / 2
                x2, y2 = cx + w / 2, cy + h / 2

                boxes[frame_idx] = [
                    round(x1, 4), round(y1, 4),
                    round(x2, 4), round(y2, 4),
                ]
                centroids[frame_idx] = [round(cx, 4), round(cy, 4)]

            if not boxes:
                return {
                    "error": "No tracking results found in the specified frame range.",
                    "success": False,
                    "object_description": object_description,
                    "frame_range": frame_range,
                }

            actual_range = [min(boxes.keys()), max(boxes.keys())]
            _log.debug(
                f"Tracking complete: {len(boxes)} frames | "
                f"range={actual_range} | obj='{object_description}'"
            )

            # Auto-chain physics analysis
            centroid_tuples = {f: (c[0], c[1]) for f, c in centroids.items()}
            physics = self._math.execute(centroids=centroid_tuples, analysis_type="all")

            if physics.get("has_anomaly"):
                anomaly_types = [a["type"] for a in physics.get("anomalies", [])]
                _log.info(
                    f"ObjectTracking physics anomalies detected: {anomaly_types}"
                )

            return {
                "success": True,
                "object_description": object_description,
                "num_frames": len(boxes),
                "frame_range": actual_range,
                "boxes": boxes,
                "centroids": centroids,
                "physics": physics,
            }

        except Exception as e:
            _log.error(f"ObjectTrackingTool.execute failed: {e}", exc_info=True)
            return {"error": str(e), "success": False}

    def close(self) -> None:
        """Close the SAM3 session (call after analyze_windows_batch)."""
        if self._tracker is not None and self._session_started:
            try:
                self._tracker.close_session()
                _log.debug("SAM3 session closed")
            except Exception as e:
                _log.warning(f"SAM3 close_session failed: {e}")
            self._session_started = False

    def shutdown(self) -> None:
        """Fully shut down the SAM3 predictor (call when done with the tool)."""
        self.close()
        if self._tracker:
            try:
                self._tracker.shutdown()
                _log.debug("SAM3 predictor shutdown")
            except Exception as e:
                _log.warning(f"SAM3 shutdown failed: {e}")
            self._tracker = None


# ── Zoom-In Tool ───────────────────────────────────────────────────────────────

class ZoomInTool(Tool):
    """
    Crop and magnify a region of one or more video frames, then run VQA.

    Supports:
      - Single frame (frame_index: int)  → crop + upscale to 1024×1024
      - Multiple frames (frame_index: list[int]) → crop each, stitch into grid

    region can be:
      - Spatial name: top_left / top_center / top_right /
                      center_left / center / center_right /
                      bottom_left / bottom_center / bottom_right / full
      - Normalized bbox: [x1, y1, x2, y2] in [0, 1]

    VQA is automatically chained — a single call returns the zoomed image path
    and the model's answer.
    """

    SPATIAL_REGIONS: Dict[str, Tuple[float, float, float, float]] = {
        "top_left":      (0.00, 0.00, 0.50, 0.50),
        "top_center":    (0.25, 0.00, 0.75, 0.50),
        "top_right":     (0.50, 0.00, 1.00, 0.50),
        "center_left":   (0.00, 0.25, 0.50, 0.75),
        "center":        (0.25, 0.25, 0.75, 0.75),
        "center_right":  (0.50, 0.25, 1.00, 0.75),
        "bottom_left":   (0.00, 0.50, 0.50, 1.00),
        "bottom_center": (0.25, 0.50, 0.75, 1.00),
        "bottom_right":  (0.50, 0.50, 1.00, 1.00),
        "full":          (0.00, 0.00, 1.00, 1.00),
    }
    SINGLE_SIZE = (1024, 1024)  # output size for single-frame zoom
    CELL_SIZE   = (640,  640)   # per-cell size in multi-frame grid
    MAX_COLS    = 3             # max columns in multi-frame grid
    GAP         = 4             # gap (px) between grid cells, filled with dark grey

    def __init__(
        self,
        frames_dir: Optional[Path] = None,
        llm_client: Optional[Any] = None,
        zoom_cache_dir: Optional[Path] = None,
    ):
        self.frames_dir = Path(frames_dir) if frames_dir else None
        self.zoom_cache_dir = Path(zoom_cache_dir) if zoom_cache_dir else None
        self._vqa = VQATool(llm_client=llm_client) if llm_client else None

    def set_frames_dir(
        self,
        frames_dir: Path,
        zoom_cache_dir: Optional[Path] = None,
    ) -> None:
        self.frames_dir = Path(frames_dir)
        if zoom_cache_dir:
            self.zoom_cache_dir = Path(zoom_cache_dir)

    @property
    def name(self) -> str:
        return "zoom_in"

    @property
    def description(self) -> str:
        return (
            "Crop and magnify a region of one or more video frames, then answer a "
            "visual question about the zoomed area. Use this when VQA on the full "
            "frame is too ambiguous and you need to inspect a specific region closely."
        )

    @property
    def parameters(self) -> Dict[str, str]:
        return {
            "frame_index": (
                "Frame to zoom: single int for one frame, or list of ints for a "
                "multi-frame grid (e.g. 12 or [10, 12, 14])."
            ),
            "region": (
                "Region to zoom into. Spatial name "
                "(top_left/top_center/top_right/center_left/center/center_right/"
                "bottom_left/bottom_center/bottom_right/full) "
                "or normalized bbox [x1, y1, x2, y2] in [0, 1]."
            ),
            "question": "Visual question to ask about the zoomed region.",
        }

    def execute(
        self,
        frame_index: Union[int, List[int]],
        region: Union[str, List[float]],
        question: str,
        **kwargs,
    ) -> Dict[str, Any]:
        if not PIL_AVAILABLE:
            return {"error": "Pillow not installed. Run: pip install Pillow", "success": False}
        if self._vqa is None:
            return {"error": "ZoomInTool has no VQA client configured.", "success": False}
        if self.frames_dir is None or not self.frames_dir.exists():
            return {"error": f"frames_dir not set or missing: {self.frames_dir}", "success": False}

        bbox = self._parse_region(region)
        multi = isinstance(frame_index, list)
        indices = frame_index if multi else [frame_index]

        crops: List[Tuple[int, Any]] = []
        for fi in indices:
            img = self._load_and_crop(fi, bbox)
            if img is not None:
                crops.append((fi, img))

        if not crops:
            return {
                "error": f"Could not load any of the requested frames from {self.frames_dir}.",
                "success": False,
                "frame_index": frame_index,
                "region": region,
            }

        if not multi or len(crops) == 1:
            final = crops[0][1].resize(self.SINGLE_SIZE, _PILImage.LANCZOS)
        else:
            final = self._stitch_grid(crops)

        cache_path = self._cache_path(frame_index, region)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        final.save(cache_path, quality=95)
        _log.debug(
            f"ZoomIn saved: {cache_path.name} | "
            f"frames={[c[0] for c in crops]} | region={region} | bbox={[round(v,3) for v in bbox]}"
        )

        vqa_result = self._vqa.execute(image_path=str(cache_path), question=question)
        answer = vqa_result.get("answer", "")

        return {
            "success": vqa_result.get("success", False),
            "frame_index": frame_index,
            "region": region,
            "bbox": [round(v, 4) for v in bbox],
            "question": question,
            "answer": answer,
            "cropped_image_path": str(cache_path),
        }

    # ── Helpers ────────────────────────────────────────────────────────────

    def _parse_region(
        self, region: Union[str, List[float]]
    ) -> Tuple[float, float, float, float]:
        if isinstance(region, str):
            bbox = self.SPATIAL_REGIONS.get(region.lower())
            if bbox is None:
                _log.warning(f"Unknown region name '{region}'; using 'center'")
                bbox = self.SPATIAL_REGIONS["center"]
            return bbox
        if isinstance(region, (list, tuple)) and len(region) == 4:
            x1, y1, x2, y2 = (float(v) for v in region)
            return (
                max(0.0, min(x1, 1.0)), max(0.0, min(y1, 1.0)),
                max(0.0, min(x2, 1.0)), max(0.0, min(y2, 1.0)),
            )
        _log.warning(f"Invalid region spec '{region}'; using 'center'")
        return self.SPATIAL_REGIONS["center"]

    def _load_and_crop(
        self, frame_index: int, bbox: Tuple[float, float, float, float]
    ) -> Optional[Any]:
        frame_path = self.frames_dir / f"frame_{frame_index:06d}.jpg"
        if not frame_path.exists():
            _log.warning(f"[ZoomIn] frame not found: {frame_path.name}")
            return None
        img = _PILImage.open(frame_path).convert("RGB")
        w, h = img.size
        x1, y1, x2, y2 = bbox
        box = (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
        if box[2] <= box[0] or box[3] <= box[1]:
            _log.warning(f"[ZoomIn] degenerate crop {box} for frame {frame_index}")
            return None
        return img.crop(box)

    def _stitch_grid(self, indexed_crops: List[Tuple[int, Any]]) -> Any:
        n = len(indexed_crops)
        cols = min(n, self.MAX_COLS)
        rows = math.ceil(n / cols)
        cw, ch = self.CELL_SIZE
        g = self.GAP
        grid = _PILImage.new(
            "RGB",
            (cols * cw + (cols - 1) * g, rows * ch + (rows - 1) * g),
            (30, 30, 30),
        )
        for idx, (_, img) in enumerate(indexed_crops):
            row, col = divmod(idx, cols)
            grid.paste(img.resize((cw, ch), _PILImage.LANCZOS), (col * (cw + g), row * (ch + g)))
        return grid

    def _cache_path(
        self, frame_index: Union[int, List[int]], region: Union[str, List[float]]
    ) -> Path:
        cache_dir = self.zoom_cache_dir or (self.frames_dir / "zoom_cache")
        if isinstance(frame_index, list):
            frame_label = "frames_" + "_".join(str(f) for f in frame_index)
        else:
            frame_label = f"frame_{frame_index:06d}"
        if isinstance(region, str):
            region_label = region
        else:
            region_label = "_".join(f"{v:.2f}" for v in region)
        return cache_dir / f"{frame_label}_{region_label}.jpg"


# ── Registry ───────────────────────────────────────────────────────────────────

class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, str]]:
        return [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in self._tools.values()
        ]

    def get_tools_description(self) -> str:
        lines = ["## Available Tools\n"]
        for tool in self._tools.values():
            lines.append(f"### {tool.name}")
            lines.append(f"{tool.description}\n")
            lines.append("Parameters:")
            for param, desc in tool.parameters.items():
                lines.append(f"  - {param}: {desc}")
            lines.append("")
        return "\n".join(lines)
