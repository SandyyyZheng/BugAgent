"""
Video preprocessing module for extracting frames and segmenting into windows.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import json
from tqdm import tqdm


class VideoPreprocessor:
    """
    Video preprocessor for extracting frames and segmenting into windows.

    This class handles:
    1. Frame extraction from videos at specified FPS
    2. Window segmentation (dividing frames into fixed-size windows)
    3. Metadata generation for tracking video properties
    """

    def __init__(
        self,
        output_path: Path = Path("data"),
        target_fps: float = 5.0,
        window_size: int = 30,
        window_overlap: int = 0
    ):
        """
        Initialize the video preprocessor.

        Args:
            output_path: Base output directory for frames and windows.
            target_fps: Target FPS for frame extraction.
            window_size: Number of frames per window.
            window_overlap: Number of overlapping frames between windows.
        """
        self.output_path = Path(output_path)
        self.target_fps = target_fps
        self.window_size = window_size
        self.window_overlap = window_overlap

    def extract_frames(
        self,
        video_path: Path,
        output_dir: Path = None,
        target_fps: float = None
    ) -> Tuple[int, Dict]:
        """
        Extract frames from a video at the specified FPS using time-based sampling.

        This method samples frames at exact time intervals (e.g., every 0.25s for 4fps),
        ensuring consistent temporal coverage regardless of the original video's fps.

        Args:
            video_path: Path to the input video file.
            output_dir: Directory to save extracted frames. If None, uses output_path/frames/<video_name>.
            target_fps: Target FPS for extraction. If None, uses instance target_fps.

        Returns:
            Tuple of (number of frames extracted, metadata dict)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Setup output directory
        video_name = video_path.stem
        if output_dir is None:
            output_dir = self.output_path / "frames" / video_name
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / original_fps if original_fps > 0 else 0

        # Determine target FPS
        if target_fps is None:
            target_fps = self.target_fps if self.target_fps else original_fps

        # Calculate time interval between samples (time-based sampling)
        sample_interval = 1.0 / target_fps  # e.g., 0.25s for 4fps

        # Calculate target timestamps
        target_timestamps = []
        t = 0.0
        while t < duration:
            target_timestamps.append(t)
            t += sample_interval

        expected_frames = len(target_timestamps)

        print(f"\nExtracting frames from: {video_path.name}")
        print(f"Video properties: {width}x{height}, {original_fps:.2f} FPS, {duration:.2f}s")
        print(f"Target FPS: {target_fps:.2f} (time-based sampling, interval={sample_interval:.4f}s)")
        print(f"Expected frames: {expected_frames}")

        # Extract frames using time-based sampling
        saved_frames = []

        for extracted_count, target_time in enumerate(tqdm(target_timestamps, desc="Extracting frames")):
            # Seek to target timestamp
            cap.set(cv2.CAP_PROP_POS_MSEC, target_time * 1000)
            ret, frame = cap.read()

            if not ret:
                # Try reading sequentially if seek fails
                break

            # Get actual position after seek
            actual_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            actual_frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # -1 because we just read

            frame_filename = f"frame_{extracted_count:06d}.jpg"
            frame_path = output_dir / frame_filename
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            saved_frames.append({
                "frame_id": extracted_count,
                "original_frame_id": actual_frame_id,
                "target_timestamp": target_time,
                "actual_timestamp": actual_time,
                "filename": frame_filename
            })

        cap.release()

        extracted_count = len(saved_frames)

        # Create metadata
        metadata = {
            "video_name": video_name,
            "video_path": str(video_path),
            "original_fps": original_fps,
            "target_fps": target_fps,
            "sample_interval": sample_interval,
            "resolution": [width, height],
            "total_original_frames": total_frames,
            "total_extracted_frames": extracted_count,
            "duration_seconds": duration,
            "frames": saved_frames
        }

        # Save metadata
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Extracted {extracted_count} frames to: {output_dir}")
        print(f"✓ Metadata saved to: {metadata_path}")

        return extracted_count, metadata

    def segment_windows(
        self,
        frames_dir: Path,
        output_dir: Path = None,
        window_size: int = None,
        window_overlap: int = None
    ) -> List[Dict]:
        """
        Segment extracted frames into windows and create stitched images.

        Args:
            frames_dir: Directory containing extracted frames.
            output_dir: Directory to save stitched window images. If None, uses output_path/windows/<video_name>.
            window_size: Number of frames per window. If None, uses instance window_size.
            window_overlap: Number of overlapping frames. If None, uses instance window_overlap.

        Returns:
            List of window metadata dicts.
        """
        frames_dir = Path(frames_dir)
        if not frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

        # Load metadata
        metadata_path = frames_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Setup output directory
        video_name = metadata["video_name"]
        if output_dir is None:
            output_dir = self.output_path / "windows" / video_name
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine window parameters
        if window_size is None:
            window_size = self.window_size
        if window_overlap is None:
            window_overlap = self.window_overlap

        # Get all frame files
        frame_files = sorted(frames_dir.glob("*.jpg"))
        total_frames = len(frame_files)

        if total_frames == 0:
            raise ValueError(f"No frames found in: {frames_dir}")

        print(f"\nSegmenting frames into windows")
        print(f"Total frames: {total_frames}")
        print(f"Window size: {window_size} frames")
        print(f"Window overlap: {window_overlap} frames")

        # Calculate window stride
        stride = window_size - window_overlap
        if stride <= 0:
            raise ValueError("Window overlap must be less than window size")

        # Create windows
        windows_metadata = []
        window_id = 0

        for start_idx in range(0, total_frames, stride):
            end_idx = min(start_idx + window_size, total_frames)

            # Collect frame info for this window
            window_frames = []
            for frame_idx in range(start_idx, end_idx):
                frame_info = metadata["frames"][frame_idx]
                # Support both old format (timestamp) and new format (target_timestamp)
                timestamp = frame_info.get("target_timestamp", frame_info.get("timestamp"))
                window_frames.append({
                    "global_frame_id": frame_idx,
                    "original_frame_id": frame_info["original_frame_id"],
                    "timestamp": timestamp,
                    "filename": frame_info["filename"]
                })

            # Calculate window time range
            start_frame_info = metadata["frames"][start_idx]
            end_frame_info = metadata["frames"][end_idx - 1]
            start_time = start_frame_info.get("target_timestamp", start_frame_info.get("timestamp"))
            end_time = end_frame_info.get("target_timestamp", end_frame_info.get("timestamp"))

            # Create stitched image for this window
            stitched_filename = f"window_{window_id:04d}_stitched.jpg"
            stitched_path = output_dir / stitched_filename

            try:
                self._stitch_window_frames(
                    frames_dir=frames_dir,
                    frame_indices=list(range(start_idx, end_idx)),
                    output_path=stitched_path
                )
            except Exception as e:
                print(f"Warning: Failed to stitch window {window_id}: {e}")
                continue

            # Save window metadata
            window_metadata = {
                "window_id": window_id,
                "video_name": video_name,
                "frame_range": [start_idx, end_idx - 1],
                "time_range": [start_time, end_time],
                "num_frames": end_idx - start_idx,
                "stitched_image": stitched_filename,
                "frames": window_frames
            }

            windows_metadata.append(window_metadata)
            window_id += 1

        # Save overall windows metadata
        overall_metadata = {
            "video_name": video_name,
            "total_windows": len(windows_metadata),
            "window_size": window_size,
            "window_overlap": window_overlap,
            "stride": stride,
            "windows": windows_metadata
        }

        overall_metadata_path = output_dir / "windows_metadata.json"
        with open(overall_metadata_path, "w") as f:
            json.dump(overall_metadata, f, indent=2)

        print(f"✓ Created {len(windows_metadata)} stitched window images in: {output_dir}")
        print(f"✓ Windows metadata saved to: {overall_metadata_path}")

        return windows_metadata

    def _stitch_window_frames(
        self,
        frames_dir: Path,
        frame_indices: List[int],
        output_path: Path
    ) -> None:
        """
        Stitch frames from a window into a single image (2 rows layout).

        Args:
            frames_dir: Directory containing the frames.
            frame_indices: List of frame indices to stitch.
            output_path: Path to save the stitched image.
        """
        num_frames = len(frame_indices)

        if num_frames == 0:
            raise ValueError("No frames to stitch")

        # Read first frame to get dimensions
        first_frame_path = frames_dir / f"frame_{frame_indices[0]:06d}.jpg"
        first_frame = cv2.imread(str(first_frame_path))
        if first_frame is None:
            raise ValueError(f"Failed to read frame: {first_frame_path}")

        frame_h, frame_w = first_frame.shape[:2]

        # Calculate layout: 2 rows, divide frames evenly
        cols = (num_frames + 1) // 2  # Ceiling division for 2 rows
        rows = 2

        # Create canvas
        canvas_w = frame_w * cols
        canvas_h = frame_h * rows
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # Place frames on canvas
        for idx, frame_idx in enumerate(frame_indices):
            # Read frame
            frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
            frame = cv2.imread(str(frame_path))
            if frame is None:
                print(f"Warning: Failed to read {frame_path}, skipping")
                continue

            # Calculate position (left to right, top to bottom)
            row = idx // cols
            col = idx % cols
            y_start = row * frame_h
            x_start = col * frame_w

            # Place frame on canvas
            canvas[y_start:y_start + frame_h, x_start:x_start + frame_w] = frame

            # Add frame number label in top-left corner (using original frame ID)
            label = f"#{frame_idx}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            color = (255, 255, 255)  # White text
            bg_color = (0, 0, 0)  # Black background

            # Get text size
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Draw background rectangle
            padding = 5
            cv2.rectangle(
                canvas,
                (x_start, y_start),
                (x_start + text_w + padding * 2, y_start + text_h + padding * 2),
                bg_color,
                -1
            )

            # Draw text
            cv2.putText(
                canvas,
                label,
                (x_start + padding, y_start + text_h + padding),
                font,
                font_scale,
                color,
                thickness
            )

        # Save stitched image
        cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])

    def process_video(
        self,
        video_path: Path,
        target_fps: float = None,
        window_size: int = None,
        window_overlap: int = None
    ) -> Dict:
        """
        Complete preprocessing pipeline: extract frames and segment into windows.

        Args:
            video_path: Path to the input video file.
            target_fps: Target FPS for extraction. If None, uses instance target_fps.
            window_size: Number of frames per window. If None, uses instance window_size.
            window_overlap: Number of overlapping frames. If None, uses instance window_overlap.

        Returns:
            Dictionary containing processing results and metadata.
        """
        video_path = Path(video_path)
        video_name = video_path.stem

        print(f"\n{'='*60}")
        print(f"Processing video: {video_path.name}")
        print(f"{'='*60}")

        # Step 1: Extract frames
        frames_dir = self.output_path / "frames" / video_name
        num_frames, frame_metadata = self.extract_frames(
            video_path=video_path,
            output_dir=frames_dir,
            target_fps=target_fps
        )

        # Step 2: Segment into windows
        windows_dir = self.output_path / "windows" / video_name
        windows_metadata = self.segment_windows(
            frames_dir=frames_dir,
            output_dir=windows_dir,
            window_size=window_size,
            window_overlap=window_overlap
        )

        result = {
            "video_name": video_name,
            "video_path": str(video_path),
            "frames_dir": str(frames_dir),
            "windows_dir": str(windows_dir),
            "num_frames": num_frames,
            "num_windows": len(windows_metadata),
            "frame_metadata": frame_metadata,
            "windows_metadata": windows_metadata
        }

        print(f"\n{'='*60}")
        print(f"✓ Processing complete!")
        print(f"  Frames: {num_frames} → {frames_dir}")
        print(f"  Windows: {len(windows_metadata)} → {windows_dir}")
        print(f"{'='*60}\n")

        return result
