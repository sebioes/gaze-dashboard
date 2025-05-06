import cv2
import numpy as np
import msgpack
import os
from tqdm import tqdm
from pathlib import Path
from src.video_processing.dashboard_mask import detect_cockpit_dashboard
import imageio
from typing import Generator, Tuple, Dict, Optional, Any


class GazeMaskAnalyzer:
    def __init__(self, recording_dir, confidence_threshold=0.9, output_dir=None):
        """Initialize the GazeMaskAnalyzer with a recording directory.

        Args:
            recording_dir (str or Path): Path to the recording directory
            confidence_threshold (float): Minimum confidence value (0-1) for a gaze point to be considered valid
            output_dir (str or Path, optional): Directory where processed results should be saved if needed
        """
        self.recording_dir = Path(recording_dir)
        self.confidence_threshold = confidence_threshold
        self.output_dir = Path(output_dir) if output_dir is not None else None

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load timestamps
        self.world_timestamps = np.load(
            os.path.join(self.recording_dir, "world_timestamps.npy")
        )
        self.gaze_timestamps = np.load(
            os.path.join(self.recording_dir, "gaze_timestamps.npy")
        )

        # Print timestamp ranges for verification
        print(
            f"World timestamps range: {self.world_timestamps[0]:.3f} to {self.world_timestamps[-1]:.3f}"
        )
        print(
            f"Gaze timestamps range: {self.gaze_timestamps[0]:.3f} to {self.gaze_timestamps[-1]:.3f}"
        )

        # Open video capture
        self.video_path = os.path.join(self.recording_dir, "world.mp4")
        self.cap = cv2.VideoCapture(str(self.video_path))

        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Load gaze data
        self.gaze_data = self.load_gaze_data()

        # Statistics
        self.frames_with_gaze = 0
        self.frames_gaze_in_mask = 0

        # Temporal smoothing
        self.last_valid_gaze = None
        self.last_valid_is_in_mask = False
        self.frames_since_last_gaze = 0
        self.max_frames_to_keep = 5  # Keep last valid gaze for 5 frames

    def load_gaze_data(self):
        """Load gaze data from the .pldata file."""
        gaze_path = os.path.join(self.recording_dir, "gaze.pldata")
        gaze_data = []

        try:
            with open(gaze_path, "rb") as f:
                unpacker = msgpack.Unpacker(f, raw=False)
                try:
                    while True:
                        topic_list = next(unpacker)
                        if not isinstance(topic_list, list) or len(topic_list) != 2:
                            next(unpacker)
                            continue

                        topic_name = topic_list[0]
                        if not isinstance(topic_name, str) or not topic_name.startswith(
                            "gaze"
                        ):
                            next(unpacker)
                            continue

                        payload = next(unpacker)
                        try:
                            if not isinstance(payload, list) or len(payload) != 2:
                                continue

                            gaze_bytes = payload[1]
                            if not isinstance(gaze_bytes, bytes):
                                continue

                            datum = msgpack.unpackb(gaze_bytes, raw=False)

                            if not isinstance(datum, dict):
                                continue

                            gaze_data.append(datum)

                        except Exception:
                            continue

                except StopIteration:
                    pass

        except Exception as e:
            print(f"Error loading gaze data: {e}")
            return []

        if not gaze_data:
            print("Warning: No valid gaze data was found in the file")
        else:
            print(f"Successfully loaded {len(gaze_data)} gaze data points")

        return gaze_data

    def find_closest_gaze(self, frame_timestamp, frame_idx):
        """Find the closest gaze point to the current frame timestamp."""
        if len(self.gaze_data) == 0 or len(self.gaze_timestamps) == 0:
            return None

        closest_idx = np.searchsorted(self.gaze_timestamps, frame_timestamp)

        if closest_idx >= len(self.gaze_timestamps):
            closest_idx = len(self.gaze_timestamps) - 1
        elif closest_idx > 0:
            prev_diff = abs(self.gaze_timestamps[closest_idx - 1] - frame_timestamp)
            curr_diff = abs(self.gaze_timestamps[closest_idx] - frame_timestamp)
            if prev_diff < curr_diff:
                closest_idx -= 1

        time_diff = abs(self.gaze_timestamps[closest_idx] - frame_timestamp)
        if time_diff > 0.1:  # 100ms threshold
            return None

        gaze_data_idx = closest_idx // 2

        if gaze_data_idx >= len(self.gaze_data):
            return None

        return self.gaze_data[gaze_data_idx]

    def is_gaze_in_mask(self, gaze_point, mask):
        """Check if the gaze point is within the dashboard mask.

        Args:
            gaze_point (dict): The gaze data point containing norm_pos
            mask (numpy.ndarray): Binary mask of the dashboard

        Returns:
            bool: True if gaze is within the mask, False otherwise
        """
        if not gaze_point or "norm_pos" not in gaze_point:
            return False

        try:
            gx, gy = gaze_point["norm_pos"]

            # Clamp normalized coordinates to [0,1] range
            gx = max(0.0, min(1.0, gx))
            gy = max(0.0, min(1.0, gy))

            # Convert to pixel coordinates
            x = int(gx * self.width)
            y = int((1 - gy) * self.height)  # Flip y coordinate

            # Define the size of the area to check (in pixels)
            tolerance_radius = 10  # Check a 20x20 pixel area

            # Calculate the area to check
            x_min = max(0, x - tolerance_radius)
            x_max = min(self.width, x + tolerance_radius)
            y_min = max(0, y - tolerance_radius)
            y_max = min(self.height, y + tolerance_radius)

            # Check if the area is valid (not empty)
            if x_max <= x_min or y_max <= y_min:
                return False

            # Extract the area from the mask
            area = mask[y_min:y_max, x_min:x_max]

            # Check if any part of the area is in the mask
            # If more than 25% of the area is in the mask, consider it a hit
            mask_pixels = np.sum(area > 0)
            total_pixels = area.size

            # Avoid division by zero
            if total_pixels == 0:
                return False

            return (mask_pixels / total_pixels) > 0.25

        except Exception:
            return False

    def draw_gaze_and_status(self, frame, gaze_point, mask, is_in_mask):
        """Draw the gaze point and status indicator on the frame."""
        # Create a grayscale version of the frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(
            gray_frame, cv2.COLOR_GRAY2BGR
        )  # Convert back to BGR for consistent color space

        # Resize mask to match frame dimensions if needed
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(
                mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST
            )

        # Create a composite image where the dashboard area is in color
        # and the rest is in grayscale
        composite = np.where(mask[..., None] > 0, frame, gray_frame)

        if not gaze_point:
            return composite

        try:
            # Check confidence threshold
            confidence = gaze_point.get("confidence", 0.0)
            if confidence < self.confidence_threshold:
                return composite

            gx, gy = gaze_point["norm_pos"]

            # Clamp normalized coordinates to [0,1] range
            gx = max(0.0, min(1.0, gx))
            gy = max(0.0, min(1.0, gy))

            # Convert to pixel coordinates
            x = int(gx * self.width)
            y = int((1 - gy) * self.height)  # Flip y coordinate

            # Draw gaze point with color based on confidence
            # Green for high confidence, yellow for medium, red for low
            if confidence >= 0.8:
                color = (0, 255, 0)  # Green
            elif confidence >= 0.6:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red

            # Draw a larger circle for better visibility
            cv2.circle(composite, (x, y), 30, color, 2)
            cv2.circle(composite, (x, y), 6, color, -1)

            # Draw status indicator with background for better readability
            status_text = (
                f"Gaze in Dashboard (conf: {confidence:.2f})"
                if is_in_mask
                else f"Gaze Outside (conf: {confidence:.2f})"
            )
            text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            cv2.rectangle(
                composite, (5, 5), (text_size[0] + 15, text_size[1] + 15), (0, 0, 0), -1
            )
            cv2.putText(
                composite, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
            )

        except Exception:
            pass

        return composite

    def process_video_stream(
        self, start_frame=None, end_frame=None, show_progress=True
    ) -> Generator[Tuple[np.ndarray, Dict[str, Any]], None, Dict[str, Any]]:
        """Process the video and yield frames as a stream with frame metadata.

        Args:
            start_frame (int, optional): Starting frame index. If None, starts from beginning.
            end_frame (int, optional): Ending frame index. If None, processes until end.
            show_progress (bool): Whether to show a progress bar

        Yields:
            tuple:
                - np.ndarray: Processed frame in RGB format
                - dict: Frame metadata including timestamp, gaze information, etc.

        Returns:
            dict: Final statistics about the gaze analysis when the generator completes
        """
        # Reset statistics in case we're reusing the analyzer
        self.frames_with_gaze = 0
        self.frames_gaze_in_mask = 0

        # Set up start and end frames
        start_frame = 0 if start_frame is None else max(0, start_frame)
        end_frame = (
            self.total_frames
            if end_frame is None
            else min(self.total_frames, end_frame)
        )

        # Ensure we're at the right position in the video
        if start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        total_frames_to_process = end_frame - start_frame

        # Set up progress bar if requested
        pbar = None
        if show_progress:
            pbar = tqdm(total=total_frames_to_process, desc="Processing frames")

        frame_idx = start_frame

        try:
            while frame_idx < end_frame:
                ret, frame = self.cap.read()
                if not ret:
                    print("Video capture ended")
                    break

                if frame_idx >= len(self.world_timestamps):
                    print(f"Reached end of timestamps at frame {frame_idx}")
                    break

                frame_timestamp = self.world_timestamps[frame_idx]

                # Detect dashboard mask
                mask = detect_cockpit_dashboard(frame)

                # Find gaze point and check if it's in the mask
                gaze_point = self.find_closest_gaze(frame_timestamp, frame_idx)

                # Update temporal smoothing
                if (
                    gaze_point
                    and gaze_point.get("confidence", 0.0) >= self.confidence_threshold
                ):
                    self.last_valid_gaze = gaze_point
                    self.last_valid_is_in_mask = self.is_gaze_in_mask(gaze_point, mask)
                    self.frames_since_last_gaze = 0
                else:
                    self.frames_since_last_gaze += 1

                # Use last valid gaze if within the keep window
                if (
                    self.frames_since_last_gaze < self.max_frames_to_keep
                    and self.last_valid_gaze
                ):
                    gaze_point = self.last_valid_gaze
                    is_in_mask = self.last_valid_is_in_mask
                else:
                    is_in_mask = False

                # Update statistics
                if (
                    gaze_point
                    and gaze_point.get("confidence", 0.0) >= self.confidence_threshold
                ):
                    self.frames_with_gaze += 1
                    if is_in_mask:
                        self.frames_gaze_in_mask += 1

                # Draw visualization
                processed_frame = self.draw_gaze_and_status(
                    frame, gaze_point, mask, is_in_mask
                )

                # Convert frame from BGR (OpenCV default) to RGB
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

                # Create frame metadata
                metadata = {
                    "frame_idx": frame_idx,
                    "timestamp": frame_timestamp,
                    "has_valid_gaze": gaze_point is not None
                    and gaze_point.get("confidence", 0.0) >= self.confidence_threshold,
                    "gaze_in_mask": is_in_mask if gaze_point else False,
                    "confidence": gaze_point.get("confidence", 0.0)
                    if gaze_point
                    else 0.0,
                }

                # Yield the frame and metadata
                yield frame_rgb, metadata

                frame_idx += 1
                if pbar:
                    pbar.update(1)

        finally:
            if pbar:
                pbar.close()

        # Prepare final statistics
        percentage = 0
        if self.frames_with_gaze > 0:
            percentage = (self.frames_gaze_in_mask / self.frames_with_gaze) * 100

        stats = {
            "total_frames_processed": frame_idx - start_frame,
            "valid_gaze_frames": self.frames_with_gaze,
            "gaze_in_mask_frames": self.frames_gaze_in_mask,
            "gaze_in_mask_percentage": percentage,
        }

        return stats

    def save_video_stream(self, output_path, start_frame=None, end_frame=None):
        """Save a processed video stream to a file.

        Args:
            output_path (str or Path): Path where the output file should be saved
            start_frame (int, optional): Starting frame index
            end_frame (int, optional): Ending frame index

        Returns:
            tuple: (path to output file, statistics)
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Create video writer using imageio
        writer = imageio.get_writer(
            str(output_path),
            fps=self.fps,
            codec="libx264",
            quality=8,
            ffmpeg_params=["-preset", "medium"],
        )

        # Process and save frames
        final_stats = None
        frame_generator = self.process_video_stream(start_frame, end_frame)

        try:
            for frame, metadata in frame_generator:
                writer.append_data(frame)
        except StopIteration as e:
            # Get the final stats when the generator completes
            final_stats = e.value
        finally:
            writer.close()

        # Print statistics
        if final_stats:
            print(f"\nGaze Analysis Results:")
            print(f"Total frames processed: {final_stats['total_frames_processed']}")
            print(
                f"Total frames with high confidence gaze (>= {self.confidence_threshold}): {final_stats['valid_gaze_frames']}"
            )
            print(
                f"Frames with high confidence gaze in dashboard: {final_stats['gaze_in_mask_frames']}"
            )
            print(
                f"Percentage of high confidence gaze in dashboard: {final_stats['gaze_in_mask_percentage']:.2f}%"
            )

        return str(output_path), final_stats


def create_stream_analyzer(recording_dir, confidence_threshold=0.9) -> GazeMaskAnalyzer:
    """Create a GazeMaskAnalyzer for streaming without writing to disk.

    Args:
        recording_dir (str or Path): Path to the recording directory
        confidence_threshold (float): Minimum confidence value (0-1)

    Returns:
        GazeMaskAnalyzer: Analyzer instance that can process video frames as a stream
    """
    return GazeMaskAnalyzer(recording_dir, confidence_threshold)


def analyze_recording(
    recording_dir, confidence_threshold=0.9, output_dir="data/processed"
):
    """Analyze a gaze recording and generate visualization.

    This maintains the original API for backward compatibility.

    Args:
        recording_dir (str or Path): Path to the recording directory
        confidence_threshold (float): Minimum confidence value (0-1)
        output_dir (str or Path): Directory where processed results should be saved

    Returns:
        str: Path to the output file
        dict: Statistics about the gaze analysis
    """
    analyzer = GazeMaskAnalyzer(recording_dir, confidence_threshold, output_dir)
    recording_name = Path(recording_dir).name
    output_path = Path(output_dir) / f"{recording_name}_analysis.mp4"
    return analyzer.save_video_stream(output_path)


# Example of how to use the streaming functionality
if __name__ == "__main__":
    # Example 1: Process as a stream without saving
    recording_dir = "000"  # Default path for standalone testing
    analyzer = create_stream_analyzer(recording_dir)

    print("Processing video stream...")

    # Example of processing a specific range (e.g., first 100 frames)
    stats = None
    frame_count = 0

    # Process only frames 50-150
    for frame, metadata in analyzer.process_video_stream(start_frame=50, end_frame=150):
        # Here you can do real-time processing with each frame
        frame_count += 1
        print(
            f"Processing frame {metadata['frame_idx']}, timestamp: {metadata['timestamp']:.3f}"
        )

        # In a real application, you might:
        # 1. Display the frame in a window
        # cv2.imshow("Stream", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(1)

        # 2. Send it over a network
        # stream.send(frame)

        # 3. Process it further
        # results = some_processing_function(frame)

    # Example 2: Save a specific segment to disk
    output_path, stats = analyzer.save_video_stream(
        "output_segment.mp4", start_frame=200, end_frame=300
    )

    print(f"Analysis saved to {output_path}")
    print(f"Dashboard gaze percentage: {stats['gaze_in_mask_percentage']:.2f}%")
