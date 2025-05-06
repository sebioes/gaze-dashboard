import cv2
import numpy as np
import msgpack
import os
from tqdm import tqdm
import imageio


class GazeVisualizer:
    def __init__(self, recording_dir):
        """Initialize the GazeVisualizer with a recording directory.

        Args:
            recording_dir (str): Path to the recording directory containing:
                - world.mp4: The video file
                - world_timestamps.npy: Frame timestamps
                - gaze_timestamps.npy: Gaze timestamps
                - gaze.pldata: Gaze data file
        """
        self.recording_dir = recording_dir

        # Load timestamps
        self.world_timestamps = np.load(
            os.path.join(recording_dir, "world_timestamps.npy")
        )
        self.gaze_timestamps = np.load(
            os.path.join(recording_dir, "gaze_timestamps.npy")
        )

        # Print timestamp ranges for verification
        print(
            f"World timestamps range: {self.world_timestamps[0]:.3f} to {self.world_timestamps[-1]:.3f}"
        )
        print(
            f"Gaze timestamps range: {self.gaze_timestamps[0]:.3f} to {self.gaze_timestamps[-1]:.3f}"
        )

        # Open video capture
        self.video_path = os.path.join(recording_dir, "world.mp4")
        self.cap = cv2.VideoCapture(self.video_path)

        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Load gaze data
        self.gaze_data = self.load_gaze_data()

    def load_gaze_data(self):
        """Load gaze data from the .pldata file.

        Returns:
            list: List of gaze data points, each containing:
                - norm_pos: Normalized gaze position [x, y] in range [0,1]
                - confidence: Confidence value
                - timestamp: Timestamp of the gaze point
                - Various 3D gaze data (eye_center_3d, gaze_normal_3d, gaze_point_3d)
        """
        gaze_path = os.path.join(self.recording_dir, "gaze.pldata")
        gaze_data = []

        try:
            with open(gaze_path, "rb") as f:
                unpacker = msgpack.Unpacker(f, raw=False)
                try:
                    while True:
                        # Read topic (which is a list)
                        topic_list = next(unpacker)
                        if not isinstance(topic_list, list) or len(topic_list) != 2:
                            next(unpacker)  # Skip payload
                            continue

                        topic_name = topic_list[0]
                        if not isinstance(topic_name, str) or not topic_name.startswith(
                            "gaze"
                        ):
                            next(unpacker)  # Skip payload
                            continue

                        # Get the payload
                        payload = next(unpacker)
                        try:
                            # The payload is a list with two elements
                            if not isinstance(payload, list) or len(payload) != 2:
                                continue

                            # The second element contains the gaze data
                            gaze_bytes = payload[1]
                            if not isinstance(gaze_bytes, bytes):
                                continue

                            # Decode the gaze data
                            datum = msgpack.unpackb(gaze_bytes, raw=False)

                            if not isinstance(datum, dict):
                                continue

                            # We have the gaze data with norm_pos
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
        """Find the closest gaze point to the current frame timestamp.

        Args:
            frame_timestamp (float): Timestamp of the current frame
            frame_idx (int): Index of the current frame

        Returns:
            dict or None: The closest gaze data point if found within 100ms, None otherwise
        """
        if len(self.gaze_data) == 0 or len(self.gaze_timestamps) == 0:
            return None

        # Find the closest gaze point to the current frame timestamp
        closest_idx = np.searchsorted(self.gaze_timestamps, frame_timestamp)

        # Handle edge cases
        if closest_idx >= len(self.gaze_timestamps):
            closest_idx = len(self.gaze_timestamps) - 1
        elif closest_idx > 0:
            # Check if the previous timestamp is closer
            prev_diff = abs(self.gaze_timestamps[closest_idx - 1] - frame_timestamp)
            curr_diff = abs(self.gaze_timestamps[closest_idx] - frame_timestamp)
            if prev_diff < curr_diff:
                closest_idx -= 1

        # Only use gaze points within 100ms of the frame
        time_diff = abs(self.gaze_timestamps[closest_idx] - frame_timestamp)
        if time_diff > 0.1:  # 100ms threshold
            return None

        # Convert the timestamp index to a gaze data index
        # Since we have twice as many timestamps as gaze data points,
        # we need to divide the index by 2
        gaze_data_idx = closest_idx // 2

        # Make sure we have a valid gaze data point
        if gaze_data_idx >= len(self.gaze_data):
            return None

        return self.gaze_data[gaze_data_idx]

    def draw_gaze(self, frame, gaze_point, frame_idx):
        """Draw the gaze point on the frame.

        Args:
            frame (numpy.ndarray): The video frame to draw on
            gaze_point (dict): The gaze data point containing norm_pos
            frame_idx (int): Index of the current frame

        Returns:
            numpy.ndarray: The frame with the gaze point drawn on it
        """
        if not gaze_point:
            return frame

        # Get normalized gaze position
        if "norm_pos" not in gaze_point:
            return frame

        try:
            gx, gy = gaze_point["norm_pos"]

            # Clamp normalized coordinates to [0,1] range
            gx = max(0.0, min(1.0, gx))
            gy = max(0.0, min(1.0, gy))

            # Convert to pixel coordinates
            x = int(gx * self.width)
            y = int((1 - gy) * self.height)  # Flip y coordinate

            # Draw gaze point
            cv2.circle(frame, (x, y), 20, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        except Exception:
            pass

        return frame

    def process_video(self, output_path):
        """Process the video and create a visualization with gaze points.

        Args:
            output_path (str): Path to save the output video
        """
        # Create video writer using imageio
        writer = imageio.get_writer(
            output_path,
            fps=self.fps,
            codec="libx264",
            quality=8,
            ffmpeg_params=["-preset", "ultrafast"],
        )

        pbar = tqdm(total=self.total_frames, desc="Processing frames")
        frame_idx = 0

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Video capture ended")
                    break

                # Get timestamp for current frame
                if frame_idx >= len(self.world_timestamps):
                    print(f"Reached end of timestamps at frame {frame_idx}")
                    break
                frame_timestamp = self.world_timestamps[frame_idx]

                # Find and draw gaze point
                gaze_point = self.find_closest_gaze(frame_timestamp, frame_idx)
                frame = self.draw_gaze(frame, gaze_point, frame_idx)

                # Convert frame from BGR (OpenCV default) to RGB (imageio default)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Write frame using imageio
                writer.append_data(frame_rgb)
                frame_idx += 1
                pbar.update(1)

        finally:
            pbar.close()
            self.cap.release()
            writer.close()
            print(f"Processed {frame_idx} frames out of {self.total_frames}")


if __name__ == "__main__":
    recording_dir = "001"  # Path to recording directory
    output_path = "gaze_visualization.mp4"

    visualizer = GazeVisualizer(recording_dir)
    visualizer.process_video(output_path)
    print(f"Visualization saved to {output_path}")
