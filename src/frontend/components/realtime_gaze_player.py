import streamlit as st
import cv2
import numpy as np
import msgpack
import os
from pathlib import Path
import time
import pandas as pd
import plotly.express as px
from src.video_processing.dashboard_mask import detect_cockpit_dashboard


class RealTimeGazeMaskAnalyzer:
    def __init__(self, recording_dir, confidence_threshold=0.9):
        """Initialize the RealTimeGazeMaskAnalyzer with a recording directory.

        Args:
            recording_dir (str or Path): Path to the recording directory
            confidence_threshold (float): Minimum confidence value (0-1) for a gaze point to be considered valid
        """
        self.recording_dir = Path(recording_dir)
        self.confidence_threshold = confidence_threshold

        # Load timestamps
        self.world_timestamps = np.load(
            os.path.join(self.recording_dir, "world_timestamps.npy")
        )
        self.gaze_timestamps = np.load(
            os.path.join(self.recording_dir, "gaze_timestamps.npy")
        )

        # Print timestamp ranges for verification
        st.sidebar.write(
            f"World timestamps range: {self.world_timestamps[0]:.3f} to {self.world_timestamps[-1]:.3f}"
        )
        st.sidebar.write(
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

        # Statistics tracking
        self.frames_with_gaze = 0
        self.frames_gaze_in_mask = 0
        self.frame_count = 0

        # Tracking data for visualization
        self.gaze_history = []

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
            st.error(f"Error loading gaze data: {e}")
            return []

        if not gaze_data:
            st.warning("Warning: No valid gaze data was found in the file")
        else:
            st.sidebar.info(f"Successfully loaded {len(gaze_data)} gaze data points")

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
        """Check if the gaze point is within the dashboard mask."""
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

    def get_frame(self, frame_idx):
        """Get a specific frame from the video."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return None

        if frame_idx >= len(self.world_timestamps):
            return None

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
        self.frame_count += 1
        if (
            gaze_point
            and gaze_point.get("confidence", 0.0) >= self.confidence_threshold
        ):
            self.frames_with_gaze += 1
            if is_in_mask:
                self.frames_gaze_in_mask += 1

            # Add to history
            confidence = gaze_point.get("confidence", 0.0)
            self.gaze_history.append(
                {
                    "frame": frame_idx,
                    "timestamp": frame_timestamp,
                    "in_mask": is_in_mask,
                    "confidence": confidence,
                    "x": gaze_point["norm_pos"][0],
                    "y": gaze_point["norm_pos"][1],
                }
            )

        # Draw visualization
        processed_frame = self.draw_gaze_and_status(frame, gaze_point, mask, is_in_mask)

        # Convert from BGR to RGB for streamlit
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        return processed_frame

    def get_statistics(self):
        """Get current statistics."""
        stats = {
            "total_frames": self.frame_count,
            "valid_gaze_frames": self.frames_with_gaze,
            "gaze_in_mask_frames": self.frames_gaze_in_mask,
            "gaze_in_mask_percentage": 0,
        }

        if self.frames_with_gaze > 0:
            stats["gaze_in_mask_percentage"] = (
                self.frames_gaze_in_mask / self.frames_with_gaze
            ) * 100

        return stats

    def get_gaze_history_df(self):
        """Get the gaze history as a DataFrame."""
        if not self.gaze_history:
            return pd.DataFrame()
        return pd.DataFrame(self.gaze_history)

    def cleanup(self):
        """Release video resources."""
        self.cap.release()


def gaze_analyzer_component(
    container,
    recording_dir=None,
    confidence_threshold=0.9,
    key_prefix="gaze_analyzer",
    show_controls=True,
    show_stats=True,
    show_plots=True,
):
    """
    A callable Streamlit component that displays a real-time gaze analyzer.

    Args:
        container: The Streamlit container to render the component in
        recording_dir (str, optional): Path to the recording directory. If None, will show an input field.
        confidence_threshold (float, optional): Minimum confidence value for gaze points.
        key_prefix (str, optional): Prefix for session state keys to avoid conflicts when using multiple components.
        show_controls (bool, optional): Whether to show playback controls.
        show_stats (bool, optional): Whether to show statistics panel.
        show_plots (bool, optional): Whether to show plots.

    Returns:
        dict: The current statistics from the analyzer
    """
    # Initialize session state keys with prefix to avoid conflicts
    analyzer_key = f"{key_prefix}_analyzer"
    playing_key = f"{key_prefix}_playing"
    frame_key = f"{key_prefix}_frame"
    speed_key = f"{key_prefix}_speed"

    # Initialize session state if needed
    if analyzer_key not in st.session_state:
        st.session_state[analyzer_key] = None
    if playing_key not in st.session_state:
        st.session_state[playing_key] = False
    if frame_key not in st.session_state:
        st.session_state[frame_key] = 0
    if speed_key not in st.session_state:
        st.session_state[speed_key] = 1.0

    # Setup container for the component
    with container:
        # Allow user to input recording directory if not provided
        if recording_dir is None:
            recording_dir = st.text_input(
                "Recording directory", value="000", key=f"{key_prefix}_rec_dir"
            )
            confidence_threshold = st.slider(
                "Confidence threshold",
                0.0,
                1.0,
                confidence_threshold,
                0.05,
                key=f"{key_prefix}_conf",
            )

        # Initialize analyzer button (only shown if not already initialized)
        if st.session_state[analyzer_key] is None:
            init_clicked = st.button("Initialize Analyzer", key=f"{key_prefix}_init")

            if init_clicked:
                # If path exists, initialize the analyzer
                if os.path.exists(recording_dir):
                    with st.spinner("Loading gaze data..."):
                        analyzer_obj = RealTimeGazeMaskAnalyzer(
                            recording_dir, confidence_threshold
                        )
                        # Only update session state after the analyzer is created
                        st.session_state[analyzer_key] = analyzer_obj
                        # Reset frame position
                        if st.session_state[frame_key] != 0:
                            st.session_state[frame_key] = 0
                    st.success("Analyzer initialized successfully!")
                else:
                    st.error(f"Recording directory not found: {recording_dir}")

        stats = {}

        # Show the analyzer interface if initialized
        if st.session_state[analyzer_key]:
            analyzer = st.session_state[analyzer_key]

            # Create layout based on what we want to show
            if show_stats and show_plots:
                col1, col2 = st.columns([7, 3])
            else:
                col1 = st
                col2 = None

            with col1:
                # Video display placeholder
                video_placeholder = st.empty()

                if show_controls:
                    # Frame slider
                    new_frame_pos = st.slider(
                        "Frame",
                        0,
                        analyzer.total_frames - 1,
                        st.session_state[frame_key],
                        key=f"{key_prefix}_slider",
                    )

                    # Update current frame if slider was moved
                    if new_frame_pos != st.session_state[frame_key]:
                        st.session_state[frame_key] = new_frame_pos

                    # Control buttons
                    cols = st.columns(5)
                    with cols[0]:
                        start_clicked = st.button("⏮ Start", key=f"{key_prefix}_start")
                        if start_clicked:
                            st.session_state[frame_key] = 0
                    with cols[1]:
                        back_clicked = st.button("⏪ -100", key=f"{key_prefix}_back")
                        if back_clicked:
                            st.session_state[frame_key] = max(
                                0, st.session_state[frame_key] - 100
                            )
                    with cols[2]:
                        if st.session_state[playing_key]:
                            pause_clicked = st.button(
                                "⏸ Pause", key=f"{key_prefix}_pause"
                            )
                            if pause_clicked:
                                st.session_state[playing_key] = False
                        else:
                            play_clicked = st.button("▶️ Play", key=f"{key_prefix}_play")
                            if play_clicked:
                                st.session_state[playing_key] = True
                    with cols[3]:
                        forward_clicked = st.button(
                            "⏩ +100", key=f"{key_prefix}_forward"
                        )
                        if forward_clicked:
                            st.session_state[frame_key] = min(
                                analyzer.total_frames - 1,
                                st.session_state[frame_key] + 100,
                            )
                    with cols[4]:
                        end_clicked = st.button("⏭ End", key=f"{key_prefix}_end")
                        if end_clicked:
                            st.session_state[frame_key] = analyzer.total_frames - 1

                    # Use the select_slider without direct session state assignment
                    speed_value = st.select_slider(
                        "Playback Speed",
                        options=[0.25, 0.5, 1.0, 2.0, 4.0],
                        value=st.session_state[speed_key],
                        key=f"{key_prefix}_speed",
                    )

                    # Update session state only if the value has changed
                    if speed_value != st.session_state[speed_key]:
                        st.session_state[speed_key] = speed_value

            # Stats panel
            if show_stats and col2:
                with col2:
                    # Statistics display
                    st.subheader("Current Statistics")
                    stats = analyzer.get_statistics()

                    # Create metrics display
                    st.metric("Processed Frames", stats["total_frames"])
                    st.metric("Valid Gaze Frames", stats["valid_gaze_frames"])
                    st.metric(
                        "Gaze in Dashboard", f"{stats['gaze_in_mask_percentage']:.2f}%"
                    )

                    # Progress bar for gaze in dashboard
                    st.progress(stats["gaze_in_mask_percentage"] / 100)

                    # Display gaze history chart if available
                    if show_plots:
                        df = analyzer.get_gaze_history_df()
                        if not df.empty:
                            st.subheader("Gaze Location History")

                            # Show gaze confidence over time
                            fig = px.scatter(
                                df,
                                x="frame",
                                y="confidence",
                                color="in_mask",
                                color_discrete_map={True: "green", False: "red"},
                                labels={
                                    "frame": "Frame",
                                    "confidence": "Confidence",
                                    "in_mask": "Gaze in Dashboard",
                                },
                            )
                            fig.update_layout(
                                height=200, margin=dict(l=0, r=0, t=20, b=20)
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # Show heatmap of gaze positions
                            st.subheader("Gaze Position Heatmap")
                            fig = px.density_heatmap(
                                df,
                                x="x",
                                y="y",
                                labels={"x": "X position", "y": "Y position"},
                            )
                            fig.update_layout(
                                height=200, margin=dict(l=0, r=0, t=20, b=20)
                            )
                            # Invert y-axis to match screen coordinates
                            fig.update_yaxes(autorange="reversed")
                            st.plotly_chart(fig, use_container_width=True)

            # Display current frame
            current_frame = analyzer.get_frame(st.session_state[frame_key])
            if current_frame is not None:
                video_placeholder.image(
                    current_frame, caption=f"Frame: {st.session_state[frame_key]}"
                )

            # Handle auto-play
            if st.session_state[playing_key]:
                # Calculate the frame increment based on play speed
                frame_increment = int(
                    analyzer.fps * st.session_state[speed_key] / 5
                )  # Update 5 times per second

                # Calculate the new frame position
                new_frame_pos = st.session_state[frame_key] + frame_increment

                # Check if we've reached the end
                if new_frame_pos >= analyzer.total_frames:
                    new_frame_pos = analyzer.total_frames - 1
                    # Set playing to False on next render to avoid session state conflicts
                    next_playing_state = False
                else:
                    next_playing_state = True

                # Store values for next render
                if new_frame_pos != st.session_state[frame_key]:
                    st.session_state[frame_key] = new_frame_pos
                # Only update playing state if needed
                if st.session_state[playing_key] != next_playing_state:
                    st.session_state[playing_key] = next_playing_state

                # Schedule the next update
                time.sleep(0.2)  # Update every 0.2 seconds
                st.rerun()

        else:
            if recording_dir:
                st.info("Click 'Initialize Analyzer' to start.")
            else:
                st.info(
                    "Please provide a recording directory and initialize the analyzer."
                )

    # Return current stats
    return stats
