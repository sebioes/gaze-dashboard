"""
Video player component for gaze recording videos.
"""

import streamlit as st
import cv2
from pathlib import Path
from typing import List, Optional
import os


class VideoPlayer:
    def __init__(self):
        """Initialize the video player component"""
        # Initialize state for video duration
        if "video_durations" not in st.session_state:
            st.session_state.video_durations = {}

    def load_videos(self, video_paths: List[Path]) -> bool:
        """Load video files and get their durations

        Args:
            video_paths: List of paths to video files

        Returns:
            bool: True if videos were loaded successfully
        """
        if not video_paths:
            return False

        # Store the video paths
        self.video_paths = [str(path) for path in video_paths if path.exists()]

        # Get video durations if not already cached
        for video_path in self.video_paths:
            if video_path not in st.session_state.video_durations:
                self._get_video_duration(video_path)

        return len(self.video_paths) > 0

    def _get_video_duration(self, video_path: str) -> float:
        """Get the duration of a video file in seconds"""
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                # Get the frame count and fps
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                # Calculate duration in seconds
                duration = frame_count / fps if fps > 0 else 0
                st.session_state.video_durations[video_path] = duration
                cap.release()
                return duration
            cap.release()
        except Exception as e:
            st.warning(f"Could not determine video duration: {e}")

        # Default duration if we can't determine it
        st.session_state.video_durations[video_path] = 120.0
        return 120.0

    def render(self):
        """Render the video player UI"""
        if not hasattr(self, "video_paths") or not self.video_paths:
            st.info("No videos loaded. Please upload videos first.")
            return

        # Create tabs for each video
        if len(self.video_paths) > 1:
            # Multiple videos - use tabs
            video_names = [Path(path).stem for path in self.video_paths]
            tabs = st.tabs(video_names)

            for i, (tab, video_path) in enumerate(zip(tabs, self.video_paths)):
                with tab:
                    self._render_single_video(video_path, f"video_{i}")
        else:
            # Single video - no tabs needed
            self._render_single_video(self.video_paths[0], "video_0")

    def _render_single_video(self, video_path: str, key_prefix: str):
        """Render a single video with timeline controls"""
        if not os.path.exists(video_path):
            st.error(f"Video file not found: {video_path}")
            return

        # Get video duration
        duration = st.session_state.video_durations.get(video_path, 120.0)

        # Timeline controls
        st.subheader("Timeline Controls")
        col1, col2 = st.columns(2)

        with col1:
            start_time = st.slider(
                "Start Time (seconds)",
                min_value=0.0,
                max_value=max(0.1, duration - 0.1),
                value=0.0,
                step=1.0,
                key=f"{key_prefix}_start",
            )

        with col2:
            end_time = st.slider(
                "End Time (seconds)",
                min_value=0.1,
                max_value=duration,
                value=min(30.0, duration),
                step=1.0,
                key=f"{key_prefix}_end",
            )

        # Ensure end time is greater than start time
        if end_time <= start_time:
            end_time = start_time + 1.0
            st.warning("End time must be greater than start time")

        # Display the video with time range
        st.video(video_path, start_time=start_time, end_time=end_time)

        # Add option to play full video
        if st.button("Play Full Video", key=f"{key_prefix}_full"):
            st.video(video_path)
