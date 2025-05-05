"""
Video player component for gaze recording videos.
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time


class VideoPlayer:
    def __init__(self):
        self.video_captures: Dict[str, cv2.VideoCapture] = {}
        self.current_frame: Dict[str, np.ndarray] = {}
        self.frame_counts: Dict[str, int] = {}
        self.fps: Dict[str, float] = {}
        self.current_frame_idx = 0
        self.is_playing = False
        self.last_update_time = 0
        self.update_interval = 1 / 30  # 30 FPS

    def load_videos(self, video_paths: List[Path]):
        """Load video files into the player"""
        # Close any existing video captures
        for cap in self.video_captures.values():
            cap.release()

        self.video_captures.clear()
        self.current_frame.clear()
        self.frame_counts.clear()
        self.fps.clear()

        for video_path in video_paths:
            if not video_path.exists():
                st.error(f"Video file not found: {video_path}")
                continue

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                st.error(f"Failed to open video: {video_path}")
                continue

            # Get video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Store video capture and properties
            video_name = video_path.stem
            self.video_captures[video_name] = cap
            self.frame_counts[video_name] = frame_count
            self.fps[video_name] = fps

            # Read first frame
            ret, frame = cap.read()
            if ret:
                self.current_frame[video_name] = frame

        if not self.video_captures:
            st.error("No valid videos loaded")
            return False

        return True

    def render(self):
        """Render the video player UI"""
        if not self.video_captures:
            st.info("No videos loaded. Please upload videos first.")
            return

        # Create columns for video display
        num_videos = len(self.video_captures)
        cols = st.columns(num_videos)

        # Display videos
        for (video_name, frame), col in zip(self.current_frame.items(), cols):
            with col:
                st.subheader(video_name)
                st.image(frame, channels="BGR", use_column_width=True)

        # Playback controls
        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            if st.button("⏮️ First"):
                self.seek_to_frame(0)

        with col2:
            if st.button("⏯️ Play/Pause"):
                self.is_playing = not self.is_playing

        with col3:
            if st.button("⏭️ Last"):
                self.seek_to_frame(max(self.frame_counts.values()) - 1)

        # Frame slider
        max_frames = min(self.frame_counts.values())
        frame_idx = st.slider(
            "Frame", 0, max_frames - 1, self.current_frame_idx, key="frame_slider"
        )

        if frame_idx != self.current_frame_idx:
            self.seek_to_frame(frame_idx)

        # Playback speed
        playback_speed = st.slider("Playback Speed", 0.25, 2.0, 1.0, 0.25)
        self.update_interval = 1 / (30 * playback_speed)

        # Update frames if playing
        if self.is_playing:
            current_time = time.time()
            if current_time - self.last_update_time >= self.update_interval:
                self.next_frame()
                self.last_update_time = current_time
                st.experimental_rerun()

    def seek_to_frame(self, frame_idx: int):
        """Seek to a specific frame in all videos"""
        self.current_frame_idx = frame_idx

        for video_name, cap in self.video_captures.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                self.current_frame[video_name] = frame

    def next_frame(self):
        """Advance to the next frame in all videos"""
        if self.current_frame_idx >= min(self.frame_counts.values()) - 1:
            self.is_playing = False
            return

        self.current_frame_idx += 1
        for video_name, cap in self.video_captures.items():
            ret, frame = cap.read()
            if ret:
                self.current_frame[video_name] = frame

    def cleanup(self):
        """Clean up video captures"""
        for cap in self.video_captures.values():
            cap.release()
        self.video_captures.clear()
        self.current_frame.clear()
        self.frame_counts.clear()
        self.fps.clear()
