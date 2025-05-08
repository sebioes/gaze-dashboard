"""
Component for saving video segments.
"""

import streamlit as st
from pathlib import Path
from src.video_processing.gaze_mask_analyzer import create_stream_analyzer
from .video_player import VideoPlayer


class SaveSegment:
    def __init__(self):
        pass

    def render(self, container, recording_dir):
        """Component to save a specific segment of processed video to disk.

        Args:
            container: Streamlit container to render the component in
            recording_dir: Path to the recording directory
        """
        container.subheader("Save Video Segment")

        # Create analyzer to get video properties
        analyzer = create_stream_analyzer(recording_dir)
        total_frames = analyzer.total_frames

        # Input for frame range and output path
        col1, col2 = container.columns(2)
        with col1:
            start_frame = st.number_input("Start Frame", 0, total_frames - 1, 0)
        with col2:
            end_frame = st.number_input(
                "End Frame", 0, total_frames - 1, min(100, total_frames - 1)
            )

        output_filename = container.text_input(
            "Output Filename", f"{Path(recording_dir).name}_segment.mp4"
        )
        output_path = Path("data/processed") / output_filename

        if container.button("Save Video Segment"):
            with st.spinner(
                f"Saving video segment from frame {start_frame} to {end_frame}..."
            ):
                try:
                    # Save the video segment
                    output_file, stats = analyzer.save_video_stream(
                        str(output_path), start_frame=start_frame, end_frame=end_frame
                    )

                    container.success(f"Video segment saved to {output_file}")
                    container.metric(
                        "Dashboard Gaze Percentage",
                        f"{stats['gaze_in_mask_percentage']:.2f}%",
                        f"{stats['valid_gaze_frames']}/{stats['total_frames_processed']} frames with valid gaze",
                    )

                    # Offer to play the saved video
                    if container.button("Play Saved Video"):
                        video_player = VideoPlayer()
                        video_player.load_videos([Path(output_file)])
                        video_player.render()

                except Exception as e:
                    container.error(f"Error saving video segment: {e}")
