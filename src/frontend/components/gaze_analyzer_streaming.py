"""
Real-time gaze analyzer streaming component.
"""

import streamlit as st
import time
from src.video_processing.gaze_mask_analyzer import create_stream_analyzer
from src.frontend.components.gaze_stats import GazeStats


class GazeAnalyzerStreaming:
    def render(self, container, recording_dir, start_frame=None, end_frame=None):
        """Real-time gaze analyzer component that uses the streaming API.

        Args:
            container: Streamlit container to render the component in
            recording_dir: Path to the recording directory
            start_frame: Optional starting frame (default: None = start from beginning)
            end_frame: Optional ending frame (default: None = process until end)
        """
        # Initialize analyzer and get video properties
        analyzer = create_stream_analyzer(recording_dir)
        # fps = analyzer.fps
        fps = 30  # TODO: hardcoded appears to work better for now
        total_frames = analyzer.total_frames
        frame_delay = 1.0 / fps

        col1, col2 = container.columns([3, 1])

        with col1:
            video_placeholder = st.empty()
            start_frame, end_frame = render_start_stop_frame_slider(total_frames)
            progress_bar = st.progress(0)
            stop_button = st.button("Stop Processing")
            metadata_sequence = analyzer.compute_metadata_for_frame_sequence(
                start_frame, end_frame
            )

            try:
                # Process video frames
                frame_generator = analyzer.process_video_stream(
                    start_frame=start_frame, end_frame=end_frame, show_progress=False
                )

                for i, (frame, metadata) in enumerate(frame_generator):
                    if stop_button:
                        container.warning("Processing stopped by user")
                        break

                    # Update progress
                    progress = (metadata["frame_idx"] - start_frame) / (
                        end_frame - start_frame
                    )
                    progress_bar.progress(min(1.0, progress))

                    # Display current frame
                    video_placeholder.image(
                        frame,
                        use_container_width=True,
                    )
                    # Maintain target FPS
                    time.sleep(frame_delay)

            except Exception as e:
                container.error(f"Error processing video: {e}")
        with col2:
            gaze_stats = GazeStats()
            gaze_stats.compute_sequence_stats(metadata_sequence)
            gaze_stats.render()


def render_start_stop_frame_slider(total_frames):
    mid_frame = total_frames // 2
    start_frame, end_frame = st.slider(
        label="Select Frame Range",
        min_value=0,
        max_value=total_frames - 1,
        value=(max(0, mid_frame - 300), min(mid_frame + 300, total_frames - 1)),
        step=30,
    )
    if start_frame == end_frame:
        start_frame = end_frame - 300
        return start_frame, end_frame
    return start_frame, end_frame
