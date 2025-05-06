"""
Home page for the Gaze Dashboard application.
"""

import streamlit as st
from pathlib import Path
import os
import cv2
import numpy as np
from ..components import (
    GazeRecordingSelector,
    VideoPlayer,
    ProcessingControls,
)
from src.video_processing.gaze_mask_analyzer import create_stream_analyzer


def gaze_analyzer_streaming_component(
    container, recording_dir, start_frame=None, end_frame=None
):
    """Real-time gaze analyzer component that uses the streaming API.

    Args:
        container: Streamlit container to render the component in
        recording_dir: Path to the recording directory
        start_frame: Optional starting frame (default: None = start from beginning)
        end_frame: Optional ending frame (default: None = process until end)
    """
    # Create analyzer
    analyzer = create_stream_analyzer(recording_dir)

    # Get video properties
    fps = analyzer.fps
    total_frames = analyzer.total_frames

    # Create frame selection sliders if not provided
    if start_frame is None or end_frame is None:
        with container:
            start_frame, end_frame = st.slider(
                "Select Frame Range",
                0,
                total_frames - 1,
                (0, min(100, total_frames - 1)),
            )

            if start_frame >= end_frame:
                st.error("Start frame must be less than end frame")
                return

    # Create placeholder for video and dashboard percentage
    video_placeholder = container.empty()
    stats_placeholder = container.empty()

    # Create stop button to interrupt streaming
    stop_button = container.button("Stop Processing")

    # Initialize stats tracking
    frames_with_gaze = 0
    frames_gaze_in_mask = 0
    processed_frames = 0

    # Create a placeholder for progress bar
    progress_bar = container.progress(0)

    # Process frames up to the maximum number or until stopped
    try:
        # Get frame generator
        frame_generator = analyzer.process_video_stream(
            start_frame=start_frame, end_frame=end_frame, show_progress=False
        )

        # List to store frames for animation
        frames = []

        # Maximum frames to store (to avoid memory issues)
        max_frames_to_store = 100

        for i, (frame, metadata) in enumerate(frame_generator):
            if stop_button:
                container.warning("Processing stopped by user")
                break

            # Update stats
            processed_frames += 1

            if metadata["has_valid_gaze"]:
                frames_with_gaze += 1
                if metadata["gaze_in_mask"]:
                    frames_gaze_in_mask += 1

            # Update progress
            progress = (metadata["frame_idx"] - start_frame) / (end_frame - start_frame)
            progress_bar.progress(min(1.0, progress))

            # Store frame for display
            frames.append(frame)

            # If we've accumulated enough frames or reached the end, show them
            if len(frames) >= 10 or i == end_frame - start_frame - 1:
                # Display the latest frame
                video_placeholder.image(
                    frames[-1],
                    caption=f"Frame: {metadata['frame_idx']}",
                    use_container_width=True,
                )

                # Update stats display
                gaze_percentage = (
                    (frames_gaze_in_mask / frames_with_gaze * 100)
                    if frames_with_gaze > 0
                    else 0
                )
                stats_placeholder.metric(
                    "Dashboard Gaze Percentage",
                    f"{gaze_percentage:.2f}%",
                    f"{frames_with_gaze}/{processed_frames} frames with valid gaze",
                )

                # Clear frames except the last one
                if len(frames) > max_frames_to_store:
                    frames = frames[-1:]

            # Small delay to make it visible in Streamlit
            if i % 3 == 0:
                st.session_state.last_frame = frame

    except Exception as e:
        container.error(f"Error processing video: {e}")
    finally:
        # Final stats
        if processed_frames > 0:
            gaze_percentage = (
                (frames_gaze_in_mask / frames_with_gaze * 100)
                if frames_with_gaze > 0
                else 0
            )
            container.success(
                f"Processing complete. Analyzed {processed_frames} frames."
            )
            container.metric(
                "Final Dashboard Gaze Percentage",
                f"{gaze_percentage:.2f}%",
                f"{frames_with_gaze}/{processed_frames} frames with valid gaze",
            )
        else:
            container.error("No frames were processed.")


def save_segment_component(container, recording_dir):
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


def show():
    """Render the home page"""
    st.title("Gaze Analysis Dashboard")

    # Recording selector
    selector = GazeRecordingSelector(data_dir="data/raw")
    selected_recording, was_new_upload = selector.render()

    # Continue only if a recording is selected
    if selected_recording:
        recording_path = Path("data/raw") / selected_recording

        # Add tabs for different visualization methods
        tab1, tab2, tab3 = st.tabs(
            ["Real-time Streaming", "Save Segments", "Standard Analysis"]
        )

        with tab1:
            # Use the new streaming gaze analyzer component
            st.subheader(f"Real-time Analysis: {selected_recording}")
            gaze_container = st.container()
            gaze_analyzer_streaming_component(gaze_container, str(recording_path))

        with tab2:
            # Component to save specific segments
            save_segment_component(st.container(), str(recording_path))

        with tab3:
            # Traditional processing controls and analysis results
            st.subheader(f"Standard Analysis: {selected_recording}")
            processor = ProcessingControls()
            # Render controls and get results (processed path and stats)
            results = processor.render(recording_path)
            if results:
                processed_video_path, stats = results
                # Display the processed video using VideoPlayer
                if processed_video_path and os.path.exists(processed_video_path):
                    st.subheader("Processed Video")
                    analysis_player = VideoPlayer()
                    analysis_player.load_videos([Path(processed_video_path)])
                    analysis_player.render()
                else:
                    st.warning(
                        "Processed video not found. Process the recording to view."
                    )
            else:
                st.info("Process the recording to view analysis results.")
    else:
        # Show guidance when no recording is selected
        st.info("Please select or upload a recording to get started.")
