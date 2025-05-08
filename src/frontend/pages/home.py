"""
Home page for the Gaze Dashboard application.
"""

import streamlit as st
from pathlib import Path
import os
from ..components import (
    GazeRecordingSelector,
    VideoPlayer,
    ProcessingControls,
    GazeAnalyzerStreaming,
    SaveSegment,
)


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
            [
                "Real-time Streaming",
                "Save Segments",
                "Standard Analysis",
            ]
        )

        with tab1:
            # Use the new streaming gaze analyzer component
            st.subheader(f"Real-time Analysis: {selected_recording}")
            gaze_container = st.container()
            gaze_analyzer = GazeAnalyzerStreaming()
            gaze_analyzer.render(gaze_container, str(recording_path))

        with tab2:
            # Component to save specific segments
            save_segment = SaveSegment()
            save_segment.render(st.container(), str(recording_path))

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
