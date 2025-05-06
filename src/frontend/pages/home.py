# src/frontend/pages/home.py

"""
Home page for the Gaze Dashboard application.
"""

import streamlit as st
from pathlib import Path
import os
from ..components import GazeRecordingSelector, VideoPlayer, ProcessingControls


def show():
    """Render the home page"""
    st.title("Gaze Analysis Dashboard")

    # Recording selector
    selector = GazeRecordingSelector(data_dir="data/raw")
    selected_recording, was_new_upload = selector.render()

    # Continue only if a recording is selected
    if selected_recording:
        recording_path = Path("data/raw") / selected_recording

        # Processing controls and analysis results
        st.subheader(f"Analysis: {selected_recording}")
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
                st.warning("Processed video not found. Process the recording to view.")
        else:
            st.info("Process the recording to view analysis results.")

    else:
        # Show guidance when no recording is selected
        st.info("Please select or upload a recording to get started.")
