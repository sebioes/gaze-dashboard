# src/frontend/pages/home.py

"""
Home page for the Gaze Dashboard application.
"""

import streamlit as st
from pathlib import Path
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

        # Create tabs for raw video and analysis
        tab1, tab2 = st.tabs(["Raw Video", "Dashboard Analysis"])

        with tab1:
            # Display info about the recording
            st.subheader(f"Recording: {selected_recording}")

            # Find video files
            video_files = list(recording_path.glob("*.mp4"))

            if video_files:
                # Initialize and display video player
                player = VideoPlayer()
                player.load_videos(video_files)
                player.render()
            else:
                st.warning("No video files found in the selected recording.")

        with tab2:
            # Process and analyze the recording
            processor = ProcessingControls()
            processor.render(recording_path)
    else:
        st.info("Please select or upload a recording to get started.")
