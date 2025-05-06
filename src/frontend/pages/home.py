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

        # Process and analyze the recording
        processor = ProcessingControls()
        processor.render(recording_path)

    else:
        st.info("Please select or upload a recording to get started.")
