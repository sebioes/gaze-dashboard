# pages/folder_upload.py

import streamlit as st
from src.frontend.components.folder_uploader import GazeRecordingSelector


def show():
    st.title("Folder Upload")

    gaze_recording_selector = GazeRecordingSelector()
    gaze_recording_selector.render()
