# src/frontend/components/file_loader.py

import streamlit as st


def load_recording_directory():
    st.title("Upload a Gaze Recording Directory")

    recording_directory = st.file_uploader(
        "Choose a gaze recording directory", accept_multiple_files=True
    )

    if recording_directory:
        st.write(recording_directory)

    return recording_directory


if __name__ == "__main__":
    load_recording_directory()
