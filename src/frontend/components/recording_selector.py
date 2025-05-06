"""
Component for selecting and copying gaze recording folders.
"""

import streamlit as st
from pathlib import Path
import shutil
import os
from typing import Optional, List, Tuple


class GazeRecordingSelector:
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize session state
        if "show_upload_form" not in st.session_state:
            st.session_state.show_upload_form = False

        if "upload_complete" not in st.session_state:
            st.session_state.upload_complete = False

        if "form_key" not in st.session_state:
            st.session_state.form_key = 0

        if "last_uploaded_path" not in st.session_state:
            st.session_state.last_uploaded_path = None

    def get_available_recordings(self) -> List[str]:
        """Get list of available recordings in the data directory"""
        try:
            return [
                d
                for d in os.listdir(str(self.data_dir))
                if (self.data_dir / d).is_dir()
            ]
        except FileNotFoundError:
            return []

    def _toggle_upload_form(self):
        """Toggle the visibility of the upload form"""
        st.session_state.show_upload_form = not st.session_state.show_upload_form
        if st.session_state.show_upload_form:
            st.session_state.form_key += 1  # Force a new form instance

    def render(self) -> Tuple[Optional[str], bool]:
        """
        Renders the recording selection component with an upload option.

        Returns:
            Tuple of (selected_recording_name, was_new_upload)
        """
        # Both controls on the same row with same height
        col1, col2 = st.columns([4, 1])

        with col1:
            recordings = self.get_available_recordings()
            selected_recording = st.selectbox(
                "Select a recording",
                options=[""] + recordings,
                index=0,
                format_func=lambda x: "Select a recording..." if x == "" else x,
                label_visibility="hidden",
            )

        with col2:
            st.markdown(
                """
        <div style="padding-top: 1.70em;">
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.button(
                "Upload New",
                use_container_width=True,
                key="upload_btn",
                on_click=self._toggle_upload_form,
            )

        # Show upload form if requested
        new_upload = False

        if st.session_state.show_upload_form:
            with st.expander("Upload New Recording", expanded=True):
                # Create a unique key for this form instance
                form_key = f"upload_form_{st.session_state.form_key}"

                folder_path = st.text_input(
                    "Enter the path to your gaze recording folder (e.g., /Users/username/recordings/000)",
                    key=f"{form_key}_path",
                    help="The folder should contain world.mp4 and gaze data files",
                )

                recording_name = st.text_input(
                    "Recording name (optional)",
                    key=f"{form_key}_name",
                    help="A descriptive name for this recording",
                )

                # Buttons side by side
                col1, col2 = st.columns(2)

                with col1:
                    process_btn = st.button("Process Folder", key=f"{form_key}_process")

                with col2:
                    st.button(
                        "Close",
                        key=f"{form_key}_close",
                        on_click=self._toggle_upload_form,
                    )

                if process_btn and folder_path:
                    # Process the folder
                    uploaded_path = self._process_folder(folder_path, recording_name)
                    if uploaded_path:
                        # Successfully uploaded
                        st.session_state.last_uploaded_path = uploaded_path
                        st.session_state.upload_complete = True
                        # Hide the form (will take effect on next rerun)
                        st.session_state.show_upload_form = False

        # Check if we just completed an upload
        if st.session_state.upload_complete and st.session_state.last_uploaded_path:
            selected_recording = st.session_state.last_uploaded_path.name
            new_upload = True
            # Reset the flag
            st.session_state.upload_complete = False

        return selected_recording, new_upload

    def _process_folder(self, folder_path: str, recording_name: str) -> Optional[Path]:
        """Process the selected folder"""
        source_path = Path(folder_path)

        if not source_path.exists() or not source_path.is_dir():
            st.error(f"The folder {folder_path} does not exist or is not a directory.")
            return None

        # Check if essential files exist
        essential_files = ["world.mp4", "gaze.pldata"]
        missing_files = [f for f in essential_files if not (source_path / f).exists()]

        if missing_files:
            st.error(f"Missing essential files: {', '.join(missing_files)}")
            return None

        # Generate target directory name based on parent folder name
        if recording_name:
            target_dir_name = f"{source_path.parts[-2]}_{recording_name}"
        else:
            target_dir_name = f"{source_path.parts[-2]}_{source_path.name}"

        target_path = self.data_dir / target_dir_name

        # Check if target directory already exists
        if target_path.exists():
            st.warning(f"A recording with the name {target_dir_name} already exists.")
            overwrite = st.checkbox("Overwrite existing recording?")
            if overwrite:
                shutil.rmtree(target_path)
            else:
                return None

        # Copy the folder with simple progress indicator
        with st.spinner(f"Copying files from {source_path} to {target_path}..."):
            try:
                shutil.copytree(source_path, target_path)
                st.success(f"Successfully copied recording to {target_path}")
                return target_path
            except Exception as e:
                st.error(f"Error copying folder: {e}")
                return None
