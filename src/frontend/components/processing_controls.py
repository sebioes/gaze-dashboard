"""
Component for controlling video processing options.
"""

import streamlit as st
from pathlib import Path
import os
from typing import Optional, Dict, Tuple
from src.video_processing.gaze_mask_analyzer import analyze_recording


class ProcessingControls:
    def __init__(self):
        # Initialize state for processing status
        if "processing_complete" not in st.session_state:
            st.session_state.processing_complete = False
        if "processing_results" not in st.session_state:
            st.session_state.processing_results = None

    def process_recording(self, recording_path, confidence_threshold=0.9):
        """Process the selected recording and store results in session state."""
        try:
            with st.spinner("Processing gaze data... This may take a while"):
                output_path, stats = analyze_recording(
                    recording_path, confidence_threshold=confidence_threshold
                )

            st.session_state.processing_complete = True
            st.session_state.processing_results = {
                "output_path": output_path,
                "stats": stats,
            }
            return True
        except Exception as e:
            st.error(f"Error processing recording: {e}")
            st.session_state.processing_complete = False
            st.session_state.processing_results = None
            return False

    def render(self, recording_path=None) -> Optional[Tuple[str, Dict]]:
        """
        Render the processing controls and return results if available.

        Args:
            recording_path (str or Path): Path to the recording directory to process

        Returns:
            Optional[Tuple[str, Dict]]: Path to the processed video and analysis stats, or None.
        """
        st.subheader("Processing Controls")

        if not recording_path:
            st.warning("Please select a recording to process.")
            return None

        recording_path = Path(recording_path)
        if not recording_path.exists():
            st.error(f"Recording directory does not exist: {recording_path}")
            return None

        # Check if the recording has already been processed
        recording_name = recording_path.name
        processed_path = Path("data/processed") / f"{recording_name}_analysis.mp4"
        already_processed = processed_path.exists()

        # Show processing button and options
        col1, col2 = st.columns(2)

        with col1:
            confidence = st.slider(
                "Gaze Confidence Threshold",
                min_value=0.5,
                max_value=1.0,
                value=0.9,
                step=0.05,
                help="Minimum confidence value (0-1) for a gaze point to be considered valid",
            )

        with col2:
            if already_processed:
                st.info("This recording has already been processed.")
                reprocess = st.button("Re-process Recording")
                process_clicked = reprocess
            else:
                process_clicked = st.button("Process Recording")

        # Handle processing
        if process_clicked:
            # Clear previous results before processing
            st.session_state.processing_complete = False
            st.session_state.processing_results = None
            self.process_recording(recording_path, confidence_threshold=confidence)
            # Rerun to update the state and display results
            st.experimental_rerun()

        # Display results if processing is complete
        if st.session_state.processing_complete and st.session_state.processing_results:
            self.display_stats(st.session_state.processing_results["stats"])
            return st.session_state.processing_results[
                "output_path"
            ], st.session_state.processing_results["stats"]

        # Display existing results if available
        if already_processed:
            # Load basic stats from previous run if possible (or just show path)
            # Note: stats might not be available if app was restarted
            stats = {}
            self.display_stats(stats)  # Display empty stats for now
            return str(processed_path), stats

        return None

    def display_stats(self, stats: Optional[Dict]):
        """Display processing statistics only."""
        st.subheader("Analysis Statistics")

        if not stats:
            st.info(
                "Statistics not available. Process the recording to view detailed stats."
            )
            return

        if "gaze_in_mask_percentage" in stats:
            percentage = stats["gaze_in_mask_percentage"]
            st.metric("Gaze on Dashboard", f"{percentage:.2f}%")

            if "valid_gaze_frames" in stats and stats["valid_gaze_frames"] > 0:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Valid Gaze Frames", stats["valid_gaze_frames"])
                with col2:
                    st.metric("Gaze in Dashboard Frames", stats["gaze_in_mask_frames"])
        else:
            st.info("Detailed statistics not available. Process the recording to view.")
