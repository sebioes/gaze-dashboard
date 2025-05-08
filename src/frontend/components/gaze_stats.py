import streamlit as st
import numpy as np


class GazeStats:
    def __init__(self):
        # Initialize session state for stats if not exists
        if "gaze_stats" not in st.session_state:
            st.session_state.gaze_stats = {
                "total_frames": 0,
                "valid_gaze_frames": 0,
                "gaze_in_mask_frames": 0,
                "gaze_in_mask_percentage": 0.0,
                "confidence_values": [],
            }

    def compute_sequence_stats(self, metadata_sequence):
        """Compute statistics for an entire sequence of frames.

        Args:
            metadata_sequence (list): List of frame metadata dictionaries from GazeMaskAnalyzer
        """
        stats = {
            "total_frames": len(metadata_sequence),
            "valid_gaze_frames": 0,
            "gaze_in_mask_frames": 0,
            "gaze_in_mask_percentage": 0.0,
            "confidence_values": [],
        }

        # Process all frames in the sequence
        for metadata in metadata_sequence:
            if metadata["has_valid_gaze"]:
                stats["valid_gaze_frames"] += 1
                stats["confidence_values"].append(metadata["confidence"])

                if metadata["gaze_in_mask"]:
                    stats["gaze_in_mask_frames"] += 1

        # Calculate percentage
        if stats["valid_gaze_frames"] > 0:
            stats["gaze_in_mask_percentage"] = (
                stats["gaze_in_mask_frames"] / stats["valid_gaze_frames"]
            ) * 100

        # Update session state
        st.session_state.gaze_stats = stats

    def render(self):
        """Render the statistics display."""
        stats = st.session_state.gaze_stats

        st.markdown("### Gaze Statistics")

        # Display basic stats
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Frames", stats["total_frames"])
            st.metric("Valid Gaze Frames", stats["valid_gaze_frames"])

        with col2:
            st.metric("Gaze in Mask Frames", stats["gaze_in_mask_frames"])
            st.metric("Gaze in Mask %", f"{stats['gaze_in_mask_percentage']:.1f}%")

        # Display confidence statistics if we have data
        if stats["confidence_values"]:
            conf_values = np.array(stats["confidence_values"])
            st.markdown("### Confidence Statistics")
            conf_col1, conf_col2, conf_col3 = st.columns(3)

            with conf_col1:
                st.metric("Average Confidence", f"{np.mean(conf_values):.2f}")
            with conf_col2:
                st.metric("Min Confidence", f"{np.min(conf_values):.2f}")
            with conf_col3:
                st.metric("Max Confidence", f"{np.max(conf_values):.2f}")

            # Display confidence distribution
            st.markdown("### Confidence Distribution")
            st.bar_chart(conf_values)
