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

    def update_stats(self, metadata):
        """Update statistics based on new frame metadata.

        Args:
            metadata (dict): Frame metadata from GazeMaskAnalyzer
        """
        stats = st.session_state.gaze_stats

        # Update total frames
        stats["total_frames"] += 1

        # Update valid gaze frames
        if metadata["has_valid_gaze"]:
            stats["valid_gaze_frames"] += 1
            stats["confidence_values"].append(metadata["confidence"])

            # Update gaze in mask frames
            if metadata["gaze_in_mask"]:
                stats["gaze_in_mask_frames"] += 1

        # Calculate percentage
        if stats["valid_gaze_frames"] > 0:
            stats["gaze_in_mask_percentage"] = (
                stats["gaze_in_mask_frames"] / stats["valid_gaze_frames"]
            ) * 100

    def reset_stats(self):
        """Reset the statistics."""
        st.session_state.gaze_stats = {
            "total_frames": 0,
            "valid_gaze_frames": 0,
            "gaze_in_mask_frames": 0,
            "gaze_in_mask_percentage": 0.0,
            "confidence_values": [],
        }

    def render(self):
        """Render the statistics display."""
        stats = st.session_state.gaze_stats

        st.markdown("#### Gaze Analysis")

        st.metric(
            "Eyes on Instruments Panel",
            f"{stats['gaze_in_mask_percentage']:.1f}%",
            help="Percentage of valid gaze points that fall within the instrument panel",
        )
        # Secondary information in an expander
        with st.expander("Processing Metrics"):
            st.markdown("#### Frame Analysis")
            frame_col1, frame_col2 = st.columns(2)
            with frame_col1:
                st.metric(
                    "Valid Gaze Frames",
                    stats["valid_gaze_frames"],
                    help="Number of frames with valid gaze detection",
                )
            with frame_col2:
                st.metric(
                    "Total Frames",
                    stats["total_frames"],
                    help="Total number of frames analyzed",
                )

            # Confidence statistics if we have data
            if stats["confidence_values"]:
                conf_values = np.array(stats["confidence_values"])
                st.markdown("#### Confidence Analysis")
                conf_col1, conf_col2, conf_col3 = st.columns(3)

                with conf_col1:
                    st.metric("Average Confidence", f"{np.mean(conf_values):.2f}")
                with conf_col2:
                    st.metric("Min Confidence", f"{np.min(conf_values):.2f}")
                with conf_col3:
                    st.metric("Max Confidence", f"{np.max(conf_values):.2f}")

                # Display confidence distribution
                # st.markdown("#### Confidence Distribution")
                # min_conf = np.min(conf_values)
                # adjusted_conf_values = conf_values - min_conf
                # st.bar_chart(adjusted_conf_values, height=200)
