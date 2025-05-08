"""
Frontend components for the Gaze Dashboard application.
"""

from .recording_selector import GazeRecordingSelector
from .video_player import VideoPlayer
from .processing_controls import ProcessingControls
from .gaze_analyzer_streaming import GazeAnalyzerStreaming
from .save_segment import SaveSegment

__all__ = [
    "GazeRecordingSelector",
    "VideoPlayer",
    "ProcessingControls",
    "GazeAnalyzerStreaming",
    "SaveSegment",
]
