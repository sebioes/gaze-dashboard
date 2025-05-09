import cv2
import numpy as np


def detect_cockpit_dashboard(frame):
    """Detect the cockpit dashboard in a video frame.

    Args:
        frame (numpy.ndarray): The video frame to process

    Returns:
        numpy.ndarray: Binary mask where white pixels (255) represent the dashboard
    """
    # Convert to grayscale and blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (99, 99), 0)

    # Threshold: everything below 40 is considered dashboard
    _, mask = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the mask - use RETR_EXTERNAL for faster processing
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour and create a convex hull
        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest)

        # Create a new mask with the convex hull
        mask_filled = np.zeros_like(mask)
        cv2.drawContours(mask_filled, [hull], -1, 255, -1)
        return mask_filled

    return mask
