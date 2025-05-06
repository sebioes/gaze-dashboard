import cv2
import numpy as np


def detect_cockpit_dashboard(frame):
    """Detect the cockpit dashboard in a video frame.

    Args:
        frame (numpy.ndarray): The video frame to process

    Returns:
        numpy.ndarray: Binary mask where white pixels (255) represent the dashboard
    """
    # Convert to grayscale - use cv2.COLOR_BGR2GRAY_FAST if available
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Downsample the image for faster processing
    scale_factor = 0.5
    small_gray = cv2.resize(
        gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA
    )

    blurred = cv2.GaussianBlur(small_gray, (99, 99), 0)

    # Threshold: everything below 40 is considered dashboard
    _, mask = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the mask - use RETR_EXTERNAL for faster processing
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour
        largest = max(contours, key=cv2.contourArea)

        # Create a convex hull around the largest contour
        hull = cv2.convexHull(largest)

        # Create a new mask with the convex hull
        mask_filled = np.zeros_like(mask)
        cv2.drawContours(mask_filled, [hull], -1, 255, -1)

        # Upsample the mask back to original size
        mask = cv2.resize(
            mask_filled, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST
        )

    return mask
