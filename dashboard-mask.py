import cv2
import numpy as np

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()

def detect_cockpit_dashboard(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (99, 99), 0)

    # Threshold: everything below 40 is considered dashboard
    _, mask = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY_INV)

    # # Morphological closing to fill gaps
    # kernel = np.ones((10, 10), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest)
        mask_filled = np.zeros_like(mask)
        cv2.drawContours(mask_filled, [hull], -1, 255, -1)
        mask = mask_filled

    dashboard = cv2.bitwise_and(frame, frame, mask=mask)


    # Testing
    # cv2.imshow("Grayscale", gray)
    # cv2.imshow("Blurred", blurred)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return dashboard

def main():
    # video_path = "world-sfu.mov"
    # video_path = "world-sgz.mov"
    # video_path = "world-pox.mov"

    video_path = "001/world.mp4"
    for frame in load_video(video_path):
        dashboard = detect_cockpit_dashboard(frame)
        cv2.imshow("Dashboard", dashboard)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
