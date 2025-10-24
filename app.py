import cv2
import numpy as np

# Initialize webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture each frame
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for better performance
    frame = cv2.resize(frame, (640, 480))

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Convert edges (single channel) to 3-channel to display side-by-side
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Combine original and edge images horizontally
    combined = np.hstack((frame, edges_colored))

    # Show the combined video
    cv2.imshow("Original (Left)  |  Edges (Right)", combined)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
