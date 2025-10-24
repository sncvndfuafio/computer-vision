import cv2
import mediapipe as mp
import pyautogui
import subprocess
import time
import numpy as np

# Launch Microsoft Paint
subprocess.Popen("mspaint")
time.sleep(3)  # wait for paint to open

# Initialize mediapipe hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Screen size
screen_w, screen_h = pyautogui.size()

# Webcam
cap = cv2.VideoCapture(0)

prev_click = False

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in handLms.landmark]

            # Get finger coordinates
            x_index, y_index = lm_list[8]  # Index finger tip
            x_thumb, y_thumb = lm_list[4]  # Thumb tip

            # Convert hand coordinates to screen coordinates
            screen_x = np.interp(x_index, [0, w], [0, screen_w])
            screen_y = np.interp(y_index, [0, h], [0, screen_h])
            pyautogui.moveTo(screen_x, screen_y, duration=0.01)

            # Distance between index and thumb (to detect "pinch" for click)
            distance = np.hypot(x_thumb - x_index, y_thumb - y_index)

            # Draw cursor on webcam feed
            cv2.circle(img, (x_index, y_index), 10, (255, 0, 255), -1)

            # Click when fingers close together
            if distance < 25:
                if not prev_click:
                    pyautogui.mouseDown()
                    prev_click = True
            else:
                if prev_click:
                    pyautogui.mouseUp()
                    prev_click = False

    cv2.imshow("Virtual Paint Controller", img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
