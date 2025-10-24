import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Capture from webcam
cap = cv2.VideoCapture(0)

# To prevent multiple clicks in one pinch
click_cooldown = 0.5  # seconds
last_click_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get index and middle finger tip coordinates
            index_tip = hand_landmarks.landmark[8]   # Index finger tip
            middle_tip = hand_landmarks.landmark[12] # Middle finger tip

            # Convert to pixel coordinates
            x1, y1 = int(index_tip.x * w), int(index_tip.y * h)
            x2, y2 = int(middle_tip.x * w), int(middle_tip.y * h)

            # Draw circles on tips
            cv2.circle(frame, (x1, y1), 10, (255, 0, 255), -1)
            cv2.circle(frame, (x2, y2), 10, (0, 255, 0), -1)

            # Move mouse based on index finger
            mouse_x = np.interp(x1, [0, w], [0, screen_width])
            mouse_y = np.interp(y1, [0, h], [0, screen_height])
            pyautogui.moveTo(mouse_x, mouse_y)

            # Distance between index and middle finger
            distance = math.hypot(x2 - x1, y2 - y1)

            # --- Gesture Actions ---
            current_time = time.time()

            # LEFT CLICK when index and middle are very close
            if distance < 25 and (current_time - last_click_time) > click_cooldown:
                pyautogui.click()
                last_click_time = current_time
                cv2.putText(frame, 'Left Click', (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # RIGHT CLICK when slightly apart but still close
            elif 25 <= distance < 50 and (current_time - last_click_time) > click_cooldown:
                pyautogui.click(button='right')
                last_click_time = current_time
                cv2.putText(frame, 'Right Click', (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display distance on screen
            cv2.putText(frame, f'Distance: {int(distance)}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("🖱️ Virtual Mouse - MediaPipe", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
