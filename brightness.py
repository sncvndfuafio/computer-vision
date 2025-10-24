import cv2
import mediapipe as mp
import numpy as np
import math
import time
import screen_brightness_control as sbc
import keyboard  # To simulate brightness key press

# Initialize mediapipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)
p_time = 0
prev_brightness = -1  # Track last brightness to prevent flicker

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        h, w, c = img.shape

        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

                # Get landmarks
                lm_list = []
                for id, lm in enumerate(handLms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append((id, cx, cy))

                # Get thumb tip (4) and index finger tip (8)
                x1, y1 = lm_list[4][1], lm_list[4][2]
                x2, y2 = lm_list[8][1], lm_list[8][2]

                # Center point
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Draw line and circles
                cv2.circle(img, (x1, y1), 10, (255, 0, 255), -1)
                cv2.circle(img, (x2, y2), 10, (255, 0, 255), -1)
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.circle(img, (cx, cy), 10, (255, 0, 0), -1)

                # Distance between fingers
                length = math.hypot(x2 - x1, y2 - y1)

                # Convert hand range (20–250) to brightness range (0–100)
                brightness = np.interp(length, [20, 250], [0, 100])

                # Update brightness if changed significantly
                if abs(brightness - prev_brightness) > 5:
                    sbc.set_brightness(int(brightness))
                    prev_brightness = brightness

                    # Simulate key press for system brightness overlay (optional)
                    if brightness > prev_brightness:
                        keyboard.press_and_release("brightness_up")
                    else:
                        keyboard.press_and_release("brightness_down")

                # Visual feedback (small circle if fingers are close)
                if length < 30:
                    cv2.circle(img, (cx, cy), 15, (0, 255, 0), -1)

        # Brightness bar
        bar_pos = np.interp(prev_brightness, [0, 100], [400, 150])
        cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 3)
        cv2.rectangle(img, (50, int(bar_pos)), (85, 400), (255, 0, 0), -1)
        cv2.putText(img, f'{int(prev_brightness)} %', (40, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        # FPS display
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(img, f'FPS: {int(fps)}', (40, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

        cv2.imshow("Hand Brightness Control", img)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()
