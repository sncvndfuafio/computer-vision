import cv2
import mediapipe as mp
import numpy as np
import math
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import keyboard  # To simulate volume key press

# Initialize mediapipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Access speaker volume via PyCaw
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

vol_range = volume.GetVolumeRange()  # (-65.25, 0.0)
min_vol = vol_range[0]
max_vol = vol_range[1]
vol = 0
vol_bar = 400
vol_per = 0
prev_vol = -1  # Track last volume to avoid flickering

# Webcam
cap = cv2.VideoCapture(0)
p_time = 0

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

                # Convert hand range (20–250) to volume range
                vol = np.interp(length, [20, 250], [min_vol, max_vol])
                vol_bar = np.interp(length, [20, 250], [400, 150])
                vol_per = np.interp(length, [20, 250], [0, 100])

                # Update system volume if changed significantly
                if abs(vol_per - prev_vol) > 3:
                    volume.SetMasterVolumeLevel(vol, None)

                    # Simulate volume key press to show system overlay
                    if vol_per > prev_vol:
                        keyboard.press_and_release("volume up")
                    else:
                        keyboard.press_and_release("volume down")

                    prev_vol = vol_per

                # Visual feedback
                if length < 30:
                    cv2.circle(img, (cx, cy), 15, (0, 255, 0), -1)

        # Volume bar
        cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 3)
        cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (255, 0, 0), -1)
        cv2.putText(img, f'{int(vol_per)} %', (40, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        # FPS display
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(img, f'FPS: {int(fps)}', (40, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

        cv2.imshow("Hand Volume Control", img)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()
