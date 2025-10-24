import cv2
import mediapipe as mp
import webbrowser
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Tip landmarks for 5 fingers
tip_ids = [4, 8, 12, 16, 20]

# URLs to open for each gesture
urls = {
    1: "https://www.youtube.com/",
    2: "https://www.google.com/",
    3: "https://web.whatsapp.com/",
    4: "https://chat.openai.com/",
    5: "https://colab.research.google.com/"
}

# Track the last opened gesture
last_opened_gesture = None
gesture_ready = True  # Will become False after a site opens until hand resets (0 fingers)

# Start webcam
cap = cv2.VideoCapture(0)
print("✋ Show 1–5 fingers to open websites!")

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    count = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            if lm_list:
                fingers = []

                # Thumb
                if lm_list[tip_ids[0]][0] > lm_list[tip_ids[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # Other 4 fingers
                for id in range(1, 5):
                    if lm_list[tip_ids[id]][1] < lm_list[tip_ids[id] - 2][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                count = fingers.count(1)
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                cv2.putText(img, f"Fingers: {count}", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

                # Open website only if gesture is ready and not opened before
                if gesture_ready and count in urls and count != 0:
                    print(f"🖐️ Opening: {urls[count]}")
                    webbrowser.open(urls[count])
                    last_opened_gesture = count
                    gesture_ready = False  # Prevent reopening same gesture

                # Reset gesture when hand shows 0 fingers
                elif count == 0:
                    gesture_ready = True

    cv2.imshow("Finger Gesture Web Launcher", img)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()