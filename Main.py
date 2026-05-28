import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
from pathlib import Path
import numpy as np

# 1. Download model
model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
model_path = Path("hand_landmarker.task")
if not model_path.exists():
    print("Downloading...")
    urllib.request.urlretrieve(model_url, model_path)

# 2. Create detector
base_options = python.BaseOptions(model_asset_path=str(model_path))
options = vision.HandLandmarkerOptions(
    base_options=base_options, running_mode=vision.RunningMode.VIDEO, num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

# 3. Camera loop
cap = cv2.VideoCapture(0)

# --- FIX: Get actual frame size ---
ret, test_frame = cap.read()
if ret:
    h, w, _ = test_frame.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
else:
    print("Couldn't get frame size, using default")
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    h, w = 480, 640

prev_point = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect_for_video(mp_image, int(cv2.getTickCount()))

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            points = hand_landmarks
            h, w, _ = frame.shape
            points_px = []
            for lm in points:
                x = int(lm.x * w)
                y = int(lm.y * h)
                points_px.append((x, y))

            index_tip_y = points_px[8][1]
            index_base_y = points_px[6][1]
            index_up = index_tip_y < index_base_y

            middle_tip_y = points_px[12][1]
            middle_base_y = points_px[10][1]
            middle_up = middle_tip_y < middle_base_y

            if index_up and middle_up:
                mode = "ERASE"
            elif index_up:
                mode = "DRAW"
            else:
                mode = "IDLE"

            print(mode)

            # Drawing logic
            current_x = points_px[8][0]
            current_y = points_px[8][1]
            current_point = (current_x, current_y)

            if mode == "DRAW":
                if prev_point is not None:
                    cv2.line(canvas, prev_point, current_point, (0, 255, 0), 5)
                prev_point = current_point
            elif mode == "ERASE":
                cv2.circle(canvas, current_point, 20, (0, 0, 0), -1)
                prev_point = None
            else:  # IDLE
                prev_point = None

            # Draw points manually
            for lm in points:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    # Overlay canvas
    mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    frame[mask > 0] = canvas[mask > 0]

    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
