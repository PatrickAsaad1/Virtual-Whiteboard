# ✋ Air Canvas — Hand Gesture Drawing with MediaPipe

Draw in the air using only your hand. Air Canvas uses your webcam and Google's MediaPipe to track your index finger in real-time, letting you sketch, erase, and create without touching a screen.

---

## Demo

| Gesture                     | Action                               |
| --------------------------- | ------------------------------------ |
| ☝️ Index finger up          | **Draw** — traces a line as you move |
| ✌️ Index + middle finger up | **Erase** — wipes a circular area    |
| ✊ Fist / other             | **Idle** — lifts the pen             |

---

## Features

- Real-time hand landmark detection via MediaPipe
- Frame-accurate drawing with persistent canvas overlay
- Gesture-based mode switching (no keyboard needed)
- Mirrored camera feed for natural interaction
- Lightweight — runs on CPU, no GPU required

---

## Requirements

- Python 3.8+
- Webcam

### Dependencies

```bash
pip install opencv-python mediapipe numpy
```

> **Note:** The MediaPipe Hand Landmarker model (`hand_landmarker.task`) is downloaded automatically on first run (~9 MB).

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/PatrickAsaad1/Virtual-Whiteboard.git
cd air-canvas

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install opencv-python mediapipe numpy

# 4. Run
python air_canvas.py
```

---

## Usage

1. Run the script and allow webcam access.
2. Hold your hand in front of the camera.
3. Use gestures to switch modes (see table above).
4. Press **`Q`** to quit.

The canvas persists across frames — your drawing stays on screen until you erase it or close the app.

---

## How It Works

```
Webcam Frame
     │
     ▼
MediaPipe HandLandmarker  ──►  21 landmarks per hand (x, y, z)
     │
     ▼
Gesture Classifier
  ├── Index up only     →  DRAW  (traces fingertip path on canvas)
  ├── Index + middle up →  ERASE (cv2.circle black fill, r=20)
  └── Otherwise         →  IDLE  (pen lifted, no trail)
     │
     ▼
Canvas Overlay
  Mask non-black pixels → blend onto live frame → display
```

Key landmarks used:

| Landmark ID | Joint                         |
| ----------- | ----------------------------- |
| 8           | Index fingertip               |
| 6           | Index PIP joint (base check)  |
| 12          | Middle fingertip              |
| 10          | Middle PIP joint (base check) |

A finger is considered **"up"** when its tip Y-coordinate is above (less than) its base joint Y-coordinate.

---

## Project Structure

```
air-canvas/
├── air_canvas.py          # Main application
├── hand_landmarker.task   # Auto-downloaded model (gitignored)
└── README.md
```

Add `hand_landmarker.task` to your `.gitignore`:

```
hand_landmarker.task
```

---

## Configuration

You can tweak these values directly in `air_canvas.py`:

| Variable      | Default       | Description                     |
| ------------- | ------------- | ------------------------------- |
| `num_hands`   | `1`           | Max hands to detect             |
| Draw color    | `(0, 255, 0)` | BGR green — change to any color |
| Brush size    | `5`           | Line thickness in pixels        |
| Eraser radius | `20`          | Circle radius in pixels         |

---

## Troubleshooting

**Webcam not detected**
Make sure no other application is using the camera. Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` if you have multiple cameras.

**Low FPS / laggy drawing**
Lower your camera resolution or close background apps consuming CPU.

**Model download fails**
Manually download the model from [Google's MediaPipe releases](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) and place it in the project root.

**Hand not detected**
Ensure even lighting with no strong backlight. Keep your hand roughly 40–80 cm from the camera.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [Google MediaPipe](https://developers.google.com/mediapipe) — hand landmark detection
- [OpenCV](https://opencv.org/) — camera capture and rendering
