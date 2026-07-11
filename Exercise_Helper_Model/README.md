# 🏋️ Exercise Helper Model — Real-Time Pose Trainer

A real-time exercise form checker using **OpenCV** and **MediaPipe Pose** detection. The app tracks your body joints through your webcam and gives you instant feedback on whether your exercise form is correct or needs improvement.

## ✨ Features

- **10 Exercises** supported with custom form analysis
- **Real-time pose detection** — 33 body landmarks tracked
- **Form feedback** — color-coded: 🟢 Good | 🟡 Adjust | 🔴 Fix Now
- **Rep counter** — automatic counting with state machine
- **Per-exercise timer** — 60 seconds default
- **30-second break** between exercises with countdown
- **Visual HUD** — angles on joints, skeleton overlay, progress bar
- **Workout summary** at the end with total stats

## 📋 Supported Exercises

| # | Exercise | What it Checks |
|---|----------|---------------|
| 1 | Bicep Curls | Elbow angle, upper arm stability |
| 2 | Squats | Knee angle, back straightness, knee position |
| 3 | Push-ups | Elbow angle, body alignment |
| 4 | Lunges | Front knee 90°, torso upright |
| 5 | Shoulder Press | Full arm extension, symmetry |
| 6 | Jumping Jacks | Arms fully up, legs spread |
| 7 | Deadlift | Back straight, hip hinge, knee bend |
| 8 | Plank | Body alignment (hold exercise) |
| 9 | High Knees | Knee height, posture |
| 10 | Lateral Raises | Arm height, symmetry, elbow angle |

## 🚀 Setup & Run

### 1. Install dependencies

```bash
cd Exercise_Helper_Model
pip install -r requirements.txt
```

### 2. Run the app

```bash
python exercise_app.py
```

### 3. Use the app

1. Terminal menu will show — select exercise number(s)
2. You can pick multiple: type `1,2,5` for Bicep Curls + Squats + Shoulder Press
3. Type `11` for full workout (all 10 exercises)
4. Webcam window opens with countdown → exercise starts
5. Watch the feedback on screen!

## ⌨️ Controls

| Key | Action |
|-----|--------|
| `Q` or `ESC` | Quit |
| `S` | Skip current exercise / Skip break |

## 🔧 Configuration

Edit these values at the top of `exercise_app.py`:

```python
EXERCISE_DURATION = 60   # seconds per exercise
BREAK_DURATION = 30      # seconds break between exercises
```

## 📦 Tech Stack

- **Python 3.x**
- **OpenCV** — webcam + visual overlay
- **MediaPipe Pose** — body landmark detection (no GPU needed)
- **NumPy** — angle calculations

## 💡 Tips

- Make sure your full body is visible in the camera
- Good lighting helps with detection accuracy
- Stand about 6-8 feet away from the camera
- Wear fitted clothing for better landmark detection
