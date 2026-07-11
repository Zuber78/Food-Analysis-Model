"""
=============================================================
  EXERCISE HELPER MODEL — Real-Time Pose Analysis & Feedback
  Uses OpenCV + MediaPipe Tasks API for body pose detection
  Run: python exercise_app.py
=============================================================
"""

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarkerResult,
    RunningMode,
    PoseLandmark,
)
import numpy as np
import time
import math
import sys
import os
import threading

# ─────────────────────────────────────────────
# Model Path
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker.task")

# ─────────────────────────────────────────────
# Color Constants (BGR for OpenCV)
# ─────────────────────────────────────────────
COLOR_GREEN = (0, 220, 0)
COLOR_YELLOW = (0, 220, 255)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_CYAN = (255, 255, 0)
COLOR_ORANGE = (0, 165, 255)
COLOR_PURPLE = (255, 50, 150)
COLOR_BG_DARK = (30, 30, 30)
COLOR_BG_PANEL = (40, 40, 50)
COLOR_ACCENT = (255, 140, 0)
COLOR_GOLD = (0, 215, 255)

# ─────────────────────────────────────────────
# Exercise Timer Config (seconds per exercise)
# ─────────────────────────────────────────────
EXERCISE_DURATION = 60      # seconds per exercise
BREAK_DURATION = 30         # seconds break between exercises

# ─────────────────────────────────────────────
# PoseLandmark index constants (same as old API)
# ─────────────────────────────────────────────
LM_NOSE = 0
LM_LEFT_SHOULDER = 11
LM_RIGHT_SHOULDER = 12
LM_LEFT_ELBOW = 13
LM_RIGHT_ELBOW = 14
LM_LEFT_WRIST = 15
LM_RIGHT_WRIST = 16
LM_LEFT_HIP = 23
LM_RIGHT_HIP = 24
LM_LEFT_KNEE = 25
LM_RIGHT_KNEE = 26
LM_LEFT_ANKLE = 27
LM_RIGHT_ANKLE = 28

# ─────────────────────────────────────────────
# Utility: Calculate angle between 3 points
# ─────────────────────────────────────────────
def calculate_angle(a, b, c):
    """
    Calculate angle at point b given 3 points a, b, c.
    Each point is (x, y). Returns angle in degrees.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(math.degrees(radians))
    if angle > 180:
        angle = 360 - angle
    return angle


def get_lm(landmarks, idx, w, h):
    """Get (x, y) pixel coordinates for a landmark by index."""
    lm = landmarks[idx]
    return (int(lm.x * w), int(lm.y * h))


def get_lm_visibility(landmarks, idx):
    """Get visibility score for a landmark by index."""
    return landmarks[idx].visibility if hasattr(landmarks[idx], 'visibility') else landmarks[idx].presence


# ─────────────────────────────────────────────
# Drawing Utilities
# ─────────────────────────────────────────────
def draw_angle_on_joint(img, angle, position, status="good"):
    """Draw angle value near a joint with color based on form quality."""
    color = COLOR_GREEN if status == "good" else (COLOR_YELLOW if status == "warning" else COLOR_RED)
    cv2.putText(img, f"{int(angle)}", (position[0] + 10, position[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def draw_progress_bar(img, x, y, w, h, progress, color=COLOR_GREEN):
    """Draw a progress bar. Progress is 0.0 to 1.0."""
    progress = max(0.0, min(1.0, progress))
    cv2.rectangle(img, (x, y), (x + w, y + h), (80, 80, 80), -1)
    fill_w = int(w * progress)
    if fill_w > 0:
        cv2.rectangle(img, (x, y), (x + fill_w, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_WHITE, 1)


def draw_skeleton(img, landmarks, w, h, form_color):
    """Draw custom colored skeleton lines on key body parts."""
    connections = [
        # Right arm
        (LM_RIGHT_SHOULDER, LM_RIGHT_ELBOW),
        (LM_RIGHT_ELBOW, LM_RIGHT_WRIST),
        # Left arm
        (LM_LEFT_SHOULDER, LM_LEFT_ELBOW),
        (LM_LEFT_ELBOW, LM_LEFT_WRIST),
        # Torso
        (LM_LEFT_SHOULDER, LM_RIGHT_SHOULDER),
        (LM_LEFT_SHOULDER, LM_LEFT_HIP),
        (LM_RIGHT_SHOULDER, LM_RIGHT_HIP),
        (LM_LEFT_HIP, LM_RIGHT_HIP),
        # Right leg
        (LM_RIGHT_HIP, LM_RIGHT_KNEE),
        (LM_RIGHT_KNEE, LM_RIGHT_ANKLE),
        # Left leg
        (LM_LEFT_HIP, LM_LEFT_KNEE),
        (LM_LEFT_KNEE, LM_LEFT_ANKLE),
    ]

    for start_idx, end_idx in connections:
        start = get_lm(landmarks, start_idx, w, h)
        end = get_lm(landmarks, end_idx, w, h)
        cv2.line(img, start, end, form_color, 3, cv2.LINE_AA)

    # Draw joints as circles
    key_indices = [
        LM_LEFT_SHOULDER, LM_RIGHT_SHOULDER,
        LM_LEFT_ELBOW, LM_RIGHT_ELBOW,
        LM_LEFT_WRIST, LM_RIGHT_WRIST,
        LM_LEFT_HIP, LM_RIGHT_HIP,
        LM_LEFT_KNEE, LM_RIGHT_KNEE,
        LM_LEFT_ANKLE, LM_RIGHT_ANKLE,
    ]
    for idx in key_indices:
        pt = get_lm(landmarks, idx, w, h)
        cv2.circle(img, pt, 6, COLOR_WHITE, -1)
        cv2.circle(img, pt, 6, form_color, 2)


# ─────────────────────────────────────────────
# HUD Drawing (Heads-Up Display)
# ─────────────────────────────────────────────
def draw_hud(img, exercise_name, reps, timer_remaining, total_time, feedback_list, form_score):
    """Draw the full HUD overlay on the frame."""
    h, w = img.shape[:2]
    overlay = img.copy()

    # ── Top panel ──
    cv2.rectangle(overlay, (0, 0), (w, 70), COLOR_BG_DARK, -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

    # Exercise name (top-left)
    cv2.putText(img, f"Exercise: {exercise_name}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_GOLD, 2, cv2.LINE_AA)

    # Rep counter (top-center)
    rep_text = f"REPS: {reps}"
    (tw, _), _ = cv2.getTextSize(rep_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(img, rep_text, (w // 2 - tw // 2, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_CYAN, 2, cv2.LINE_AA)

    # Timer (top-right)
    timer_min = int(timer_remaining) // 60
    timer_sec = int(timer_remaining) % 60
    timer_text = f"TIME: {timer_min}:{timer_sec:02d}"
    timer_color = COLOR_GREEN if timer_remaining > 10 else (COLOR_YELLOW if timer_remaining > 5 else COLOR_RED)
    (tw2, _), _ = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(img, timer_text, (w - tw2 - 15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, timer_color, 2, cv2.LINE_AA)

    # Form status indicator
    form_label = "FORM: "
    if form_score == "good":
        form_status = "PERFECT"
        form_color = COLOR_GREEN
    elif form_score == "warning":
        form_status = "ADJUST"
        form_color = COLOR_YELLOW
    else:
        form_status = "FIX NOW!"
        form_color = COLOR_RED

    cv2.putText(img, form_label + form_status, (15, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, form_color, 2, cv2.LINE_AA)

    # Progress bar (below top panel)
    progress = 1.0 - (timer_remaining / total_time) if total_time > 0 else 0
    bar_color = COLOR_GREEN if progress < 0.7 else (COLOR_YELLOW if progress < 0.9 else COLOR_ORANGE)
    draw_progress_bar(img, 10, 72, w - 20, 8, progress, bar_color)

    # ── Bottom feedback panel ──
    if feedback_list:
        panel_h = 30 + len(feedback_list) * 30
        overlay2 = img.copy()
        cv2.rectangle(overlay2, (0, h - panel_h - 10), (w, h), COLOR_BG_DARK, -1)
        cv2.addWeighted(overlay2, 0.8, img, 0.2, 0, img)

        for i, (msg, color) in enumerate(feedback_list):
            y_pos = h - panel_h + 10 + i * 30
            cv2.putText(img, msg, (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════
# EXERCISE ANALYZERS
# Each returns: (reps, stage, form_score, feedback_list)
# ═══════════════════════════════════════════════════════════

# ── 1. BICEP CURLS ──
def analyze_bicep_curls(landmarks, w, h, img, reps, stage):
    feedback = []
    form = "good"

    l_shoulder = get_lm(landmarks, LM_LEFT_SHOULDER, w, h)
    l_elbow = get_lm(landmarks, LM_LEFT_ELBOW, w, h)
    l_wrist = get_lm(landmarks, LM_LEFT_WRIST, w, h)

    r_shoulder = get_lm(landmarks, LM_RIGHT_SHOULDER, w, h)
    r_elbow = get_lm(landmarks, LM_RIGHT_ELBOW, w, h)
    r_wrist = get_lm(landmarks, LM_RIGHT_WRIST, w, h)

    # Calculate elbow angles
    left_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
    right_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
    avg_angle = (left_angle + right_angle) / 2

    draw_angle_on_joint(img, left_angle, l_elbow, "good" if 30 < left_angle < 170 else "bad")
    draw_angle_on_joint(img, right_angle, r_elbow, "good" if 30 < right_angle < 170 else "bad")

    # Rep counting
    if avg_angle > 155:
        if stage == "down":
            reps += 1
        stage = "up"
    elif avg_angle < 40:
        stage = "down"

    # Form checks - upper arm stability
    l_hip = get_lm(landmarks, LM_LEFT_HIP, w, h)
    r_hip = get_lm(landmarks, LM_RIGHT_HIP, w, h)
    left_shoulder_angle = calculate_angle(l_hip, l_shoulder, l_elbow)
    right_shoulder_angle = calculate_angle(r_hip, r_shoulder, r_elbow)

    if left_shoulder_angle > 45 or right_shoulder_angle > 45:
        form = "warning"
        feedback.append((">> Upper arm hil rahi hai - elbow fix rakho!", COLOR_YELLOW))

    if stage == "down" and avg_angle > 50:
        form = "warning"
        feedback.append((">> Aur upar lao - full curl karo!", COLOR_YELLOW))

    if form == "good":
        feedback.append((">> Bilkul sahi! Keep it up!", COLOR_GREEN))

    return reps, stage, form, feedback


# ── 2. SQUATS ──
def analyze_squats(landmarks, w, h, img, reps, stage):
    feedback = []
    form = "good"

    l_hip = get_lm(landmarks, LM_LEFT_HIP, w, h)
    l_knee = get_lm(landmarks, LM_LEFT_KNEE, w, h)
    l_ankle = get_lm(landmarks, LM_LEFT_ANKLE, w, h)
    l_shoulder = get_lm(landmarks, LM_LEFT_SHOULDER, w, h)

    r_hip = get_lm(landmarks, LM_RIGHT_HIP, w, h)
    r_knee = get_lm(landmarks, LM_RIGHT_KNEE, w, h)
    r_ankle = get_lm(landmarks, LM_RIGHT_ANKLE, w, h)

    left_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
    right_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
    avg_knee = (left_knee_angle + right_knee_angle) / 2

    back_angle = calculate_angle(l_shoulder, l_hip, l_knee)

    draw_angle_on_joint(img, left_knee_angle, l_knee, "good" if avg_knee < 100 else "warning")
    draw_angle_on_joint(img, right_knee_angle, r_knee, "good" if avg_knee < 100 else "warning")
    draw_angle_on_joint(img, back_angle, l_hip, "good" if back_angle > 60 else "bad")

    # Rep counting
    if avg_knee > 165:
        if stage == "down":
            reps += 1
        stage = "up"
    elif avg_knee < 100:
        stage = "down"

    if stage == "down" and avg_knee > 110:
        form = "warning"
        feedback.append((">> Aur neeche jao - thighs parallel honi chahiye!", COLOR_YELLOW))

    if back_angle < 50:
        form = "bad"
        feedback.append((">> Kamar seedhi rakho! Aage mat jhuko!", COLOR_RED))

    if l_knee[0] > l_ankle[0] + 50 or r_knee[0] > r_ankle[0] + 50:
        if form != "bad":
            form = "warning"
        feedback.append((">> Ghutne zyada aage ja rahe - ankle ke upar rakho!", COLOR_YELLOW))

    if form == "good":
        feedback.append((">> Perfect squat form! Bahut accha!", COLOR_GREEN))

    return reps, stage, form, feedback


# ── 3. PUSH-UPS ──
def analyze_pushups(landmarks, w, h, img, reps, stage):
    feedback = []
    form = "good"

    l_shoulder = get_lm(landmarks, LM_LEFT_SHOULDER, w, h)
    l_elbow = get_lm(landmarks, LM_LEFT_ELBOW, w, h)
    l_wrist = get_lm(landmarks, LM_LEFT_WRIST, w, h)
    l_hip = get_lm(landmarks, LM_LEFT_HIP, w, h)
    l_ankle = get_lm(landmarks, LM_LEFT_ANKLE, w, h)

    elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
    body_angle = calculate_angle(l_shoulder, l_hip, l_ankle)

    draw_angle_on_joint(img, elbow_angle, l_elbow, "good" if elbow_angle < 100 else "warning")
    draw_angle_on_joint(img, body_angle, l_hip, "good" if 150 < body_angle < 200 else "bad")

    # Rep counting
    if elbow_angle > 155:
        if stage == "down":
            reps += 1
        stage = "up"
    elif elbow_angle < 90:
        stage = "down"

    if body_angle < 150:
        form = "bad"
        feedback.append((">> Hip upar hai - body seedhi line me rakho!", COLOR_RED))
    elif body_angle > 200:
        form = "bad"
        feedback.append((">> Hip neeche gir rahi - core tight rakho!", COLOR_RED))

    if stage == "down" and elbow_angle > 110:
        form = "warning"
        feedback.append((">> Aur neeche jao - chest zameen ke paas lao!", COLOR_YELLOW))

    if form == "good":
        feedback.append((">> Great push-up form! Shandar!", COLOR_GREEN))

    return reps, stage, form, feedback


# ── 4. LUNGES ──
def analyze_lunges(landmarks, w, h, img, reps, stage):
    feedback = []
    form = "good"

    l_hip = get_lm(landmarks, LM_LEFT_HIP, w, h)
    l_knee = get_lm(landmarks, LM_LEFT_KNEE, w, h)
    l_ankle = get_lm(landmarks, LM_LEFT_ANKLE, w, h)

    r_hip = get_lm(landmarks, LM_RIGHT_HIP, w, h)
    r_knee = get_lm(landmarks, LM_RIGHT_KNEE, w, h)
    r_ankle = get_lm(landmarks, LM_RIGHT_ANKLE, w, h)

    l_shoulder = get_lm(landmarks, LM_LEFT_SHOULDER, w, h)

    left_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
    right_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)

    front_angle = min(left_knee_angle, right_knee_angle)

    draw_angle_on_joint(img, left_knee_angle, l_knee)
    draw_angle_on_joint(img, right_knee_angle, r_knee)

    torso_angle = calculate_angle(l_shoulder, l_hip, l_knee)
    draw_angle_on_joint(img, torso_angle, l_hip, "good" if torso_angle > 70 else "warning")

    # Rep counting
    if front_angle > 160:
        if stage == "down":
            reps += 1
        stage = "up"
    elif front_angle < 110:
        stage = "down"

    if stage == "down":
        if front_angle > 120:
            form = "warning"
            feedback.append((">> Aur neeche jao - 90 degree tak!", COLOR_YELLOW))
        if front_angle < 70:
            form = "warning"
            feedback.append((">> Zyada neeche mat jao - knee safe rakho!", COLOR_YELLOW))

    if torso_angle < 60:
        form = "bad"
        feedback.append((">> Seedha khade raho - aage mat jhuko!", COLOR_RED))

    if form == "good":
        feedback.append((">> Perfect lunge! Balance ache hai!", COLOR_GREEN))

    return reps, stage, form, feedback


# ── 5. SHOULDER PRESS ──
def analyze_shoulder_press(landmarks, w, h, img, reps, stage):
    feedback = []
    form = "good"

    l_shoulder = get_lm(landmarks, LM_LEFT_SHOULDER, w, h)
    l_elbow = get_lm(landmarks, LM_LEFT_ELBOW, w, h)
    l_wrist = get_lm(landmarks, LM_LEFT_WRIST, w, h)

    r_shoulder = get_lm(landmarks, LM_RIGHT_SHOULDER, w, h)
    r_elbow = get_lm(landmarks, LM_RIGHT_ELBOW, w, h)
    r_wrist = get_lm(landmarks, LM_RIGHT_WRIST, w, h)

    l_hip = get_lm(landmarks, LM_LEFT_HIP, w, h)

    left_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
    right_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
    avg_angle = (left_angle + right_angle) / 2

    left_arm_raise = calculate_angle(l_hip, l_shoulder, l_elbow)

    draw_angle_on_joint(img, left_angle, l_elbow)
    draw_angle_on_joint(img, right_angle, r_elbow)

    # Rep counting
    if avg_angle > 160:
        if stage == "down":
            reps += 1
        stage = "up"
    elif avg_angle < 90:
        stage = "down"

    if stage == "up" and avg_angle < 165:
        form = "warning"
        feedback.append((">> Arms poori tarah upar extend karo!", COLOR_YELLOW))

    angle_diff = abs(left_angle - right_angle)
    if angle_diff > 20:
        form = "warning"
        feedback.append((">> Dono haath barabar rakho - symmetry fix karo!", COLOR_YELLOW))

    if left_arm_raise < 60:
        form = "warning"
        feedback.append((">> Elbows zyada neeche hai - shoulder level pe lao!", COLOR_YELLOW))

    if form == "good":
        feedback.append((">> Shandar shoulder press! Ekdum sahi!", COLOR_GREEN))

    return reps, stage, form, feedback


# ── 6. JUMPING JACKS ──
def analyze_jumping_jacks(landmarks, w, h, img, reps, stage):
    feedback = []
    form = "good"

    l_shoulder = get_lm(landmarks, LM_LEFT_SHOULDER, w, h)
    l_elbow = get_lm(landmarks, LM_LEFT_ELBOW, w, h)
    l_wrist = get_lm(landmarks, LM_LEFT_WRIST, w, h)
    l_hip = get_lm(landmarks, LM_LEFT_HIP, w, h)
    l_ankle = get_lm(landmarks, LM_LEFT_ANKLE, w, h)

    r_shoulder = get_lm(landmarks, LM_RIGHT_SHOULDER, w, h)
    r_wrist = get_lm(landmarks, LM_RIGHT_WRIST, w, h)
    r_hip = get_lm(landmarks, LM_RIGHT_HIP, w, h)
    r_ankle = get_lm(landmarks, LM_RIGHT_ANKLE, w, h)

    left_arm_angle = calculate_angle(l_hip, l_shoulder, l_wrist)
    right_arm_angle = calculate_angle(r_hip, r_shoulder, r_wrist)
    avg_arm = (left_arm_angle + right_arm_angle) / 2

    hip_width = abs(l_hip[0] - r_hip[0])
    ankle_spread = abs(l_ankle[0] - r_ankle[0])
    spread_ratio = ankle_spread / max(hip_width, 1)

    draw_angle_on_joint(img, left_arm_angle, l_shoulder)
    draw_angle_on_joint(img, right_arm_angle, r_shoulder)

    # Rep counting
    if avg_arm > 150 and spread_ratio > 1.8:
        if stage == "down":
            reps += 1
        stage = "up"
    elif avg_arm < 50 and spread_ratio < 1.2:
        stage = "down"

    if stage == "up" and avg_arm < 160:
        form = "warning"
        feedback.append((">> Haath poore upar le jao - fully extend!", COLOR_YELLOW))

    if stage == "up" and spread_ratio < 1.5:
        form = "warning"
        feedback.append((">> Paer aur failao - wider stance!", COLOR_YELLOW))

    left_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
    if left_elbow_angle < 150:
        form = "warning"
        feedback.append((">> Arms seedhi rakho - elbow mat modo!", COLOR_YELLOW))

    if form == "good":
        feedback.append((">> Great jumping jacks! Energy high hai!", COLOR_GREEN))

    return reps, stage, form, feedback


# ── 7. DEADLIFT ──
def analyze_deadlift(landmarks, w, h, img, reps, stage):
    feedback = []
    form = "good"

    l_shoulder = get_lm(landmarks, LM_LEFT_SHOULDER, w, h)
    l_hip = get_lm(landmarks, LM_LEFT_HIP, w, h)
    l_knee = get_lm(landmarks, LM_LEFT_KNEE, w, h)
    l_ankle = get_lm(landmarks, LM_LEFT_ANKLE, w, h)

    hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
    knee_angle = calculate_angle(l_hip, l_knee, l_ankle)

    draw_angle_on_joint(img, hip_angle, l_hip)
    draw_angle_on_joint(img, knee_angle, l_knee)

    # Rep counting
    if hip_angle > 160:
        if stage == "down":
            reps += 1
        stage = "up"
    elif hip_angle < 100:
        stage = "down"

    if knee_angle > 175:
        form = "warning"
        feedback.append((">> Ghutne thode bend rakho - lock mat karo!", COLOR_YELLOW))

    if knee_angle < 130:
        form = "warning"
        feedback.append((">> Yeh squat nahi - ghutne kam modo!", COLOR_YELLOW))

    if stage == "down":
        if l_shoulder[1] > l_hip[1] + 100:
            form = "bad"
            feedback.append((">> KAMAR SEEDHI RAKHO! Round mat karo!", COLOR_RED))

    if form == "good":
        feedback.append((">> Perfect deadlift form! Back safe hai!", COLOR_GREEN))

    return reps, stage, form, feedback


# ── 8. PLANK ──
def analyze_plank(landmarks, w, h, img, reps, stage):
    """Plank is a hold exercise - form check only."""
    feedback = []
    form = "good"

    l_shoulder = get_lm(landmarks, LM_LEFT_SHOULDER, w, h)
    l_hip = get_lm(landmarks, LM_LEFT_HIP, w, h)
    l_ankle = get_lm(landmarks, LM_LEFT_ANKLE, w, h)

    body_angle = calculate_angle(l_shoulder, l_hip, l_ankle)
    draw_angle_on_joint(img, body_angle, l_hip, "good" if 155 < body_angle < 195 else "bad")

    if body_angle < 150:
        form = "bad"
        feedback.append((">> Hip upar hai - body ek line me rakho!", COLOR_RED))
    elif body_angle > 195:
        form = "bad"
        feedback.append((">> Hip neeche gir rahi - core ko tight karo!", COLOR_RED))
    elif body_angle < 160:
        form = "warning"
        feedback.append((">> Thoda hip neeche karo - almost straight!", COLOR_YELLOW))
    elif body_angle > 185:
        form = "warning"
        feedback.append((">> Hip thoda upar karo - sagging ho rahi!", COLOR_YELLOW))

    if form == "good":
        if stage != "holding":
            stage = "holding"
        feedback.append((">> Perfect plank! Hold karo! Core tight!", COLOR_GREEN))
    else:
        stage = "not_holding"

    return reps, stage, form, feedback


# ── 9. HIGH KNEES ──
def analyze_high_knees(landmarks, w, h, img, reps, stage):
    feedback = []
    form = "good"

    l_hip = get_lm(landmarks, LM_LEFT_HIP, w, h)
    l_knee = get_lm(landmarks, LM_LEFT_KNEE, w, h)
    r_hip = get_lm(landmarks, LM_RIGHT_HIP, w, h)
    r_knee = get_lm(landmarks, LM_RIGHT_KNEE, w, h)
    l_shoulder = get_lm(landmarks, LM_LEFT_SHOULDER, w, h)

    left_knee_raised = l_knee[1] < l_hip[1]
    right_knee_raised = r_knee[1] < r_hip[1]

    # Rep counting
    if left_knee_raised or right_knee_raised:
        if stage == "down":
            reps += 1
            stage = "up"
    else:
        stage = "down"

    if stage == "up":
        if left_knee_raised and l_knee[1] > l_hip[1] - 20:
            form = "warning"
            feedback.append((">> Ghutna aur upar uthao - hip level tak!", COLOR_YELLOW))
        if right_knee_raised and r_knee[1] > r_hip[1] - 20:
            form = "warning"
            feedback.append((">> Ghutna aur upar uthao - hip level tak!", COLOR_YELLOW))

    if l_shoulder[1] > l_hip[1] - 50:
        form = "bad"
        feedback.append((">> Seedha khade raho - jhukna mat!", COLOR_RED))

    if form == "good":
        feedback.append((">> Zabardast high knees! Speed badha do!", COLOR_GREEN))

    if left_knee_raised:
        cv2.circle(img, l_knee, 10, COLOR_GREEN, -1)
    if right_knee_raised:
        cv2.circle(img, r_knee, 10, COLOR_GREEN, -1)

    return reps, stage, form, feedback


# ── 10. LATERAL RAISES ──
def analyze_lateral_raises(landmarks, w, h, img, reps, stage):
    feedback = []
    form = "good"

    l_shoulder = get_lm(landmarks, LM_LEFT_SHOULDER, w, h)
    l_elbow = get_lm(landmarks, LM_LEFT_ELBOW, w, h)
    l_wrist = get_lm(landmarks, LM_LEFT_WRIST, w, h)
    l_hip = get_lm(landmarks, LM_LEFT_HIP, w, h)

    r_shoulder = get_lm(landmarks, LM_RIGHT_SHOULDER, w, h)
    r_elbow = get_lm(landmarks, LM_RIGHT_ELBOW, w, h)
    r_wrist = get_lm(landmarks, LM_RIGHT_WRIST, w, h)
    r_hip = get_lm(landmarks, LM_RIGHT_HIP, w, h)

    left_raise = calculate_angle(l_hip, l_shoulder, l_elbow)
    right_raise = calculate_angle(r_hip, r_shoulder, r_elbow)
    avg_raise = (left_raise + right_raise) / 2

    left_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
    right_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

    draw_angle_on_joint(img, left_raise, l_shoulder)
    draw_angle_on_joint(img, right_raise, r_shoulder)
    draw_angle_on_joint(img, left_elbow_angle, l_elbow, "good" if left_elbow_angle > 140 else "warning")

    # Rep counting
    if avg_raise > 75:
        if stage == "down":
            reps += 1
        stage = "up"
    elif avg_raise < 25:
        stage = "down"

    if stage == "up" and avg_raise > 105:
        form = "warning"
        feedback.append((">> Arms shoulder se upar mat le jao!", COLOR_YELLOW))

    if stage == "up" and avg_raise < 70:
        form = "warning"
        feedback.append((">> Arms aur upar uthao - shoulder level tak!", COLOR_YELLOW))

    raise_diff = abs(left_raise - right_raise)
    if raise_diff > 15:
        form = "warning"
        feedback.append((">> Dono arms ek saath uthao - symmetry rakho!", COLOR_YELLOW))

    avg_elbow = (left_elbow_angle + right_elbow_angle) / 2
    if avg_elbow < 130:
        form = "warning"
        feedback.append((">> Elbow zyada mat modo - thoda sa bend enough hai!", COLOR_YELLOW))

    if form == "good":
        feedback.append((">> Perfect lateral raise! Controlled movement!", COLOR_GREEN))

    return reps, stage, form, feedback


# ═══════════════════════════════════════════════════════════
# Exercise Registry
# ═══════════════════════════════════════════════════════════
EXERCISES = {
    1: {"name": "Bicep Curls", "func": analyze_bicep_curls, "emoji": "💪"},
    2: {"name": "Squats", "func": analyze_squats, "emoji": "🦵"},
    3: {"name": "Push-ups", "func": analyze_pushups, "emoji": "🫸"},
    4: {"name": "Lunges", "func": analyze_lunges, "emoji": "🏃"},
    5: {"name": "Shoulder Press", "func": analyze_shoulder_press, "emoji": "🏋️"},
    6: {"name": "Jumping Jacks", "func": analyze_jumping_jacks, "emoji": "⭐"},
    7: {"name": "Deadlift", "func": analyze_deadlift, "emoji": "🏗️"},
    8: {"name": "Plank", "func": analyze_plank, "emoji": "🧘"},
    9: {"name": "High Knees", "func": analyze_high_knees, "emoji": "🦿"},
    10: {"name": "Lateral Raises", "func": analyze_lateral_raises, "emoji": "🤸"},
}


# ═══════════════════════════════════════════════════════════
# Break Screen
# ═══════════════════════════════════════════════════════════
def show_break_screen(cap, break_duration, next_exercise_name):
    """Show a 30-second break countdown between exercises."""
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), COLOR_BG_DARK, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        elapsed = time.time() - start_time
        remaining = max(0, break_duration - elapsed)

        # Title
        title = "REST TIME"
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        cv2.putText(frame, title, (w // 2 - tw // 2, h // 2 - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_CYAN, 3, cv2.LINE_AA)

        # Countdown
        count_text = f"{int(remaining)}"
        (cw, ch), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 4.0, 5)
        count_color = COLOR_GREEN if remaining > 10 else (COLOR_YELLOW if remaining > 5 else COLOR_RED)
        cv2.putText(frame, count_text, (w // 2 - cw // 2, h // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, count_color, 5, cv2.LINE_AA)

        # Next exercise info
        next_text = f"Next: {next_exercise_name}"
        (nw, _), _ = cv2.getTextSize(next_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(frame, next_text, (w // 2 - nw // 2, h // 2 + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_GOLD, 2, cv2.LINE_AA)

        # Motivational text
        motiv = "Saans lo... agla round aane wala hai!"
        (mw, _), _ = cv2.getTextSize(motiv, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(frame, motiv, (w // 2 - mw // 2, h // 2 + 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2, cv2.LINE_AA)

        # Progress bar
        progress = elapsed / break_duration
        draw_progress_bar(frame, 50, h - 50, w - 100, 15, progress, COLOR_CYAN)

        cv2.imshow("Exercise Helper", frame)

        if remaining <= 0:
            break

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            return False
        if key == ord('s'):
            return True

    return True


# ═══════════════════════════════════════════════════════════
# Summary Screen
# ═══════════════════════════════════════════════════════════
def show_summary_screen(cap, results):
    """Show final workout summary."""
    while True:
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        frame = cv2.flip(frame, 1) if ret else frame
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), COLOR_BG_DARK, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        title = "WORKOUT COMPLETE!"
        (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.putText(frame, title, (w // 2 - tw // 2, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_GOLD, 3, cv2.LINE_AA)

        y_start = 120
        for i, (name, reps) in enumerate(results):
            y = y_start + i * 45
            cv2.putText(frame, f"{i + 1}. {name}", (50, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2, cv2.LINE_AA)
            cv2.putText(frame, f"Reps: {reps}", (400, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_CYAN, 2, cv2.LINE_AA)

        total = sum(r for _, r in results)
        cv2.putText(frame, f"Total Reps: {total}", (50, y_start + len(results) * 45 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_GREEN, 2, cv2.LINE_AA)

        cv2.putText(frame, "Press 'Q' to exit | Bahut accha kaam kiya!", (50, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_PURPLE, 2, cv2.LINE_AA)

        cv2.imshow("Exercise Helper", frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            break


# ═══════════════════════════════════════════════════════════
# Countdown Before Exercise Starts
# ═══════════════════════════════════════════════════════════
def show_countdown(cap, exercise_name, countdown=5):
    """Show a GET READY countdown before each exercise."""
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        elapsed = time.time() - start
        remaining = max(0, countdown - elapsed)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), COLOR_BG_DARK, -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        (tw, _), _ = cv2.getTextSize(exercise_name, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.putText(frame, exercise_name, (w // 2 - tw // 2, h // 2 - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_GOLD, 3, cv2.LINE_AA)

        ready_text = "GET READY!"
        (rw, _), _ = cv2.getTextSize(ready_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(frame, ready_text, (w // 2 - rw // 2, h // 2 - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 2, cv2.LINE_AA)

        if remaining <= 0:
            count_text = "GO!"
            count_color = COLOR_GREEN
        else:
            count_text = str(int(remaining) + 1)
            count_color = COLOR_CYAN

        (cw, _), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 4.0, 5)
        cv2.putText(frame, count_text, (w // 2 - cw // 2, h // 2 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, count_color, 5, cv2.LINE_AA)

        pos_text = "Camera ke saamne position lo!"
        (pw, _), _ = cv2.getTextSize(pos_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(frame, pos_text, (w // 2 - pw // 2, h // 2 + 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 2, cv2.LINE_AA)

        cv2.imshow("Exercise Helper", frame)

        if remaining <= -1:
            break

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            return False

    return True


# ═══════════════════════════════════════════════════════════
# Main Exercise Loop (VIDEO mode — synchronous per-frame)
# ═══════════════════════════════════════════════════════════
def run_exercise(cap, exercise_info, duration):
    """Run a single exercise session. Returns rep count."""
    exercise_name = exercise_info["name"]
    analyze_func = exercise_info["func"]

    reps = 0
    stage = "up"
    start_time = time.time()

    # Create PoseLandmarker in VIDEO mode (synchronous per-frame)
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_timestamp_ms = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Webcam read failed!")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            elapsed = time.time() - start_time
            remaining = max(0, duration - elapsed)

            if remaining <= 0:
                break

            # Convert to MediaPipe Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect pose (VIDEO mode — synchronous)
            frame_timestamp_ms += 33  # ~30 FPS
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            form_score = "good"
            feedback_list = []

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]  # First person

                # Analyze the exercise
                reps, stage, form_score, feedback_list = analyze_func(
                    landmarks, w, h, frame, reps, stage
                )

                # Draw skeleton
                if form_score == "good":
                    skel_color = COLOR_GREEN
                elif form_score == "warning":
                    skel_color = COLOR_YELLOW
                else:
                    skel_color = COLOR_RED

                draw_skeleton(frame, landmarks, w, h, skel_color)
            else:
                feedback_list = [(">> Body detect nahi ho rahi - camera me aao!", COLOR_RED)]
                form_score = "bad"

            # Draw HUD
            draw_hud(frame, exercise_name, reps, remaining, duration, feedback_list, form_score)

            # Controls help
            cv2.putText(frame, "Q=Quit | S=Skip", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

            cv2.imshow("Exercise Helper", frame)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q') or key == 27:
                return -1
            if key == ord('s'):
                break

    return reps


# ═══════════════════════════════════════════════════════════
# Menu System
# ═══════════════════════════════════════════════════════════
def show_menu():
    """Display exercise selection menu in terminal."""
    print("\n" + "=" * 55)
    print("   EXERCISE HELPER MODEL — Real-Time Trainer")
    print("=" * 55)
    print()
    for key, ex in EXERCISES.items():
        print(f"    {key:2d}. {ex['emoji']}  {ex['name']}")
    print()
    print("    11.   ALL EXERCISES (Full Workout)")
    print("     0.   Exit")
    print()
    print("=" * 55)

    while True:
        try:
            choice = input("\n  Kaunsi exercise karni hai? (number daalo): ").strip()
            if choice == "0":
                return []
            elif choice == "11":
                return list(EXERCISES.keys())
            else:
                choices = [int(c.strip()) for c in choice.split(",")]
                valid = all(c in EXERCISES for c in choices)
                if valid and len(choices) > 0:
                    return choices
                else:
                    print("  Galat number! Dobara try karo.")
        except ValueError:
            print("  Sirf number daalo! Comma se multiple choose karo (e.g., 1,2,5)")


# ═══════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════
def main():
    # Check model file
    if not os.path.exists(MODEL_PATH):
        print(f"\n  ERROR: Model file not found at: {MODEL_PATH}")
        print("  Download it first:")
        print("  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task")
        print("  Save as 'pose_landmarker.task' in the same folder as this script.")
        sys.exit(1)

    print("\n  Webcam start ho raha hai...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ERROR: Webcam open nahi ho raha! Camera check karo.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        while True:
            selected = show_menu()

            if not selected:
                print("\n  Bye bye! Exercise karte raho!")
                break

            selected_names = [EXERCISES[s]["name"] for s in selected]
            print(f"\n  Selected: {', '.join(selected_names)}")
            print(f"  Har exercise: {EXERCISE_DURATION}s | Break: {BREAK_DURATION}s")
            print(f"  Webcam window me dekho... Camera ke saamne position lo!")

            results = []

            for i, ex_id in enumerate(selected):
                exercise = EXERCISES[ex_id]
                print(f"\n  Starting: {exercise['name']}...")

                if not show_countdown(cap, exercise["name"]):
                    break

                rep_count = run_exercise(cap, exercise, EXERCISE_DURATION)

                if rep_count == -1:
                    break

                results.append((exercise["name"], rep_count))
                print(f"  {exercise['name']} done! Reps: {rep_count}")

                if i < len(selected) - 1:
                    next_ex = EXERCISES[selected[i + 1]]["name"]
                    print(f"  {BREAK_DURATION}s break... Next: {next_ex}")
                    cont = show_break_screen(cap, BREAK_DURATION, next_ex)
                    if not cont:
                        break

            if results:
                print("\n  Workout Summary:")
                for name, reps in results:
                    print(f"     {name}: {reps} reps")
                total = sum(r for _, r in results)
                print(f"     Total: {total} reps")
                print(f"\n  Bahut accha! Workout complete!")

                show_summary_screen(cap, results)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n  Camera band ho gaya. Program exit.")


if __name__ == "__main__":
    main()
