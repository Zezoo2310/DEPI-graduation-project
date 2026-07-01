import numpy as np
from collections import deque

# Iris landmarks
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]


def _compute_gaze_ratio(face_landmarks, w, h):
    left_eye_left = face_landmarks.landmark[33]
    left_eye_right = face_landmarks.landmark[133]
    lx1 = int(left_eye_left.x * w)
    lx2 = int(left_eye_right.x * w)

    left_iris = [
        (int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h))
        for idx in LEFT_IRIS
    ]
    left_center = np.mean(left_iris, axis=0)
    left_ratio = (left_center[0] - lx1) / max((lx2 - lx1), 1)

    right_eye_left = face_landmarks.landmark[362]
    right_eye_right = face_landmarks.landmark[263]
    rx1 = int(right_eye_left.x * w)
    rx2 = int(right_eye_right.x * w)

    right_iris = [
        (int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h))
        for idx in RIGHT_IRIS
    ]
    right_center = np.mean(right_iris, axis=0)
    right_ratio = (right_center[0] - rx1) / max((rx2 - rx1), 1)

    gaze_ratio = np.clip((left_ratio + right_ratio) / 2.0, 0.0, 1.0)
    return float(gaze_ratio)


class EyeGazeTracker:
    def __init__(self, history_len=10, stable_frames=3):
        self.history = deque(maxlen=history_len)
        self.current_direction = "CENTER"
        self._candidate_direction = "CENTER"
        self._candidate_count = 0
        self.stable_frames = stable_frames

    def update(self, gaze_ratio):
        self.history.append(gaze_ratio)
        average_ratio = float(np.mean(self.history))

        if average_ratio < 0.36:
            candidate = "LEFT"
        elif average_ratio > 0.64:
            candidate = "RIGHT"
        else:
            candidate = "CENTER"

        if candidate == self._candidate_direction:
            self._candidate_count += 1
        else:
            self._candidate_direction = candidate
            self._candidate_count = 1

        if self._candidate_count >= self.stable_frames:
            self.current_direction = candidate

        return {
            "direction": self.current_direction,
            "ratio": round(average_ratio, 2)
        }


def detect_eye(face_landmarks, w, h, tracker=None):
    gaze_ratio = _compute_gaze_ratio(face_landmarks, w, h)
    if tracker is not None:
        return tracker.update(gaze_ratio)

    if gaze_ratio < 0.36:
        direction = "LEFT"
    elif gaze_ratio > 0.64:
        direction = "RIGHT"
    else:
        direction = "CENTER"

    return {
        "direction": direction,
        "ratio": round(gaze_ratio, 2)
    }
