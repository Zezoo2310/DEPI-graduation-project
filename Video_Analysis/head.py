import cv2
import numpy as np
from collections import deque

HEAD_POINTS = [33, 263, 1, 61, 291, 199]


def _get_pose_angles(face_landmarks, img_w, img_h):
    face_2d = []
    face_3d = []

    for idx in HEAD_POINTS:
        lm = face_landmarks.landmark[idx]
        x = int(lm.x * img_w)
        y = int(lm.y * img_h)
        z = float(lm.z * max(img_w, img_h))
        face_2d.append([x, y])
        face_3d.append([x, y, z])

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    focal_length = img_w
    cam_matrix = np.array(
        [
            [focal_length, 0, img_w / 2],
            [0, focal_length, img_h / 2],
            [0, 0, 1]
        ],
        dtype=np.float64
    )
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, trans_vec = cv2.solvePnP(
        face_3d,
        face_2d,
        cam_matrix,
        dist_matrix,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return {"x_angle": 0.0, "y_angle": 0.0}

    rmat, _ = cv2.Rodrigues(rot_vec)
    pitch = np.arctan2(-rmat[2, 0], np.sqrt(rmat[2, 1]**2 + rmat[2, 2]**2))
    yaw = np.arctan2(rmat[1, 0], rmat[0, 0])

    return {
        "x_angle": float(np.degrees(pitch)),
        "y_angle": float(np.degrees(yaw))
    }


def _classify_direction(x_angle, y_angle):
    if y_angle <= -7:
        return "LEFT"
    if y_angle >= 7:
        return "RIGHT"
    if x_angle <= -6:
        return "DOWN"
    if x_angle >= 6:
        return "UP"
    return "FORWARD"


class HeadPoseTracker:
    def __init__(self, history_len=8, stable_frames=1):
        self.history = deque(maxlen=history_len)
        self.current_direction = "FORWARD"
        self._candidate_direction = "FORWARD"
        self._candidate_count = 0
        self.stable_frames = stable_frames

    def update(self, angles):
        self.history.append(angles)
        recent = list(self.history)[-3:]
        avg_x = float(np.mean([entry["x_angle"] for entry in recent]))
        avg_y = float(np.mean([entry["y_angle"] for entry in recent]))

        candidate = _classify_direction(avg_x, avg_y)

        if candidate == self._candidate_direction:
            self._candidate_count += 1
        else:
            self._candidate_direction = candidate
            self._candidate_count = 1

        if self._candidate_count >= self.stable_frames:
            self.current_direction = candidate

        return {
            "direction": self.current_direction,
            "x_angle": round(avg_x, 2),
            "y_angle": round(avg_y, 2),
            "candidate": candidate
        }


def detect_head_pose(face_landmarks, img_w, img_h, tracker=None):
    angles = _get_pose_angles(face_landmarks, img_w, img_h)
    if tracker is not None:
        return tracker.update(angles)

    direction = _classify_direction(angles["x_angle"], angles["y_angle"])
    return {
        "direction": direction,
        "x_angle": round(angles["x_angle"], 2),
        "y_angle": round(angles["y_angle"], 2)
    }
