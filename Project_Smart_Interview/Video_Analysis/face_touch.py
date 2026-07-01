import time
import math
from collections import deque

# Face touch analyzer using MediaPipe Face Mesh + Hands
# Lightweight, stateful tracker that counts touches, durations, and events.

# Landmark index sets (MediaPipe Face Mesh)
MOUTH_IDX = [13, 14, 78, 308, 61, 291]
FOREHEAD_IDX = [10, 338, 297, 337, 9, 151]
CHIN_IDX = [152]


class FaceTouchAnalyzer:
    def __init__(self, fps=30):
        self.fps = fps

        self.face_touch_count = 0
        self.face_touch_duration = 0.0
        self.long_face_touch_events = 0

        self.mouth_cover_count = 0
        self.forehead_touch_count = 0
        self.head_support_events = 0

        self._touching = False
        self._touch_start_ts = None
        self._touch_frame_count = 0
        self._long_touch_recorded = False

        self._mouth_covering = False
        self._mouth_frame_count = 0

        self._forehead_touching = False
        self._forehead_frame_count = 0

        self._head_supporting = False
        self._head_support_start = None
        self._head_support_frame_count = 0
        self._head_support_recorded = False

    def _landmark_px(self, lm, w, h):
        return (int(lm.x * w), int(lm.y * h))

    def _min_dist(self, points_a, points_b):
        md = None
        for ax, ay in points_a:
            for bx, by in points_b:
                d = (ax - bx) ** 2 + (ay - by) ** 2
                if md is None or d < md:
                    md = d
        return math.sqrt(md) if md is not None else float('inf')

    def update(self, face_landmarks, hand_results, frame, ts=None):
        if ts is None:
            ts = time.time()

        h, w = frame.shape[:2]
        touching = False
        mouth_cover = False
        forehead_touch = False
        head_support = False

        face_points = []
        mouth_points = []
        forehead_points = []
        chin_points = []

        if face_landmarks is not None:
            for idx, lm in enumerate(face_landmarks.landmark):
                face_points.append(self._landmark_px(lm, w, h))
            for i in MOUTH_IDX:
                if i < len(face_landmarks.landmark):
                    mouth_points.append(self._landmark_px(face_landmarks.landmark[i], w, h))
            for i in FOREHEAD_IDX:
                if i < len(face_landmarks.landmark):
                    forehead_points.append(self._landmark_px(face_landmarks.landmark[i], w, h))
            for i in CHIN_IDX:
                if i < len(face_landmarks.landmark):
                    chin_points.append(self._landmark_px(face_landmarks.landmark[i], w, h))

        hands_points = []
        wrist_points = []
        if hand_results and getattr(hand_results, 'multi_hand_landmarks', None):
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    hands_points.append(self._landmark_px(lm, w, h))
                try:
                    wrist_points.append(self._landmark_px(hand_landmarks.landmark[0], w, h))
                except Exception:
                    pass

        face_scale = 200
        if face_points:
            xs = [p[0] for p in face_points]
            face_scale = max(1, max(xs) - min(xs))

        touch_threshold = max(50, face_scale * 0.25)
        mouth_threshold = max(45, face_scale * 0.20)
        forehead_threshold = max(50, face_scale * 0.25)
        head_support_threshold = max(50, face_scale * 0.20)

        if hands_points and face_points:
            d_face = self._min_dist(hands_points, face_points)
            if d_face <= touch_threshold:
                touching = True

            if mouth_points:
                d_mouth = self._min_dist(hands_points, mouth_points)
                if d_mouth <= mouth_threshold:
                    mouth_cover = True

            if forehead_points:
                d_fore = self._min_dist(hands_points, forehead_points)
                if d_fore <= forehead_threshold:
                    forehead_touch = True

            if chin_points and hands_points:
                d_ws = self._min_dist(hands_points, chin_points)
                if d_ws <= head_support_threshold:
                    head_support = True

        if touching:
            self._touch_frame_count += 1
        else:
            self._touch_frame_count = 0

        is_touch_confirmed = self._touch_frame_count >= 3

        if is_touch_confirmed and not self._touching:
            self.face_touch_count += 1
            self._touch_start_ts = ts
            self._touching = True
            self._long_touch_recorded = False
        elif not is_touch_confirmed and self._touching:
            if self._touch_start_ts is not None:
                dur = ts - self._touch_start_ts
                self.face_touch_duration += dur
                if dur >= 2.0 and not self._long_touch_recorded:
                    self.long_face_touch_events += 1
            self._touch_start_ts = None
            self._touching = False
            self._long_touch_recorded = False

        if self._touching and self._touch_start_ts is not None:
            current_duration = ts - self._touch_start_ts
            self.face_touch_duration += max(0.0, current_duration)
            if current_duration >= 2.0 and not self._long_touch_recorded:
                self.long_face_touch_events += 1
                self._long_touch_recorded = True
            self._touch_start_ts = ts

        if mouth_cover:
            self._mouth_frame_count += 1
        else:
            self._mouth_frame_count = 0

        mouth_confirmed = self._mouth_frame_count >= 3
        if mouth_confirmed and not self._mouth_covering:
            self.mouth_cover_count += 1
            self._mouth_covering = True
        elif not mouth_confirmed and self._mouth_covering:
            self._mouth_covering = False

        if forehead_touch:
            self._forehead_frame_count += 1
        else:
            self._forehead_frame_count = 0

        forehead_confirmed = self._forehead_frame_count >= 3
        if forehead_confirmed and not self._forehead_touching:
            self.forehead_touch_count += 1
            self._forehead_touching = True
        elif not forehead_confirmed and self._forehead_touching:
            self._forehead_touching = False

        if head_support:
            self._head_support_frame_count += 1
        else:
            self._head_support_frame_count = 0

        head_support_confirmed = self._head_support_frame_count >= 3
        if head_support_confirmed and not self._head_supporting:
            self._head_support_start = ts
            self._head_supporting = True
            self._head_support_recorded = False
        elif not head_support_confirmed and self._head_supporting:
            self._head_support_start = None
            self._head_supporting = False
            self._head_support_recorded = False

        if self._head_supporting and self._head_support_start is not None and not self._head_support_recorded:
            if ts - self._head_support_start >= 3.0:
                self.head_support_events += 1
                self._head_support_recorded = True

        return {
            "found": self._touching,
            "face_touch_count": self.face_touch_count,
            "face_touch_duration": round(self.face_touch_duration, 2),
            "long_face_touch_events": self.long_face_touch_events,
            "mouth_cover_count": self.mouth_cover_count,
            "forehead_touch_count": self.forehead_touch_count,
            "head_support_events": self.head_support_events,
        }
