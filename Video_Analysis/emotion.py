import cv2
from deepface import DeepFace


def _crop_face(frame, face_landmarks):
    if face_landmarks is None:
        return None

    h, w = frame.shape[:2]
    xs = [int(lm.x * w) for lm in face_landmarks.landmark]
    ys = [int(lm.y * h) for lm in face_landmarks.landmark]
    if not xs or not ys:
        return None

    x1 = max(min(xs) - 10, 0)
    y1 = max(min(ys) - 10, 0)
    x2 = min(max(xs) + 10, w)
    y2 = min(max(ys) + 10, h)

    crop = frame[y1:y2, x1:x2]
    return crop if crop.size != 0 else None


def detect_emotion(frame, face_landmarks=None):
    try:
        cropped = _crop_face(frame, face_landmarks)
        target = cropped if cropped is not None else frame

        result = DeepFace.analyze(
            target,
            actions=["emotion"],
            enforce_detection=False,
            silent=True
        )

        if isinstance(result, list) and result:
            result = result[0]

        emotion = result.get("dominant_emotion", "Unknown")
        emotions = result.get("emotion", {})

        return {
            "emotion": emotion,
            "scores": emotions
        }
    except Exception:
        return {
            "emotion": "Unknown",
            "scores": {}
        }
