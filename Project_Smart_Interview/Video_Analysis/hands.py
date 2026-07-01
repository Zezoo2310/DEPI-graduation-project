import numpy as np
from collections import deque

prev_centers = [None, None]

movement_history = deque(maxlen=50)

one_hand_frames = 0
two_hand_frames = 0
no_hand_frames = 0


def detect_hands(results, frame_shape):

    global prev_centers
    global movement_history
    global one_hand_frames
    global two_hand_frames
    global no_hand_frames

    h, w = frame_shape[:2]

    hands_count = 0

    hand_visibility = "No Hands"
    one_hand_percent = 0
    two_hand_percent = 0

    status = "Too Static"
    hand_score = 40
    avg_movement = 0.0

    if results.multi_hand_landmarks:

        hands_count = len(results.multi_hand_landmarks)

        for hand_idx, hand_landmarks in enumerate(
            results.multi_hand_landmarks
        ):

            points = []

            for lm in hand_landmarks.landmark:

                x = int(lm.x * w)
                y = int(lm.y * h)

                points.append((x, y))

            center = np.mean(points, axis=0)

            if hand_idx < 2:

                if prev_centers[hand_idx] is not None:

                    movement = np.linalg.norm(
                        np.array(center)
                        -
                        np.array(prev_centers[hand_idx])
                    )

                    movement_history.append(movement)

                prev_centers[hand_idx] = center

    # =====================
    # Hand Visibility
    # =====================

    if hands_count == 0:

        hand_visibility = "No Hands"
        no_hand_frames += 1

    elif hands_count == 1:

        hand_visibility = "One Hand"
        one_hand_frames += 1

    else:

        hand_visibility = "Two Hands"
        two_hand_frames += 1

    # =====================
    # Usage Percentages
    # =====================

    total_frames = (
        one_hand_frames +
        two_hand_frames +
        no_hand_frames
    )

    if total_frames > 0:

        one_hand_percent = (
            one_hand_frames /
            total_frames * 100
        )

        two_hand_percent = (
            two_hand_frames /
            total_frames * 100
        )

    # =====================
    # Hand Activity
    # =====================

    if len(movement_history) > 10:

        avg_movement = np.mean(movement_history)

        if avg_movement < 5:

            status = "Too Static"
            hand_score = 40

        elif avg_movement < 20:

            status = "Normal"
            hand_score = 90

        elif avg_movement < 35:

            status = "Acceptable"
            hand_score = 70

        else:

            status = "Overactive"
            hand_score = 40

    return {

        "hands_count": hands_count,

        "visibility": hand_visibility,

        "one_hand_percent":
            round(one_hand_percent, 1),

        "two_hand_percent":
            round(two_hand_percent, 1),

        "avg_movement":
            round(float(avg_movement), 1),

        "status": status,

        "score": hand_score
    }