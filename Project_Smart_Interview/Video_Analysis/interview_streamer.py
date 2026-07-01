"""
interview_streamer.py
─────────────────────────────────────────────────────────────
Generator function stream_interview(duration_minutes) that:

  1. Opens the webcam ONCE
  2. Runs ALL analysis inline (eye, head, hands, emotion, face-touch)
  3. Records audio in a background thread
  4. Yields MJPEG frames to Flask (/video_feed)
  5. When duration ends → saves Video_Report.txt, runs Voice_classifier.py,
     merges both → writes Report_Smart_Interview.txt

Only ONE process touches the camera — no subprocess conflicts.
"""

import os
import sys
import time
import queue
import shutil
import threading
import subprocess
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────────
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_VOICE_DIR  = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, "Voice_Analysis"))
_REPORT_OUT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, "Report_Smart_Interview.txt"))

# Make sure our own directory is importable
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# TF env-vars must be set before TF/mediapipe imports
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


# ── Lazy-import heavy analysis modules ─────────────────────────────────────────
def _load_analysis_modules():
    """Import heavy CV/ML modules once, return them in a dict."""
    import cv2
    import mediapipe as mp
    from eye        import detect_eye,       EyeGazeTracker
    from head       import detect_head_pose, HeadPoseTracker
    from hands      import detect_hands
    from emotion    import detect_emotion
    from face_touch import FaceTouchAnalyzer
    from report_generator import ReportGenerator

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hands_det = mp.solutions.hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return {
        "cv2": cv2,
        "mp": mp,
        "face_mesh": face_mesh,
        "hands_det": hands_det,
        "mp_draw": mp.solutions.drawing_utils,
        "mp_hands": mp.solutions.hands,
        "detect_eye": detect_eye,
        "EyeGazeTracker": EyeGazeTracker,
        "detect_head_pose": detect_head_pose,
        "HeadPoseTracker": HeadPoseTracker,
        "detect_hands": detect_hands,
        "detect_emotion": detect_emotion,
        "FaceTouchAnalyzer": FaceTouchAnalyzer,
        "ReportGenerator": ReportGenerator,
    }


# ── Audio helpers ──────────────────────────────────────────────────────────────
def _start_audio(audio_path: str):
    """Start sounddevice recording.  Returns (stream, thread, flag_list) or None."""
    try:
        import sounddevice as sd
        import soundfile as sf

        samplerate, channels = 16000, 1
        q: queue.Queue = queue.Queue()
        flag = [True]          # mutable flag: flag[0] = is_recording

        def _cb(indata, frames, t, status):
            if status:
                print(f"[audio] {status}", flush=True)
            q.put(indata.copy())

        stream = sd.InputStream(samplerate=samplerate, channels=channels, callback=_cb)
        stream.start()

        def _writer():
            with sf.SoundFile(audio_path, mode="w",
                              samplerate=samplerate, channels=channels) as fh:
                while flag[0] or not q.empty():
                    try:
                        fh.write(q.get(timeout=0.1))
                    except queue.Empty:
                        pass

        t = threading.Thread(target=_writer, daemon=True)
        t.start()
        print("[interview_streamer] Audio recording started", flush=True)
        return stream, t, flag

    except Exception as exc:
        print(f"[interview_streamer] Audio unavailable: {exc}", flush=True)
        return None


def _stop_audio(audio_info):
    if audio_info is None:
        return
    stream, t, flag = audio_info
    flag[0] = False
    try:
        stream.stop()
        stream.close()
    except Exception:
        pass
    t.join(timeout=10)
    print("[interview_streamer] Audio recording stopped", flush=True)


# ── Voice analysis (subprocess, non-blocking) ──────────────────────────────────
def _run_voice_and_merge(audio_path: str, video_report_text: str,
                         reporter, report_out_path: str,
                         num_speakers: int | None = None) -> str:
    """Run Voice_classifier.py and merge its output into the final report."""
    merged = video_report_text
    voice_text = ""

    if os.path.exists(audio_path):
        print("[interview_streamer] Running voice analysis…", flush=True)
        orig_cwd = os.getcwd()
        try:
            dest_audio = os.path.join(_VOICE_DIR, "interview_audio.wav")
            shutil.copy(audio_path, dest_audio)
            os.chdir(_VOICE_DIR)

            cmd = [sys.executable, "Voice_classifier.py", os.path.abspath(dest_audio)]
            if num_speakers is not None:
                cmd.append(str(num_speakers))

            subprocess.run(
                cmd,
                check=True,
                timeout=300,
            )

            voice_report = os.path.join(_VOICE_DIR, "Classification_Report.md")
            if os.path.exists(voice_report):
                with open(voice_report, "r", encoding="utf-8") as fh:
                    voice_text = fh.read()
                merged += (
                    "\n\n" + "=" * 50 + "\n"
                    "VOICE CLASSIFICATION REPORT\n"
                    + "=" * 50 + "\n"
                    + voice_text
                )
                print("[interview_streamer] Voice analysis merged.", flush=True)
        except Exception as exc:
            print(f"[interview_streamer] Voice analysis failed: {exc}", flush=True)
        finally:
            os.chdir(orig_cwd)

    reporter.save(merged, path=report_out_path)
    print(f"[interview_streamer] Final report → {report_out_path}", flush=True)
    return voice_text


# ── Overlay helpers ────────────────────────────────────────────────────────────
def _draw_hud(cv2, frame, elapsed, total, eye_dir, head_dir, emotion):
    h, w = frame.shape[:2]
    remaining = max(0, total - int(elapsed))
    m, s = divmod(remaining, 60)
    pct  = min(elapsed / max(total, 1), 1.0)

    # Top dark bar
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 50), (15, 15, 25), -1)
    cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)

    # REC dot (blinks every second)
    if int(elapsed) % 2 == 0:
        cv2.circle(frame, (16, 25), 7, (0, 0, 220), -1)
    cv2.putText(frame, "REC", (28, 31),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)

    # Timer
    timer_str = f"{m:02d}:{s:02d}"
    cv2.putText(frame, timer_str, (w // 2 - 38, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (130, 200, 255), 2)

    # AI Active
    cv2.putText(frame, "AI Active", (w - 92, 31),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 80), 1)

    # Bottom progress bar
    cv2.rectangle(frame, (0, h - 5), (w, h), (40, 40, 60), -1)
    cv2.rectangle(frame, (0, h - 5), (int(w * pct), h), (99, 102, 241), -1)


def _placeholder(cv2, message=""):
    import numpy as np
    frame = np.zeros((480, 640, 3), dtype="uint8")
    frame[:] = (20, 20, 30)
    if message:
        f = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(message, f, 0.65, 2)
        cv2.putText(frame, message, ((640 - tw) // 2, (480 + th) // 2),
                    f, 0.65, (180, 180, 200), 2)
    return frame


def _encode(cv2, frame, quality=80):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"


# ── Main generator ─────────────────────────────────────────────────────────────
def stream_interview(duration_minutes: float = 2.0, num_speakers: int | None = None, language: str = "English"):
    """
    Yields MJPEG frames while performing full interview analysis.
    Writes Report_Smart_Interview.txt when done.
    """
    duration_seconds = int(duration_minutes * 60)
    audio_path = os.path.join(_THIS_DIR, "interview_audio.wav")

    # ── 1. Remove stale report ─────────────────────────────────────────────────
    for old in [_REPORT_OUT, os.path.join(_THIS_DIR, "Video_Report.txt")]:
        try:
            if os.path.exists(old):
                os.remove(old)
        except OSError:
            pass

    # ── 2. Load analysis modules ───────────────────────────────────────────────
    try:
        m = _load_analysis_modules()
    except Exception as exc:
        import cv2 as _cv2
        frame = _placeholder(_cv2, f"Import error: {exc}")
        yield _encode(_cv2, frame)
        return

    cv2 = m["cv2"]

    # ── 3. Open camera ─────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        frame = _placeholder(cv2, "Camera not available")
        yield _encode(cv2, frame)
        return

    # ── 4. Start audio ─────────────────────────────────────────────────────────
    audio_info = _start_audio(audio_path)

    # ── 5. Analysis state ──────────────────────────────────────────────────────
    eye_tracker       = m["EyeGazeTracker"]()
    head_tracker      = m["HeadPoseTracker"]()
    face_touch_a      = m["FaceTouchAnalyzer"](fps=30)
    reporter          = m["ReportGenerator"]()

    eye_stats         = defaultdict(int)
    head_stats        = defaultdict(int)
    emotion_stats     = defaultdict(int)
    hand_counts       = {"No Hands": 0, "One Hand": 0, "Two Hands": 0}
    hand_move_total   = 0.0
    hand_move_samples = 0
    hand_act_status   = "N/A"

    last_touch        = None
    emotion           = "Neutral"
    emotion_data      = None
    frame_counter     = 0

    face_mesh  = m["face_mesh"]
    hands_det  = m["hands_det"]
    mp_draw    = m["mp_draw"]
    mp_hands   = m["mp_hands"]

    session_start = time.time()
    print(f"[interview_streamer] Interview started — {duration_seconds}s", flush=True)

    # ── 6. Main capture / analysis loop ───────────────────────────────────────
    # We store collected data here so the finally block can use it
    _data = {}

    try:
        while True:
            elapsed = time.time() - session_start
            if elapsed >= duration_seconds:
                break

            ok, frame = cap.read()
            if not ok:
                time.sleep(0.04)
                continue

            frame_counter += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ── Face & hand detection ──────────────────────────────────────────
            face_res = face_mesh.process(rgb)
            hand_res = hands_det.process(rgb)

            face_lm = (
                face_res.multi_face_landmarks[0]
                if face_res.multi_face_landmarks else None
            )

            # Draw hand landmarks
            if hand_res.multi_hand_landmarks:
                for hl in hand_res.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

            # ── Eye ───────────────────────────────────────────────────────────
            eye_dir = "N/A"
            head_dir = "N/A"
            if face_lm:
                h_px, w_px, _ = frame.shape
                ed = m["detect_eye"](face_lm, w_px, h_px, tracker=eye_tracker)
                eye_dir = ed["direction"]
                eye_stats[eye_dir] += 1

                hd = m["detect_head_pose"](face_lm, w_px, h_px, tracker=head_tracker)
                head_dir = hd["direction"]
                head_stats[head_dir] += 1

            # ── Emotion (every 40 frames) ──────────────────────────────────────
            if face_lm is not None and frame_counter % 40 == 0:
                emotion_data = m["detect_emotion"](frame, face_lm)
            if emotion_data is not None:
                emotion = emotion_data.get("emotion", emotion)
            if frame_counter % 20 == 0 and emotion:
                emotion_stats[emotion] += 1

            # ── Hands ──────────────────────────────────────────────────────────
            hand_d = m["detect_hands"](hand_res, frame.shape)
            vis = hand_d.get("visibility", "No Hands")
            hand_counts[vis] = hand_counts.get(vis, 0) + 1
            mv = hand_d.get("avg_movement", 0.0)
            if mv > 0:
                hand_move_total   += float(mv)
                hand_move_samples += 1
            hand_act_status = hand_d.get("status", hand_act_status)

            # ── Face touch (every 2 frames) ────────────────────────────────────
            if frame_counter % 2 == 0:
                try:
                    last_touch = face_touch_a.update(
                        face_lm, hand_res, frame, time.time()
                    )
                except Exception:
                    pass

            # ── HUD ────────────────────────────────────────────────────────────
            _draw_hud(cv2, frame, elapsed, duration_seconds,
                      eye_dir, head_dir, emotion)

            # ── Yield MJPEG frame ──────────────────────────────────────────────
            chunk = _encode(cv2, frame)
            if chunk:
                yield chunk

            time.sleep(0.04)   # ~25 fps

    except GeneratorExit:
        # Browser disconnected — still save the report
        pass

    finally:
        cap.release()
        face_mesh.close()
        hands_det.close()
        _stop_audio(audio_info)

        # Collect all data
        _data = {
            "duration_sec":       int(time.time() - session_start),
            "session_start":      session_start,
            "eye_stats":          dict(eye_stats),
            "head_stats":         dict(head_stats),
            "hand_counts":        hand_counts,
            "hand_avg_movement":  (
                hand_move_total / hand_move_samples
                if hand_move_samples > 0 else 0.0
            ),
            "hand_activity_status": hand_act_status,
            "emotion_stats":      dict(emotion_stats),
            "face_touch":            last_touch or {},
        }
        print(f"[interview_streamer] Captured {frame_counter} frames in "
              f"{_data['duration_sec']}s", flush=True)

        # ── Generate Video Report ──────────────────────────────────────────────
        english_video_report_txt = reporter.generate(_data, language="English")
        video_report_path = os.path.join(_THIS_DIR, "Video_Report.txt")
        reporter.save(english_video_report_txt, path=video_report_path)
        print(f"[interview_streamer] Video report → {video_report_path}", flush=True)

        # ── Generate localized report copy if needed ───────────────────────────
        localized_report_path = None
        if language == "Arabic":
            arabic_video_report_txt = reporter.generate(_data, language="Arabic")
            localized_report_path = os.path.join(_THIS_DIR, "Report_Smart_Interview_Arabic.txt")
            reporter.save(arabic_video_report_txt, path=localized_report_path)
            print(f"[interview_streamer] Localized Arabic report → {localized_report_path}", flush=True)

        # ── Run Voice Analysis + merge in a background thread ─────────────────
        def _bg_report():
            report_text = arabic_video_report_txt if language == "Arabic" else english_video_report_txt
            voice_text = _run_voice_and_merge(
                audio_path, report_text,
                reporter, _REPORT_OUT,
                num_speakers=num_speakers
            )
            if localized_report_path and voice_text:
                with open(localized_report_path, "a", encoding="utf-8") as loc_fh:
                    loc_fh.write("\n\n" + "=" * 50 + "\n")
                    loc_fh.write("تقرير تحليل الصوت\n" if language == "Arabic" else "VOICE CLASSIFICATION REPORT\n")
                    loc_fh.write("=" * 50 + "\n")
                    loc_fh.write(voice_text)

        t = threading.Thread(target=_bg_report, daemon=False)
        t.start()
        print("[interview_streamer] Report generation thread started.", flush=True)
