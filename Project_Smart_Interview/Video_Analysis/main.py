import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import cv2
import mediapipe as mp
import time
from collections import defaultdict
import sys
import threading
import queue
import sounddevice as sd
import soundfile as sf
import shutil

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VOICE_PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, "Voice_Analysis"))
REPORT_DIR = CURRENT_DIR


from eye import detect_eye, EyeGazeTracker
from head import detect_head_pose, HeadPoseTracker
from hands import detect_hands
from emotion import detect_emotion
from face_touch import FaceTouchAnalyzer
from report_generator import ReportGenerator

# ==========================
# Mediapipe Setup
# 
# ==========================

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands_detector = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================
# Camera
# ==========================

cap = cv2.VideoCapture(0)

# ==========================
# Variables
# ==========================

frame_counter = 0
emotion = "neutral"

# Face touch analyzer and runtime stats
face_touch_analyzer = FaceTouchAnalyzer(fps=30)
last_touch_snapshot = None

# Session timer (seconds)
try:
    minutes = float(os.environ.get("INTERVIEW_DURATION_MINUTES", "2"))
except ValueError:
    minutes = 2.0
INTERVIEW_DURATION = int(minutes * 60)
session_start = time.time()

# Hand aggregation
hand_counts = { 'No Hands': 0, 'One Hand': 0, 'Two Hands': 0 }
hand_movement_total = 0.0
hand_movement_samples = 0
hand_activity_status = 'N/A'

# Report generator
reporter = ReportGenerator()

# Runtime trackers
eye_tracker = EyeGazeTracker()
head_tracker = HeadPoseTracker()

# Lightweight interview stats
eye_stats = defaultdict(int)
head_stats = defaultdict(int)
emotion_stats = defaultdict(int)


# ==========================
# Main Loop
# ==========================
# Audio Recording Setup
# ==========================
audio_filename = "interview_audio.wav"
samplerate = 16000
channels = 1
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

print("Initializing background audio recording...")
audio_stream = sd.InputStream(samplerate=samplerate, channels=channels, callback=audio_callback)
audio_stream.start()

is_recording = True
def record_audio_thread(filename):
    with sf.SoundFile(filename, mode='w', samplerate=samplerate, channels=channels) as file:
        while is_recording or not q.empty():
            try:
                data = q.get(timeout=0.1)
                file.write(data)
            except queue.Empty:
                pass

recording_thread = threading.Thread(target=record_audio_thread, args=(audio_filename,))
recording_thread.start()

frame_counter = 0
emotion = "Neutral"
emotion_data = None

while True:

    success, frame = cap.read()

    if not success:
        break

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Session timer
    elapsed = time.time() - session_start

    if elapsed >= INTERVIEW_DURATION:
        break

    face_results = face_mesh.process(rgb)

    hand_results = hands_detector.process(rgb)

    # ==========================

    # Draw Hands
    # ==========================

    if hand_results.multi_hand_landmarks:

        for hand_landmarks in hand_results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # ==========================
    # Face Analysis
    # ==========================

    eye_direction = "N/A"
    head_direction = "N/A"
    face_landmarks = face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None

    if face_landmarks:
        h, w, _ = frame.shape

        eye_data = detect_eye(
            face_landmarks,
            w,
            h,
            tracker=eye_tracker
        )
        eye_direction = eye_data["direction"]

        head_data = detect_head_pose(
            face_landmarks,
            w,
            h,
            tracker=head_tracker
        )
        head_direction = head_data["direction"]

        eye_stats[eye_direction] += 1
        head_stats[head_direction] += 1

    # ==========================
    # Emotion Analysis
    # ==========================

    frame_counter += 1

    if face_landmarks is not None and frame_counter % 40 == 0:

      emotion_data = detect_emotion(
        frame,
        face_landmarks
      )

    if emotion_data is not None:
                 emotion = emotion_data.get("emotion", emotion)
   

    # ==========================
    # Hand Analysis
    # ==========================

    hand_data = detect_hands(
        hand_results,
        frame.shape
    )

    # aggregate hand counts and movement
    vis = hand_data.get('visibility', 'No Hands')
    if vis not in hand_counts:
        hand_counts[vis] = 0
    hand_counts[vis] += 1
    mv = hand_data.get('avg_movement', 0.0)
    if mv > 0:
        hand_movement_total += float(mv)
        hand_movement_samples += 1
    hand_activity_status = hand_data.get('status', hand_activity_status)

    # ==========================
    # Display Results
    # ==========================

    # ==========================
    # Compact Info Panel (fixed size) - shows ONLY the requested 5 items
    # Time, Eye Direction, Head Direction, Emotion, Face Touch Count
    # Panel is fixed size, top-left, dark translucent rectangle, consistent font size
    # ==========================
    H, W = frame.shape[:2]
    panel_w = 360
    panel_h = 140
    panel_x = 20
    panel_y = 20
    padding = 12

    overlay = frame.copy()
    panel_color = (10, 10, 10)  # dark background (BGR)
    alpha = 0.65
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), panel_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Colors (BGR)
    COLOR_TEXT = (220, 220, 220)
    COLOR_VALUE = (200, 200, 200)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1

    # Build fixed 5 lines
    face_touch_count = (last_touch_snapshot or {}).get('face_touch_count', 0)

    panel_lines = [
        ("Time", f"{int(elapsed)}s/{INTERVIEW_DURATION}s"),
        ("Eye", eye_direction),
        ("Head", head_direction),
        ("Emotion", emotion),
        ("Face Touch Count", str(face_touch_count))
    ]

    # draw title
    title_y = panel_y + padding
    cv2.putText(frame, "Interview Info", (panel_x + padding, title_y + 6), font, scale, COLOR_TEXT, 2)

    # draw lines; right-align values inside panel
    line_start_y = panel_y + padding + 28
    line_h = 20
    for i, (label, val) in enumerate(panel_lines):
        y = line_start_y + i * line_h
        # label
        cv2.putText(frame, f"{label}", (panel_x + padding, y), font, scale, COLOR_TEXT, thickness)
        # value right-aligned
        (text_w, text_h), _ = cv2.getTextSize(val, font, scale, thickness)
        val_x = panel_x + panel_w - padding - text_w
        val_color = COLOR_VALUE
        cv2.putText(frame, val, (val_x, y), font, scale, val_color, 2)

    # ==========================
    # Face touch analysis (lightweight, cached every 2 frames)
    # ==========================
    if frame_counter % 2 == 0:
        try:
            last_touch_snapshot = face_touch_analyzer.update(
                face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None,
                hand_results,
                frame,
                time.time()
            )
        except Exception:
            last_touch_snapshot = last_touch_snapshot or {
                "found": False,
                "face_touch_count": 0,
                "face_touch_duration": 0.0,
                
            }

    # increment emotion stats when updated (every 20 frames)
    if frame_counter % 20 == 0 and emotion is not None:
        emotion_stats[emotion] += 1

    # Draw face touch metrics overlay (dynamic placement)
        # Draw face touch metrics overlay (dynamic placement)
    ft = last_touch_snapshot or {}
    H, W = frame.shape[:2]

    base_x = int(W * 0.03)
    base_y = int(H * 0.65)

    title_scale = 0.70
    small_scale = 0.60

    title_color = (200, 200, 200)
    value_color = (0, 255, 255)
    secondary_color = (255, 128, 0)

    cv2.putText(
        frame,
        "[ Face Touch ]",
        (base_x, base_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        title_scale,
        title_color,
        2
    )

    y = base_y + int(H * 0.04)

    cv2.putText(
        frame,
        f"Face Touch Count: {ft.get('face_touch_count',0)}",
        (base_x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        small_scale,
        value_color,
        2
    )

    y += int(H * 0.035)

    cv2.putText(
        frame,
        f"Face Touch Duration: {ft.get('face_touch_duration',0.0)}s",
        (base_x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        small_scale,
        value_color,
        2
    )

    y += int(H * 0.035)

    cv2.putText(
        frame,
        f"Mouth Cover Count: {ft.get('mouth_cover_count',0)}",
        (base_x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        small_scale,
        secondary_color,
        2
    )

    y += int(H * 0.035)

    cv2.putText(
        frame,
        f"Forehead Touch Count: {ft.get('forehead_touch_count',0)}",
        (base_x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        small_scale,
        secondary_color,
        2
    )

    y += int(H * 0.035)

    cv2.putText(
        frame,
        f"Head Support Events: {ft.get('head_support_events',0)}",
        (base_x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        small_scale,
        secondary_color,
        2
    )

    cv2.imshow(
        "Interview Analyzer",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break
# Stop recording
is_recording = False
audio_stream.stop()
audio_stream.close()
recording_thread.join()
print(f"Audio recorded successfully to {audio_filename}.")

# Build data dict and generate report

data = {
'duration_sec': int(time.time() - session_start),
'session_start': session_start,
'eye_stats': dict(eye_stats),
'head_stats': dict(head_stats),
'hand_counts': hand_counts,
'hand_avg_movement': (
hand_movement_total / hand_movement_samples
) if hand_movement_samples > 0 else 0.0,
'hand_activity_status': hand_activity_status,
'emotion_stats': dict(emotion_stats),
'face_touch': last_touch_snapshot or {}
}

report_text = reporter.generate(data)

# 1. Save Video Report Alone
video_report_path = os.path.join(REPORT_DIR, "Video_Report.txt")
reporter.save(report_text, path=video_report_path)
print(f"Video analysis report saved to {video_report_path}")

merged_report_text = report_text

# Run Voice Analysis Offline via Subprocess to avoid memory/dependency clashes
if os.path.exists(audio_filename):
    print("\nStarting offline voice analysis on the recorded audio...")
    original_cwd = os.getcwd()
    try:
        os.chdir(VOICE_PROJECT_DIR)
        shutil.copy(os.path.join(original_cwd, audio_filename), os.path.join(VOICE_PROJECT_DIR, audio_filename))
        
        print("Running voice classifier pipeline...")
        import subprocess
        abs_audio_path = os.path.abspath(audio_filename)
        subprocess.run([sys.executable, "Voice_classifier.py", abs_audio_path], check=True)
        
        report_path = os.path.join(VOICE_PROJECT_DIR, "Classification_Report.md")
        if os.path.exists(report_path):
            # 2. Voice report is generated inside VOICE_PROJECT_DIR (Voice_Analysis)
            # We just need to read it to merge it, no need to copy it to the root directory
            print(f"Voice analysis report saved to {report_path}")

            with open(report_path, 'r', encoding='utf-8') as f:
                voice_report_text = f.read()
            
            # Extract Voice Stats for PNG
            voice_stats = []
            speaker_blocks = voice_report_text.split('## 👤 ')[1:]
            for block in speaker_blocks:
                lines = block.strip().split('\n')
                speaker_name = lines[0].strip()
                for line in lines:
                    if line.startswith('| **') and '|' in line:
                        parts = [p.strip() for p in line.split('|')]
                        if len(parts) >= 4:
                            interval = parts[1].replace('**', '')
                            state = parts[2].replace('**', '')
                            voice_stats.append(f"{speaker_name} ({interval}): {state}")
            data['voice_stats'] = voice_stats
            
            # 3. Create Merged Report
            merged_report_text += "\n\n" + "="*50 + "\n"
            merged_report_text += "VOICE CLASSIFICATION REPORT\n"
            merged_report_text += "="*50 + "\n"
            merged_report_text += voice_report_text
            print("Voice analysis successfully merged into the final report.")
    except Exception as e:
        print(f"Voice analysis failed: {e}")
    finally:
        os.chdir(original_cwd)

# 4. Save Merged Report
report_out_path = os.path.join(CURRENT_DIR, os.pardir, "Report_Smart_Interview.txt")
reporter.save(merged_report_text, path=report_out_path)

try:
    png_out_path = os.path.join(CURRENT_DIR, os.pardir, "Report_Smart_Interview.png")
    reporter.save_png(data, path=png_out_path)
    print(f'Interview report written to {report_out_path} and {png_out_path}')

except Exception as e:
    print(f'Interview report written to {report_out_path}')
    print('Failed to write PNG report:', e)