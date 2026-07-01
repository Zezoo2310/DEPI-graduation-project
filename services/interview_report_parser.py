"""
interview_report_parser.py - Parse Interview Report Text Files

Reads the Report_Smart_Interview.txt file and converts it into
a structured Python dictionary for analysis and advice generation.
"""

import re


def parse_interview_report(text: str) -> dict:
    """
    Parse the interview report text into a structured dictionary.

    Args:
        text: Raw text content of the report file.

    Returns:
        Dictionary with all parsed sections.
    """
    report = {
        "session_duration": 0,
        "eye": {"CENTER": 0, "LEFT": 0, "RIGHT": 0},
        "eye_percentages": {"CENTER": 0, "LEFT": 0, "RIGHT": 0},
        "head": {"FORWARD": 0, "LEFT": 0, "RIGHT": 0, "UP": 0, "DOWN": 0},
        "head_percentages": {"FORWARD": 0, "LEFT": 0, "RIGHT": 0, "UP": 0, "DOWN": 0},
        "hands": {"No Hands": 0, "One Hand": 0, "Two Hands": 0},
        "hands_percentages": {"No Hands": 0, "One Hand": 0, "Two Hands": 0},
        "hand_avg_movement": 0.0,
        "hand_activity_status": "N/A",
        "emotions": {},
        "dominant_emotion": "N/A",
        "critical_events": [],
        "face_touch_count": 0,
        "face_touch_duration": 0.0,
        "mouth_cover_events": 0,
        "forehead_touch_events": 0,
        "head_support_events": 0,
        "interview_score": 0,
        "strengths": [],
        "needs_improvement": [],
        "final_assessment": "",
        "voice_analysis": [],
    }

    def startswith_any(line: str, prefixes: list[str]) -> bool:
        return any(line.strip().startswith(prefix) for prefix in prefixes)

    eye_keys = {
        "CENTER": "CENTER", "LEFT": "LEFT", "RIGHT": "RIGHT",
        "مركز": "CENTER", "يسار": "LEFT", "يمين": "RIGHT",
    }
    head_keys = {
        "FORWARD": "FORWARD", "LEFT": "LEFT", "RIGHT": "RIGHT", "UP": "UP", "DOWN": "DOWN",
        "أمام": "FORWARD", "يسار": "LEFT", "يمين": "RIGHT", "أعلى": "UP", "أسفل": "DOWN",
    }
    hand_keys = {
        "No Hands": "No Hands", "One Hand": "One Hand", "Two Hands": "Two Hands",
        "لا توجد يدين": "No Hands", "يد واحدة": "One Hand", "يدان": "Two Hands",
    }

    section = None
    lines = text.strip().split("\n")
    for line in lines:
        # Session Duration
        if report["session_duration"] == 0:
            m = re.search(r"(?:Session Duration|مدة الجلسة):\s*(\d+)", line)
            if m:
                report["session_duration"] = int(m.group(1))

        if startswith_any(line, ["Eye Analysis:", "تحليل العين:"]):
            section = "eye"
            continue
        if startswith_any(line, ["Head Pose Analysis:", "تحليل وضع الرأس:"]):
            section = "head"
            continue
        if startswith_any(line, ["Hand Analysis:", "تحليل اليدين:"]):
            section = "hands"
            continue
        if startswith_any(line, ["Emotion Analysis:", "تحليل العواطف:"]):
            section = "emotion"
            continue
        if startswith_any(line, ["Critical Events:", "الأحداث الحرجة:"]):
            section = "critical"
            continue
        if startswith_any(line, ["Face Touch Analysis:", "تحليل لمس الوجه:"]):
            section = "face_touch"
            continue
        if startswith_any(line, ["Interview Quality Score:", "تقييم جودة المقابلة:"]):
            section = "score"
            continue
        if startswith_any(line, ["Strengths:", "نقاط القوة:"]):
            section = "strengths"
            continue
        if startswith_any(line, ["Needs Improvement:", "نقاط التحسين:"]):
            section = "needs"
            continue
        if startswith_any(line, ["Final Assessment:", "التقييم النهائي:"]):
            section = "final"
            continue

        # Section parsing
        if section == "eye":
            m = re.match(r"\s*-\s*([^:]+):\s*(\d+)\s*\((\d+)%\)", line)
            if m:
                key = m.group(1).strip()
                if key in eye_keys:
                    report["eye"][eye_keys[key]] = int(m.group(2))
                    report["eye_percentages"][eye_keys[key]] = int(m.group(3))
                continue

        if section == "head":
            m = re.match(r"\s*-\s*([^:]+):\s*(\d+)\s*\((\d+)%\)", line)
            if m:
                key = m.group(1).strip()
                if key in head_keys:
                    report["head"][head_keys[key]] = int(m.group(2))
                    report["head_percentages"][head_keys[key]] = int(m.group(3))
                continue

        if section == "hands":
            m = re.match(r"\s*-\s*([^:]+):\s*(\d+)\s*[^\d]*\((\d+)%\)", line)
            if m:
                key = m.group(1).strip()
                if key in hand_keys:
                    report["hands"][hand_keys[key]] = int(m.group(2))
                    report["hands_percentages"][hand_keys[key]] = int(m.group(3))
                continue
            m = re.match(r"\s*-\s*(?:Average Hand Movement|متوسط حركة اليد):\s*([\d.]+)", line)
            if m:
                report["hand_avg_movement"] = float(m.group(1))
                continue
            m = re.match(r"\s*-\s*(?:Hand Activity Status|حالة نشاط اليد):\s*(.+)", line)
            if m:
                report["hand_activity_status"] = m.group(1).strip()
                continue

        if section == "emotion":
            m = re.match(r"\s*-\s*(?:Dominant Emotion|الشعور المسيطر):\s*(.+)", line)
            if m:
                report["dominant_emotion"] = m.group(1).strip()
                continue
            m = re.match(r"\s*-\s*(.+?):\s*(\d+)\s*\((\d+)%\)", line)
            if m:
                emotion_name = m.group(1).strip()
                report["emotions"][emotion_name] = {
                    "count": int(m.group(2)),
                    "percent": int(m.group(3)),
                }
                continue

        if section == "critical":
            if line.strip().startswith("-") and "None" not in line and "لا توجد" not in line:
                report["critical_events"].append(line.strip().lstrip("- ").strip())
                continue

        if section == "face_touch":
            m = re.match(r"\s*-\s*(?:Face Touch Count|عدد لمس الوجه):\s*(\d+)", line)
            if m:
                report["face_touch_count"] = int(m.group(1))
                continue
            m = re.match(r"\s*-\s*(?:Face Touch Duration|مدة لمس الوجه):\s*([\d.]+)", line)
            if m:
                report["face_touch_duration"] = float(m.group(1))
                continue
            m = re.match(r"\s*-\s*(?:Mouth Cover Events|عدد مرات تغطية الفم):\s*(\d+)", line)
            if m:
                report["mouth_cover_events"] = int(m.group(1))
                continue
            m = re.match(r"\s*-\s*(?:Forehead Touch Events|عدد مرات لمس الجبهة):\s*(\d+)", line)
            if m:
                report["forehead_touch_events"] = int(m.group(1))
                continue
            m = re.match(r"\s*-\s*(?:Head Support Events|أحداث دعم الرأس):\s*(\d+)", line)
            if m:
                report["head_support_events"] = int(m.group(1))
                continue

        if section == "score":
            m = re.match(r"\s*-\s*(?:Score|النقاط):\s*(\d+)/(?:100|100)", line)
            if m:
                report["interview_score"] = int(m.group(1))
                continue

        if section == "strengths":
            if line.strip().startswith("-"):
                report["strengths"].append(line.strip().lstrip("- ").strip())
                continue
            if line.strip() == "":
                section = None
                continue

        if section == "needs":
            if line.strip().startswith("-"):
                report["needs_improvement"].append(line.strip().lstrip("- ").strip())
                continue
            if line.strip() == "":
                section = None
                continue

        if section == "final":
            if line.strip().startswith("-"):
                report["final_assessment"] = line.strip().lstrip("- ").strip()
                continue

    # --- Voice Analysis ---
    voice_pattern = re.compile(
        r"\*\*(\d+s\s*-\s*\d+s)\*\*\s*\|\s*\*\*(\w+)\*\*\s*\|\s*(.+)"
    )
    current_speaker = None
    for line in lines:
        sm = re.match(r"##\s*(.*Speaker\d+)", line)
        if sm:
            current_speaker = sm.group(1).strip()
            continue
        vm = voice_pattern.search(line)
        if vm and current_speaker:
            report["voice_analysis"].append({
                "speaker": current_speaker,
                "interval": vm.group(1).strip(),
                "state": vm.group(2).strip(),
                "details": vm.group(3).strip(),
            })

    return report
