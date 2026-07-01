import math
from collections import Counter
from PIL import Image, ImageDraw, ImageFont


def _percent(part, total):
    if total == 0:
        return 0
    return round(part / total * 100)


def _rate_score(value, thresholds, weights):
    for threshold, score in zip(thresholds, weights):
        if value <= threshold:
            return score
    return weights[-1]


class ReportGenerator:
    def __init__(self):
        pass

    def generate(self, data: dict, language: str = "English") -> str:
        lang = "Arabic" if language == "Arabic" else "English"
        lines = []
        duration = int(data.get('duration_sec', 0))

        if lang == "Arabic":
            lines.append("تقرير المقابلة\n")
            lines.append(f"مدة الجلسة: {duration} ثانية\n\n")
            lines.append("تحليل العين:\n")
        else:
            lines.append("Interview Report\n")
            lines.append(f"Session Duration: {duration} sec\n\n")
            lines.append("Eye Analysis:\n")

        eye = data.get('eye_stats', {})
        total_eye = sum(eye.values())
        for key in ['CENTER', 'LEFT', 'RIGHT']:
            count = eye.get(key, 0)
            pct = _percent(count, total_eye)
            if lang == "Arabic":
                label = key if key != 'CENTER' else 'مركز'
                lines.append(f" - {label}: {count} ({pct}%)\n")
            else:
                lines.append(f" - {key}: {count} ({pct}%)\n")
        lines.append("\n")

        if lang == "Arabic":
            lines.append("تحليل وضع الرأس:\n")
        else:
            lines.append("Head Pose Analysis:\n")
        head = data.get('head_stats', {})
        total_head = sum(head.values())
        for key in ['FORWARD', 'LEFT', 'RIGHT', 'UP', 'DOWN']:
            count = head.get(key, 0)
            pct = _percent(count, total_head)
            if lang == "Arabic":
                label = {
                    'FORWARD': 'أمام',
                    'LEFT': 'يسار',
                    'RIGHT': 'يمين',
                    'UP': 'أعلى',
                    'DOWN': 'أسفل'
                }[key]
                if key == 'FORWARD':
                    lines.append(f" - {label}: {count} ({pct}%) <-- التركيز الأمامي يتم احتسابه فقط عند عدم النظر إلى مكان آخر\n")
                else:
                    lines.append(f" - {label}: {count} ({pct}%)\n")
            else:
                if key == 'FORWARD':
                    lines.append(f" - {key}: {count} ({pct}%) <-- center focus counts only when not looking elsewhere\n")
                else:
                    lines.append(f" - {key}: {count} ({pct}%)\n")
        lines.append("\n")

        if lang == "Arabic":
            lines.append("تحليل اليدين:\n")
        else:
            lines.append("Hand Analysis:\n")
        hc = data.get('hand_counts', {})
        total_hands = sum(hc.values())
        for key in ['No Hands', 'One Hand', 'Two Hands']:
            count = hc.get(key, 0)
            pct = _percent(count, total_hands)
            if lang == "Arabic":
                label = {
                    'No Hands': 'لا توجد يدين',
                    'One Hand': 'يد واحدة',
                    'Two Hands': 'يدان'
                }[key]
                lines.append(f" - {label}: {count} إطارًا ({pct}%)\n")
            else:
                lines.append(f" - {key}: {count} frames ({pct}%)\n")
        avg_move = data.get('hand_avg_movement', 0.0)
        status = data.get('hand_activity_status', 'N/A')
        if lang == "Arabic":
            lines.append(f" - متوسط حركة اليد: {round(avg_move, 1)}\n")
            lines.append(f" - حالة نشاط اليد: {status}\n\n")
        else:
            lines.append(f" - Average Hand Movement: {round(avg_move, 1)}\n")
            lines.append(f" - Hand Activity Status: {status}\n\n")

        if lang == "Arabic":
            lines.append("تحليل العواطف:\n")
        else:
            lines.append("Emotion Analysis:\n")
        em = data.get('emotion_stats', {})
        total_em = sum(em.values())
        if total_em == 0:
            lines.append(" - لا توجد مشاعر مكتشفة\n\n" if lang == "Arabic" else " - No emotions detected\n\n")
        else:
            for emotion, count in em.items():
                pct = _percent(count, total_em)
                if lang == "Arabic":
                    lines.append(f" - {emotion}: {count} ({pct}%)\n")
                else:
                    lines.append(f" - {emotion}: {count} ({pct}%)\n")
            dominant = max(em.items(), key=lambda x: x[1])[0]
            if lang == "Arabic":
                lines.append(f" - الشعور المسيطر: {dominant}\n\n")
            else:
                lines.append(f" - Dominant Emotion: {dominant}\n\n")

        if lang == "Arabic":
            lines.append("الأحداث الحرجة:\n")
            lines.append(" - لا توجد\n\n")
        else:
            lines.append("Critical Events:\n")
            lines.append(" - None\n\n")

        if lang == "Arabic":
            lines.append("تحليل لمس الوجه:\n")
            lines.append(f" - عدد لمس الوجه: {data.get('face_touch', {}).get('face_touch_count', 0)}\n")
            lines.append(f" - مدة لمس الوجه: {data.get('face_touch', {}).get('face_touch_duration', 0.0)} ثانية\n")
            lines.append(f" - أحداث لمس الوجه الطويلة: {data.get('face_touch', {}).get('long_face_touch_events', 0)}\n")
            lines.append(f" - عدد مرات تغطية الفم: {data.get('face_touch', {}).get('mouth_cover_count', 0)}\n")
            lines.append(f" - عدد مرات لمس الجبهة: {data.get('face_touch', {}).get('forehead_touch_count', 0)}\n")
            lines.append(f" - أحداث دعم الرأس: {data.get('face_touch', {}).get('head_support_events', 0)}\n\n")
        else:
            lines.append("Face Touch Analysis:\n")
            lines.append(f" - Face Touch Count: {data.get('face_touch', {}).get('face_touch_count', 0)}\n")
            lines.append(f" - Face Touch Duration: {data.get('face_touch', {}).get('face_touch_duration', 0.0)} sec\n")
            lines.append(f" - Long Face Touch Events: {data.get('face_touch', {}).get('long_face_touch_events', 0)}\n")
            lines.append(f" - Mouth Cover Events: {data.get('face_touch', {}).get('mouth_cover_count', 0)}\n")
            lines.append(f" - Forehead Touch Events: {data.get('face_touch', {}).get('forehead_touch_count', 0)}\n")
            lines.append(f" - Head Support Events: {data.get('face_touch', {}).get('head_support_events', 0)}\n\n")

        if lang == "Arabic":
            lines.append("تقييم جودة المقابلة:\n")
        else:
            lines.append("Interview Quality Score:\n")
        score, details = self._compute_quality_score(data)
        if lang == "Arabic":
            lines.append(f" - النقاط: {score}/100\n\n")
        else:
            lines.append(f" - Score: {score}/100\n\n")

        strengths, needs = self._build_feedback(data, language=lang)
        if lang == "Arabic":
            lines.append("نقاط القوة:\n")
        else:
            lines.append("Strengths:\n")
        if strengths:
            for item in strengths:
                lines.append(f" - {item}\n")
        else:
            lines.append(" - لا توجد نقاط قوة قوية محددة\n" if lang == "Arabic" else " - No strong strengths identified\n")
        lines.append("\n")

        if lang == "Arabic":
            lines.append("نقاط التحسين:\n")
        else:
            lines.append("Needs Improvement:\n")
        if needs:
            for item in needs:
                lines.append(f" - {item}\n")
        else:
            lines.append(" - لا توجد مجالات تحسين كبيرة محددة\n" if lang == "Arabic" else " - No major improvement areas identified\n")
        lines.append("\n")

        if lang == "Arabic":
            lines.append("التقييم النهائي:\n")
            if score >= 80:
                lines.append(" - أداء قوي في المقابلة مع تركيز وتحكم موثوق.\n")
            elif score >= 60:
                lines.append(" - أداء جيد مع بعض المجالات التي تحتاج للاستقرار.\n")
            else:
                lines.append(" - الجلسة تظهر فرصًا لتحسين التركيز والاستقرار.\n")
        else:
            lines.append("Final Assessment:\n")
            if score >= 80:
                lines.append(" - Strong interview performance with reliable focus and control.\n")
            elif score >= 60:
                lines.append(" - Good performance with a few areas to stabilize.\n")
            else:
                lines.append(" - The session shows improvement opportunities in focus and stability.\n")

        return "".join(lines)

    def _compute_quality_score(self, data: dict):
        eye = data.get('eye_stats', {})
        total_eye = sum(eye.values())
        center_pct = _percent(eye.get('CENTER', 0), total_eye)
        eye_score = min(25, max(0, (center_pct - 30)))

        head = data.get('head_stats', {})
        total_head = sum(head.values())
        forward_pct = _percent(head.get('FORWARD', 0), total_head)
        head_score = min(25, max(0, forward_pct - 20))

        avg_move = data.get('hand_avg_movement', 0.0)
        if avg_move < 8:
            hand_score = 20
        elif avg_move < 22:
            hand_score = 17
        elif avg_move < 32:
            hand_score = 12
        else:
            hand_score = 7

        ft = data.get('face_touch', {})
        touch_count = ft.get('face_touch_count', 0)
        if touch_count <= 2:
            face_touch_score = 20
        elif touch_count <= 5:
            face_touch_score = 14
        elif touch_count <= 8:
            face_touch_score = 9
        else:
            face_touch_score = 4

        score = int(min(100, eye_score + head_score + hand_score + face_touch_score))
        return score, {
            'eye_score': eye_score,
            'head_score': head_score,
            'hand_score': hand_score,
            'face_touch_score': face_touch_score,
        }

    def _build_feedback(self, data: dict, language: str = "English"):
        strengths = []
        needs = []
        arabic = language == "Arabic"

        eye = data.get('eye_stats', {})
        total_eye = sum(eye.values())
        center_pct = _percent(eye.get('CENTER', 0), total_eye)
        if center_pct >= 60:
            strengths.append("Good eye contact" if not arabic else "تواصل بصري جيد")
        elif center_pct < 40:
            needs.append("Looking away too often" if not arabic else "كنت تنظر بعيدًا كثيرًا")

        head = data.get('head_stats', {})
        forward_pct = _percent(head.get('FORWARD', 0), sum(head.values()))
        if forward_pct >= 70:
            strengths.append("Stable head position" if not arabic else "وضعية رأس مستقرة")
        else:
            needs.append("Head position is unstable" if not arabic else "وضعية الرأس غير مستقرة")

        ft = data.get('face_touch', {})
        touch_count = ft.get('face_touch_count', 0)
        if touch_count <= 2:
            strengths.append("Low face touch frequency" if not arabic else "تردد لمس الوجه منخفض")
        elif touch_count > 5:
            needs.append("Frequent face touching" if not arabic else "لمس وجه متكرر")

        avg_move = data.get('hand_avg_movement', 0.0)
        if avg_move < 22:
            strengths.append("Controlled hand movement" if not arabic else "حركة يد محكومة")
        else:
            needs.append("Excessive hand movement" if not arabic else "حركة يد مفرطة")

        return strengths, needs

    def save(self, text: str, path: str = 'interview_report.txt'):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)

    def save_png(self, data: dict, path: str = 'interview_report.png'):
        # Create a modern dark-themed dashboard PNG summarizing the report
        W = 1200
        H = 1600
        bg = (18, 18, 20)  # dark background
        img = Image.new('RGB', (W, H), color=bg)
        draw = ImageDraw.Draw(img)

        # Fonts (try common system fonts, fallback to default)
        try:
            title_font = ImageFont.truetype("arialbd.ttf", 40)
            header_font = ImageFont.truetype("arial.ttf", 20)
            text_font = ImageFont.truetype("arial.ttf", 16)
            small_font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            title_font = ImageFont.load_default()
            header_font = ImageFont.load_default()
            text_font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        # Colors (RGB)
        WHITE = (240, 240, 240)
        MUTED = (190, 190, 200)
        ACCENT = (58, 150, 255)
        GREEN = (75, 210, 140)
        ORANGE = (255, 165, 90)
        RED = (255, 95, 120)
        CARD = (28, 28, 34)
        SHADOW = (10, 10, 12)

        margin = 40
        gap = 20

        # Helpers
        def rounded_rect(xy, radius=12, fill=(0, 0, 0)):
            draw.rounded_rectangle(xy, radius=radius, fill=fill)

        def draw_card(x, y, w, h, title=None, fill=CARD):
            # shadow
            draw.rounded_rectangle((x+4, y+6, x+w+4, y+h+6), radius=14, fill=(8, 8, 10))
            draw.rounded_rectangle((x, y, x+w, y+h), radius=14, fill=fill)
            if title:
                draw.text((x+18, y+14), title, font=header_font, fill=WHITE)

        # Layout: two columns
        col_w = (W - margin*2 - gap) // 2
        left_x = margin
        right_x = margin + col_w + gap
        y = margin

        # Title
        draw.text((left_x, y), 'Interview Report', font=title_font, fill=WHITE)
        y += 60

        # Row 1: Eye, Head, Hand (stacked on left) and Score gauge on right
        card_h = 140
        # Left column stacked cards (Eye, Head)
        draw_card(left_x, y, col_w, card_h, title='Eye Analysis')
        # Eye content
        eye = data.get('eye_stats', {})
        total_eye = sum(eye.values()) or 1
        ex = left_x + 18
        ey = y + 48
        for k in ['CENTER', 'LEFT', 'RIGHT']:
            pct = _percent(eye.get(k,0), total_eye)
            draw.text((ex, ey), f"{k}", font=text_font, fill=MUTED)
            draw.text((ex + 140, ey), f"{eye.get(k,0)} ({pct}%)", font=text_font, fill=WHITE)
            ey += 22

        # Head Pose card
        y2 = y + card_h + gap
        draw_card(left_x, y2, col_w, card_h, title='Head Pose Analysis')
        hx = left_x + 18
        hy = y2 + 48
        head = data.get('head_stats', {})
        total_head = sum(head.values()) or 1
        for k in ['FORWARD', 'LEFT', 'RIGHT', 'UP', 'DOWN']:
            pct = _percent(head.get(k,0), total_head)
            draw.text((hx, hy), f"{k}", font=text_font, fill=MUTED)
            draw.text((hx + 140, hy), f"{head.get(k,0)} ({pct}%)", font=text_font, fill=WHITE)
            hy += 20

        # Right column: Score Gauge
        score_card_h = card_h*2 + gap
        draw_card(right_x, y, col_w, score_card_h, title='Interview Quality Score')
        score, _ = self._compute_quality_score(data)
        # gauge circle
        cx = right_x + col_w//2 + 40
        cy = y + score_card_h//2 + 10
        radius = 80
        # background circle
        draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=(22,22,28))
        # arc for score
        from math import pi
        end_angle = int(360 * (score / 100))
        # use pieslice for arc overlay
        if score >= 75:
            col = GREEN
        elif score >= 50:
            col = ORANGE
        else:
            col = RED
        # draw arc by drawing a mask segment
        draw.pieslice((cx-radius, cy-radius, cx+radius, cy+radius), -90, -90+end_angle, fill=col)
        # inner circle to create ring
        inner = 50
        draw.ellipse((cx-inner, cy-inner, cx+inner, cy+inner), fill=(22,22,28))
        # score text
        score_text = f"{score} / 100"
        bbox = draw.textbbox((0, 0), score_text, font=header_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((cx - tw/2, cy - th/2), score_text, font=header_font, fill=WHITE)

        # Emotion Analysis (full width under left column)
        em_y = y2 + card_h + gap
        em_h = 180
        draw_card(left_x, em_y, col_w, em_h, title='Emotion Analysis')
        em = data.get('emotion_stats', {})
        total_em = sum(em.values()) or 1
        ex = left_x + 18
        ey = em_y + 48
        max_bar_w = col_w - 60
        # draw top 6 emotions sorted
        items = sorted(em.items(), key=lambda x: x[1], reverse=True)[:6]
        if not items:
            draw.text((ex, ey), "No emotions detected", font=text_font, fill=MUTED)
        else:
            for k, v in items:
                pct = _percent(v, total_em)
                draw.text((ex, ey), f"{k}", font=text_font, fill=MUTED)
                # bar background
                bx = ex + 140
                by = ey - 4
                draw.rounded_rectangle((bx, by, bx + max_bar_w, by + 16), radius=8, fill=(36,36,40))
                # bar fill
                fill_w = int(max_bar_w * (pct/100))
                draw.rounded_rectangle((bx, by, bx + fill_w, by + 16), radius=8, fill=ACCENT)
                draw.text((bx + max_bar_w + 8, ey), f"{pct}%", font=text_font, fill=WHITE)
                ey += 28

        # Face Touch Analysis card (right column under score)
        ft_y = y + score_card_h + gap
        ft_h = 220
        draw_card(right_x, ft_y, col_w, ft_h, title='Face Touch Analysis')
        fx = right_x + 18
        fy = ft_y + 48
        ft = data.get('face_touch', {})
        draw.text((fx, fy), f"Face Touch Count: {ft.get('face_touch_count',0)}", font=text_font, fill=WHITE)
        fy += 22
        draw.text((fx, fy), f"Face Touch Duration (s): {ft.get('face_touch_duration',0.0)}", font=text_font, fill=WHITE)

        # Voice Analysis (left column under emotion)
        ft_y = ph_y + ph_h + gap
        ft_h = 180
        draw_card(right_x, ft_y, col_w, ft_h, title='Face Touch Analysis')
        fx = right_x + 18
        fy = ft_y + 48
        ft = data.get('face_touch', {})
        draw.text((fx, fy), f"Face Touch Count: {ft.get('face_touch_count',0)}", font=text_font, fill=WHITE)
        fy += 22
        draw.text((fx, fy), f"Face Touch Duration (s): {ft.get('face_touch_duration',0.0)}", font=text_font, fill=WHITE)

        # Voice Analysis (left column under emotion)
        va_y = em_y + em_h + gap
        va_h = 180
        draw_card(left_x, va_y, col_w, va_h, title='Voice Analysis')
        vx = left_x + 18
        vy = va_y + 48
        voice_stats = data.get('voice_stats', [])
        if voice_stats:
            for vs in voice_stats[:6]:
                draw.text((vx, vy), f"- {vs}", font=small_font, fill=WHITE)
                vy += 22
        else:
            draw.text((vx, vy), "No voice data available", font=text_font, fill=MUTED)

        # Strengths and Needs (two small cards)
        sn_y = max(va_y + va_h, ft_y + ft_h) + gap
        small_h = 180
        draw_card(left_x, sn_y, col_w, small_h, title='Strengths')
        sx = left_x + 18
        sy = sn_y + 48
        strengths, needs = self._build_feedback(data)
        if strengths:
            for s in strengths[:8]:
                draw.text((sx, sy), f"- {s}", font=small_font, fill=GREEN)
                sy += 20
        else:
            draw.text((sx, sy), '- None', font=small_font, fill=MUTED)

        draw_card(right_x, sn_y, col_w, small_h, title='Needs Improvement')
        nx = right_x + 18
        ny = sn_y + 48
        if needs:
            for n in needs[:8]:
                draw.text((nx, ny), f"- {n}", font=small_font, fill=ORANGE)
                ny += 20
        else:
            draw.text((nx, ny), '- None', font=small_font, fill=MUTED)

        # Final Assessment full-width card at bottom
        fa_y = sn_y + small_h + gap
        fa_h = 140
        draw_card(left_x, fa_y, W - margin*2, fa_h, title='Final Assessment')
        final_x = left_x + 18
        final_y = fa_y + 48
        if score >= 80:
            final_text = 'Strong interview performance with reliable focus and control.'
        elif score >= 60:
            final_text = 'Good performance with a few areas to stabilize.'
        else:
            final_text = 'The session shows improvement opportunities in focus and stability.'
        draw.text((final_x, final_y), final_text, font=text_font, fill=WHITE)

        img.save(path)
