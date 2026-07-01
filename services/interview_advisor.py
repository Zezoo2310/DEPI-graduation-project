"""
interview_advisor.py - Smart Interview Advisor (Rule-Based)

Reads the parsed interview report and generates personalized advice
based on the specific weaknesses found in each section.
"""


def generate_advice(report: dict) -> dict:
    """
    Analyze the parsed interview report and generate targeted advice.

    Args:
        report: Parsed report dictionary from interview_report_parser.

    Returns:
        Dictionary with:
            - overall_summary: str
            - score: int
            - advice_sections: list of {category, icon, status, messages}
    """
    score = report.get("interview_score", 0)
    advice_sections = []

    interview_language = report.get("interview_language", "English")
    arabic = interview_language == "Arabic"

    def tr(en: str, ar: str) -> str:
        return ar if arabic else en

    def tr_status(status: str) -> str:
        if not arabic:
            return status
        return {
            "Excellent": "ممتاز",
            "Good": "جيد",
            "Needs Work": "يحتاج تحسين",
            "Warning": "تحذير",
            "Critical": "حرج",
        }.get(status, status)

    def translate_item(item: str) -> str:
        translations = {
            "Good eye contact": "تواصل بصري جيد",
            "Looking away too often": "كنت تنظر بعيدًا كثيرًا",
            "Stable head position": "وضعية رأس مستقرة",
            "Head position is unstable": "وضعية الرأس غير مستقرة",
            "Low face touch frequency": "تردد لمس الوجه منخفض",
            "Frequent face touching": "لمس وجه متكرر",
            "Controlled hand movement": "حركة يد محكومة",
            "Excessive hand movement": "حركة يد مفرطة",
        }
        return translations.get(item, item)

    # ─── 1. Interview Language ───
    if arabic:
        language_msgs = [
            "وضع المقابلة المحدد: العربية.",
            "النتائج معروضة وفقًا لاختيارك للغة المقابلة."
        ]
        category_label = "لغة المقابلة"
        score_detail = f"المحدد: {interview_language}"
    else:
        language_msgs = [
            f"Interview mode selected: {interview_language}.",
            "Your results are presented with this language preference in mind."
        ]
        category_label = "Interview Language"
        score_detail = f"Selected: {interview_language}"

    advice_sections.append({
        "category": category_label,
        "icon": "fa-language",
        "status": "good",
        "score_detail": score_detail,
        "messages": language_msgs,
    })

    # ─── 2. EYE CONTACT ───
    eye_center = report.get("eye_percentages", {}).get("CENTER", 0)
    eye_msgs = []
    if eye_center >= 80:
        eye_status = "excellent"
        eye_msgs.append(tr(
            "You maintained excellent eye contact throughout the interview. This shows confidence and engagement.",
            "حافظت على تواصل بصري ممتاز طوال المقابلة. هذا يظهر الثقة والانخراط."
        ))
    elif eye_center >= 60:
        eye_status = "good"
        eye_msgs.append(tr(
            "Your eye contact was good but could be more consistent.",
            "كان تواصلك البصري جيدًا لكن يمكن أن يكون أكثر اتساقًا."
        ))
        eye_msgs.append(tr(
            "Tip: Place a small sticker next to your camera lens as a reminder to look directly at it while speaking.",
            "نصيحة: ضع ملصقًا صغيرًا بجانب عدسة الكاميرا لتذكرك بالنظر إليها مباشرة أثناء الحديث."
        ))
        eye_msgs.append(tr(
            "Tip: Practice the 50/70 rule - maintain eye contact 50% of the time while speaking and 70% while listening.",
            "نصيحة: مارس قاعدة 50/70 - حافظ على التواصل البصري 50% من الوقت أثناء التحدث و70% أثناء الاستماع."
        ))
    else:
        eye_status = "needs_work"
        eye_msgs.append(tr(
            "Your eye contact was below average. You were looking away from the screen frequently.",
            "كان تواصلك البصري أقل من المتوسط. كنت تنظر بعيدًا عن الشاشة كثيرًا."
        ))
        eye_msgs.append(tr(
            "Tip: Position your camera at eye level so looking at the screen naturally feels like eye contact.",
            "نصيحة: ضع الكاميرا على مستوى العين حتى يبدو النظر إلى الشاشة كتواصل بصري طبيعي."
        ))
        eye_msgs.append(tr(
            "Tip: If you feel nervous, look at the interviewer's forehead area instead of directly into their eyes - it appears the same on camera.",
            "نصيحة: إذا شعرت بالتوتر، انظر إلى جبين المحاور بدلاً من عينيه مباشرة؛ يبدو نفس الشيء على الكاميرا."
        ))
        eye_msgs.append(tr(
            "Tip: Avoid reading from notes placed away from your screen during the interview.",
            "نصيحة: تجنب القراءة من ملاحظات موجودة بعيدًا عن الشاشة أثناء المقابلة."
        ))

    advice_sections.append({
        "category": tr("Eye Contact", "التواصل البصري"),
        "icon": "fa-eye",
        "status": eye_status,
        "score_detail": tr(f"{eye_center}% center focus", f"{eye_center}% تركيز مركزي"),
        "messages": eye_msgs,
    })

    # ─── 2. HEAD STABILITY ───
    head_forward = report.get("head_percentages", {}).get("FORWARD", 0)
    head_right = report.get("head_percentages", {}).get("RIGHT", 0)
    head_left = report.get("head_percentages", {}).get("LEFT", 0)
    head_msgs = []

    head_unstable = (head_forward < 40) or (head_right > 50) or (head_left > 50)
    if not head_unstable:
        head_status = "excellent"
        head_msgs.append(tr(
            "Your head position was stable and centered. Great job maintaining a professional posture!",
            "وضعية رأسك كانت مستقرة ومركزية. أحسنت في الحفاظ على وضعية مهنية!"
        ))
    else:
        head_status = "needs_work"
        if head_right > 50:
            head_msgs.append(tr(
                f"You tilted your head to the RIGHT {head_right}% of the time. This can signal discomfort or distraction.",
                f"أدرَت رأسك إلى اليمين {head_right}% من الوقت. هذا قد يشير إلى عدم راحة أو تشتيت."
            ))
        if head_left > 50:
            head_msgs.append(tr(
                f"You tilted your head to the LEFT {head_left}% of the time.",
                f"أدرَت رأسك إلى اليسار {head_left}% من الوقت."
            ))
        head_msgs.append(tr(
            "Tip: Sit upright with your back against the chair. Imagine a string pulling the top of your head toward the ceiling.",
            "نصيحة: اجلس منتصبًا وظهرك إلى الكرسي. تخيل خيطًا يسحب أعلى رأسك نحو السقف."
        ))
        head_msgs.append(tr(
            "Tip: Keep your screen directly in front of you, not to the side, so you naturally face forward.",
            "نصيحة: اجعل الشاشة أمامك مباشرةً، وليس إلى الجانب، حتى تواجه للأمام بشكل طبيعي."
        ))
        head_msgs.append(tr(
            "Tip: Practice recording yourself for 2 minutes daily to become aware of unconscious head movements.",
            "نصيحة: تدرب على تسجيل نفسك لمدة دقيقتين يوميًا لتصبح مدركًا لحركات الرأس اللاواعية."
        ))

    advice_sections.append({
        "category": tr("Head Position", "وضعية الرأس"),
        "icon": "fa-head-side",
        "status": head_status,
        "score_detail": tr(f"{head_forward}% forward", f"{head_forward}% للأمام"),
        "messages": head_msgs,
    })

    # ─── 3. HAND GESTURES ───
    no_hands_pct = report.get("hands_percentages", {}).get("No Hands", 0)
    hand_status_text = report.get("hand_activity_status", "N/A")
    hand_avg = report.get("hand_avg_movement", 0)
    hand_msgs = []

    if hand_status_text == "Too Static" or no_hands_pct > 85:
        hand_status = "needs_work"
        hand_msgs.append(tr(
            f"Your hands were not visible {no_hands_pct}% of the time. This can make you appear stiff and nervous.",
            f"لم تكن يداك مرئيتين خلال {no_hands_pct}% من الوقت. هذا قد يجعلك تبدو متشدداً ومتوتراً."
        ))
        hand_msgs.append(tr(
            "Tip: Keep your hands visible on the desk or table in front of you.",
            "نصيحة: اجعل يديك مرئيتيْن على المكتب أو الطاولة أمامك."
        ))
        hand_msgs.append(tr(
            "Tip: Use natural hand gestures when explaining a point - it makes you look more confident and engaged.",
            "نصيحة: استخدم حركات يدوية طبيعية عند شرح نقطة - هذا يجعلك تبدو أكثر ثقة وتفاعلًا."
        ))
        hand_msgs.append(tr(
            "Tip: Practice the 'open palm' technique - showing your palms while speaking signals honesty and openness.",
            "نصيحة: مارس تقنية 'الراحة المفتوحة' - إظهار راحة اليدين أثناء الكلام يرسل رسالة أصالة وانفتاح."
        ))
    elif hand_avg > 60:
        hand_status = "warning"
        hand_msgs.append(tr(
            "Your hand movement was excessive. Too much gesturing can be distracting to the interviewer.",
            "كانت حركة يدك مفرطة. الكثير من الإيماءات قد يشتت المحاور."
        ))
        hand_msgs.append(tr(
            "Tip: Keep gestures within a 'box' from your shoulders to your waist.",
            "نصيحة: اجعل الإيماءات داخل 'مربع' من الكتفين إلى الخصر."
        ))
        hand_msgs.append(tr(
            "Tip: Rest your hands on the table between points to create natural pauses.",
            "نصيحة: ضع يديك على الطاولة بين النقاط لخلق توقفات طبيعية."
        ))
    else:
        hand_status = "excellent"
        hand_msgs.append(tr(
            "Your hand gestures were natural and well-controlled. This shows good communication skills!",
            "كانت إيماءات يديك طبيعية ومحكمة. هذا يظهر مهارات تواصل جيدة!"
        ))

    hand_status_label = hand_status_text
    if arabic:
        hand_status_label = {
            "Too Static": "ثابت جدًا",
            "Active": "نشط",
            "Moderate": "معتدل",
            "N/A": "غير متاح"
        }.get(hand_status_text, hand_status_text)

    advice_sections.append({
        "category": tr("Hand Gestures", "حركات اليد"),
        "icon": "fa-hand",
        "status": hand_status,
        "score_detail": tr(f"Activity: {hand_status_text}", f"النشاط: {hand_status_label}"),
        "messages": hand_msgs,
    })

    # ─── 4. EMOTIONS ───
    emotions = report.get("emotions", {})
    dominant = report.get("dominant_emotion", "N/A")
    emotion_msgs = []

    sad_pct = emotions.get("sad", {}).get("percent", 0)
    angry_pct = emotions.get("angry", {}).get("percent", 0)
    fear_pct = emotions.get("fear", {}).get("percent", 0)
    happy_pct = emotions.get("happy", {}).get("percent", 0)
    neutral_total = emotions.get("neutral", {}).get("percent", 0) + emotions.get("Neutral", {}).get("percent", 0)

    negative_total = sad_pct + angry_pct + fear_pct

    if negative_total > 30:
        emotion_status = "needs_work"
        if sad_pct > 15:
            emotion_msgs.append(tr(
                f"Sadness was detected {sad_pct}% of the time. This may signal low confidence or anxiety.",
                f"تم اكتشاف الحزن بنسبة {sad_pct}% من الوقت. هذا قد يشير إلى ضعف الثقة أو القلق."
            ))
            emotion_msgs.append(tr(
                "Tip: Before the interview, practice 'power posing' for 2 minutes (standing tall with hands on hips). Research shows this reduces stress hormones.",
                "نصيحة: قبل المقابلة، مارس وضعيات القوة لمدة دقيقتين (وقوفًا مستقيمًا واليدان على الوركين). الأبحاث تظهر أنها تقلل هرمونات التوتر."
            ))
        if angry_pct > 15:
            emotion_msgs.append(tr(
                f"Anger/frustration was detected {angry_pct}% of the time.",
                f"تم اكتشاف الغضب/الإحباط بنسبة {angry_pct}% من الوقت."
            ))
            emotion_msgs.append(tr(
                "Tip: Take a deep breath before answering difficult questions. Count to 3 in your head.",
                "نصيحة: خذ نفسًا عميقًا قبل الإجابة على الأسئلة الصعبة. عد إلى 3 في ذهنك."
            ))
        if fear_pct > 15:
            emotion_msgs.append(tr(
                f"Fear was detected {fear_pct}% of the time.",
                f"تم اكتشاف الخوف بنسبة {fear_pct}% من الوقت."
            ))
            emotion_msgs.append(tr(
                "Tip: Prepare answers for common questions beforehand. Familiarity reduces anxiety.",
                "نصيحة: حضر إجابات للأسئلة الشائعة مسبقًا. الألفة تقلل القلق."
            ))
        emotion_msgs.append(tr(
            "Tip: Practice the 4-7-8 breathing technique before the interview: Inhale for 4 seconds, hold for 7, exhale for 8.",
            "نصيحة: مارس تقنية التنفس 4-7-8 قبل المقابلة: استنشق 4 ثوان، احبس 7 ثوان، أزفر 8 ثوان."
        ))
    elif happy_pct > 30:
        emotion_status = "excellent"
        emotion_msgs.append(tr(
            f"Your dominant emotion was '{dominant}'. You appeared positive and confident!",
            f"العاطفة المهيمنة كانت '{dominant}'. بدوت إيجابيًا وواثقًا!"
        ))
        emotion_msgs.append(tr(
            "Tip: Keep this positive energy! A genuine smile makes you more likeable and memorable.",
            "نصيحة: استمر في هذه الطاقة الإيجابية! الابتسامة الحقيقية تجعلك أكثر ودًا ولا تُنسى."
        ))
    else:
        emotion_status = "good"
        emotion_msgs.append(tr(
            f"Your emotional state was mostly neutral/stable. Dominant emotion: '{dominant}'.",
            f"حالَتك العاطفية كانت محايدة/مستقرة إلى حد كبير. العاطفة المهيمنة: '{dominant}'."
        ))
        emotion_msgs.append(tr(
            "Tip: Try to smile naturally at the start and end of the interview. It creates a positive first and last impression.",
            "نصيحة: حاول أن تبتسم بشكل طبيعي في بداية ونهاية المقابلة. هذا يخلق انطباعًا أوليًا وأخيرًا إيجابيًا."
        ))

    advice_sections.append({
        "category": tr("Emotional State", "الحالة العاطفية"),
        "icon": "fa-face-smile",
        "status": emotion_status,
        "score_detail": tr(f"Dominant: {dominant}", f"المهيمن: {dominant}"),
        "messages": emotion_msgs,
    })

    # ─── 5. FACE TOUCHING ───
    face_count = report.get("face_touch_count", 0)
    face_dur = report.get("face_touch_duration", 0)
    mouth_cover = report.get("mouth_cover_events", 0)
    forehead_touch = report.get("forehead_touch_events", 0)
    face_msgs = []

    if face_count > 5:
        face_status = "needs_work"
        face_msgs.append(tr(
            f"You touched your face {face_count} times for a total of {face_dur} seconds.",
            f"لمست وجهك {face_count} مرات بإجمالي {face_dur} ثانية."
        ))
        if mouth_cover > 0:
            face_msgs.append(tr(
                f"You covered your mouth {mouth_cover} time(s). This is a subconscious signal of uncertainty or hiding information.",
                f"غطّيت فمك {mouth_cover} مرة. هذا إشارة لاواعية إلى عدم اليقين أو إخفاء المعلومات."
            ))
            face_msgs.append(tr(
                "Tip: When you feel the urge to cover your mouth, redirect your hand to rest on the table instead.",
                "نصيحة: عندما تشعر بالرغبة في تغطية فمك، أعد يدك لتستريح على الطاولة بدلاً من ذلك."
            ))
        if forehead_touch > 0:
            face_msgs.append(tr(
                f"You touched your forehead {forehead_touch} time(s). This can signal stress or overthinking.",
                f"لمست جبينك {forehead_touch} مرات. هذا قد يشير إلى التوتر أو الإفراط في التفكير."
            ))
            face_msgs.append(tr(
                "Tip: If you need a moment to think, it's perfectly fine to say 'Let me think about that for a moment' instead of touching your face.",
                "نصيحة: إذا احتجت لحظة للتفكير، من الأفضل أن تقول 'دعني أفكر في ذلك للحظة' بدلًا من لمس وجهك."
            ))
        face_msgs.append(tr(
            "Tip: Keep your hands clasped loosely on the desk. This naturally prevents face touching.",
            "نصيحة: أبق يديك متشابكتين بشكل مريح على المكتب. هذا يمنع لمس الوجه بشكل طبيعي."
        ))
        face_msgs.append(tr(
            "Tip: Practice mock interviews with a friend who signals you every time you touch your face.",
            "نصيحة: تدرب على مقابلات وهمية مع صديق يشتّتك في كل مرة تلمس فيها وجهك."
        ))
    elif face_count > 0:
        face_status = "good"
        face_msgs.append(tr(
            f"You touched your face {face_count} time(s). This is within a normal range but can be improved.",
            f"لمست وجهك {face_count} مرات. هذا ضمن النطاق الطبيعي لكن يمكن تحسينه."
        ))
        face_msgs.append(tr(
            "Tip: Be aware of nervous habits. Self-awareness is the first step to eliminating them.",
            "نصيحة: كن واعيًا للعادات العصبية. الوعي الذاتي هو الخطوة الأولى لإزالتها."
        ))
    else:
        face_status = "excellent"
        face_msgs.append(tr(
            "No face touching detected. This shows excellent composure and self-control!",
            "لم يتم اكتشاف لمس الوجه. هذا يدل على رباطة جأش وضبط نفس ممتاز!"
        ))

    advice_sections.append({
        "category": tr("Face Touching", "لمس الوجه"),
        "icon": "fa-hand-dots",
        "status": face_status,
        "score_detail": tr(f"{face_count} touches", f"{face_count} مرات لمس"),
        "messages": face_msgs,
    })

    # ─── 7. VOICE ANALYSIS ───
    voice_data = report.get("voice_analysis", [])
    voice_msgs = []

    if voice_data:
        states = [v["state"] for v in voice_data]
        hesitant_count = states.count("Hesitant")
        stressed_count = states.count("Stressed")
        confident_count = states.count("Confident")
        enthusiastic_count = states.count("Enthusiastic")
        total = len(states)

        if hesitant_count > total * 0.3:
            voice_status = "needs_work"
            voice_msgs.append(tr(
                f"Your voice was classified as 'Hesitant' in {hesitant_count}/{total} intervals.",
                f"تم تصنيف صوتك على أنه 'متردد' في {hesitant_count}/{total} فترات."
            ))
            voice_msgs.append(tr(
                "Tip: Speak slightly louder than you think is necessary. On video calls, softer voices sound even quieter.",
                "نصيحة: تكلم بصوت أعلى قليلاً مما تعتقد أنه ضروري. في مكالمات الفيديو، الأصوات الخافتة تبدو أكثر هدوءًا."
            ))
            voice_msgs.append(tr(
                "Tip: Practice your answers out loud before the interview. Rehearsal builds vocal confidence.",
                "نصيحة: تدرب على إجاباتك بصوت عالٍ قبل المقابلة. التمرين يبني ثقة صوتية."
            ))
            voice_msgs.append(tr(
                "Tip: Avoid filler words (um, uh, like). Instead, pause silently - it actually sounds more professional.",
                "نصيحة: تجنب كلمات الحشو (أم، آه، مثلًا). بدلاً من ذلك، توقف بصمت - هذا يبدو أكثر احترافية."
            ))
        elif stressed_count > total * 0.3:
            voice_status = "warning"
            voice_msgs.append(tr(
                f"Your voice was classified as 'Stressed' in {stressed_count}/{total} intervals.",
                f"تم تصنيف صوتك على أنه 'متوتر' في {stressed_count}/{total} فترات."
            ))
            voice_msgs.append(tr(
                "Tip: Slow down your speech rate. Stressed people tend to rush through their answers.",
                "نصيحة: أبطئ من وتيرة كلامك. الأشخاص المتوترون يميلون لمسرعة إجاباتهم."
            ))
            voice_msgs.append(tr(
                "Tip: Drink water before and during the interview. A dry throat increases vocal tension.",
                "نصيحة: اشرب ماءً قبل وأثناء المقابلة. الحلق الجاف يزيد التوتر الصوتي."
            ))
            voice_msgs.append(tr(
                "Tip: Warm up your voice before the interview by humming or reading aloud for 5 minutes.",
                "نصيحة: سخّن صوتك قبل المقابلة بالهمهمة أو القراءة بصوت عالٍ لمدة 5 دقائق."
            ))
        elif confident_count > total * 0.5:
            voice_status = "excellent"
            voice_msgs.append(tr(
                f"Your voice was classified as 'Confident' in {confident_count}/{total} intervals. Excellent vocal presence!",
                f"تم تصنيف صوتك على أنه 'واثق' في {confident_count}/{total} فترات. حضور صوتي ممتاز!"
            ))
            voice_msgs.append(tr(
                "Tip: Keep using varied intonation to maintain the interviewer's engagement.",
                "نصيحة: استمر في استخدام نبرة صوت متنوعة للحفاظ على تفاعل المحاور."
            ))
        else:
            voice_status = "good"
            voice_msgs.append(tr(
                "Your vocal performance was mixed. Keep practicing to build more consistency.",
                "كان أداءك الصوتي مختلطًا. استمر في التدريب لبناء مزيد من الثبات."
            ))
            voice_msgs.append(tr(
                "Tip: Try recording yourself answering common questions to spot where you drop your energy.",
                "نصيحة: حاول تسجيل نفسك وأنت تجيب عن الأسئلة الشائعة لتلاحظ أين تفقد طاقتك."
            ))
            voice_msgs.append(tr(
                "Tip: Use strategic pauses instead of filler words when transitioning between ideas.",
                "نصيحة: استخدم فترات توقف مدروسة بدلًا من كلمات الحشو عند الانتقال بين الأفكار."
            ))

        active_total = total - states.count("Inactive")
        if active_total > 0:
            hesitant_pct = int((hesitant_count / active_total) * 100)
            stressed_pct = int((stressed_count / active_total) * 100)
            confident_pct = int(((confident_count + enthusiastic_count) / active_total) * 100)
            summary_msg = f"VOICE_SUMMARY:{confident_pct},{hesitant_pct},{stressed_pct}"
            voice_msgs.append(summary_msg)
    else:
        voice_status = "neutral"
        voice_msgs.append(tr(
            "No voice analysis data available in this report.",
            "لا تتوفر بيانات تحليل الصوت في هذا التقرير."
        ))

    advice_sections.append({
        "category": tr("Voice & Tone", "الصوت والنبرة"),
        "icon": "fa-microphone",
        "status": voice_status,
        "score_detail": tr(f"{len(voice_data)} intervals analyzed", f"تم تحليل {len(voice_data)} فترات"),
        "messages": voice_msgs,
    })

    # ─── OVERALL SUMMARY ───
    if score >= 80:
        overall = tr(
            "Excellent performance! You demonstrated strong interview skills. Keep up the great work!",
            "أداء ممتاز! أظهرت مهارات مقابلة قوية. واصل العمل الجيد!"
        )
    elif score >= 60:
        overall = tr(
            "Good performance with some areas to improve. Focus on the tips below to strengthen your weak points.",
            "أداء جيد مع بعض المجالات التي تحتاج إلى تحسين. ركز على النصائح أدناه لتقوية نقاط الضعف."
        )
    elif score >= 40:
        overall = tr(
            "Average performance. Several areas need attention. Practice the recommendations below before your next interview.",
            "أداء متوسط. هناك عدة مجالات بحاجة إلى اهتمام. مارس التوصيات أدناه قبل مقابلتك التالية."
        )
    else:
        overall = tr(
            "Your interview performance needs significant improvement. Don't worry - follow the advice below and practice regularly!",
            "أداء المقابلة يحتاج إلى تحسين كبير. لا تقلق - اتبع النصائح أدناه وتدرّب بانتظام!"
        )

    strengths = report.get("strengths", [])
    needs_improvement = report.get("needs_improvement", [])
    if arabic:
        strengths = [translate_item(item) for item in strengths]
        needs_improvement = [translate_item(item) for item in needs_improvement]

    return {
        "overall_summary": overall,
        "score": score,
        "strengths": strengths,
        "needs_improvement": needs_improvement,
        "final_assessment": report.get("final_assessment", ""),
        "advice_sections": advice_sections,
        "interview_language": interview_language,
    }
