"""
app.py - ATS Resume Checker Flask Application

Main entry point for the web application.
Orchestrates the processing pipeline:
    1. Receive PDF upload OR select a sample resume from Datasets
    2. Extract text & metadata  (pdf_parser)
    3. NLP preprocessing        (nlp_processor)
    4. Feature extraction       (feature_extractor)
    5. ATS scoring              (ats_scorer)
    6. Render dashboard

Designed for future integration with a Resume-Job Matching module.
"""

import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt

from services.pdf_parser import parse_pdf
from services.nlp_processor import process_text
from services.feature_extractor import extract_all_features
from services.ats_scorer import calculate_ats_score
from services.job_matcher_model import predict_match
from services.job_description_parser import parse_job_description
from services.interview_report_parser import parse_interview_report
from services.interview_advisor import generate_advice
from services.llm_coach import get_ai_coach_response

# ═══════════════════════════════════════════════════════════════════════════
#  APP CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add the Smart Interview Video_Analysis folder to sys.path so we can import interview_streamer
video_analysis_path = os.path.join(BASE_DIR, "Project_Smart_Interview", "Video_Analysis")
if os.path.isdir(video_analysis_path) and video_analysis_path not in sys.path:
    sys.path.append(video_analysis_path)

try:
    from interview_streamer import stream_interview
    print("[app] interview_streamer loaded OK", flush=True)
except ImportError as _ie:
    print(f"[app] WARNING: interview_streamer not found: {_ie}", flush=True)
    stream_interview = None

app = Flask(__name__)
app.secret_key = "ats-resume-checker-secret-key-2024"

# Database & Auth configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message_category = "info"

# Upload configuration
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DATASETS_DIR = os.path.join(BASE_DIR, "Datasets")
ALLOWED_EXTENSIONS = {"pdf"}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB max

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Ensure the uploads directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
#  DATABASE MODELS
# ═══════════════════════════════════════════════════════════════════════════

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))



def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_test_resumes() -> list:
    """Load sample resumes from Datasets/test_resumes.json."""
    path = os.path.join(DATASETS_DIR, "test_resumes.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


# ═══════════════════════════════════════════════════════════════════════════
#  SHARED ANALYSIS PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def analyze_text(raw_text: str, pdf_metadata: dict = None) -> tuple:
    """
    Run the full NLP → Feature Extraction → ATS Scoring pipeline on raw text.

    Args:
        raw_text: The resume text to analyze.
        pdf_metadata: Optional PDF metadata dict. If None, uses defaults.

    Returns:
        (features, ats_result) tuple.
    """
    if pdf_metadata is None:
        pdf_metadata = {
            "page_count": 1,
            "has_image": False,
            "has_table": False,
            "multi_column_layout": False,
            "resume_length_score": "good",
        }

    # NLP preprocessing
    nlp_result = process_text(raw_text)
    cleaned_text = nlp_result["cleaned_text"]
    entities = nlp_result["entities"]
    word_count = nlp_result["word_count"]
    lines = nlp_result["lines"]

    # Feature extraction
    features = extract_all_features(
        cleaned_text=cleaned_text,
        entities=entities,
        lines=lines,
        word_count=word_count,
        pdf_metadata=pdf_metadata,
    )

    # ATS scoring
    ats_result = calculate_ats_score(features)

    return features, ats_result


# ═══════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def home():
    """Landing Page"""
    return render_template("home.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("hub"))
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")
        
        user_exists = User.query.filter_by(email=email).first()
        if user_exists:
            flash("Email already registered. Please login.", "error")
            return redirect(url_for("register"))
            
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(name=name, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        flash("Account created successfully! Please login.", "success")
        return redirect(url_for("login"))
        
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("hub"))
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = User.query.filter_by(email=email).first()
        
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash(f"Welcome back, {user.name}!", "success")
            return redirect(url_for("hub"))
        else:
            flash("Login Unsuccessful. Please check email and password.", "error")
            
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))

@app.route("/hub")
@login_required
def hub():
    """Main App Hub"""
    return render_template("hub.html")


# ── Modules ──

@app.route("/ats-checker", methods=["GET"])
@login_required
def ats_checker():
    """ATS Resume Checker page"""
    samples = load_test_resumes()
    return render_template("ats_checker.html", samples=samples)

@app.route("/job-matcher")
@login_required
def job_matcher():
    """Resume-Job Matcher Page"""
    return render_template("job_matcher.html")

@app.route("/match-upload", methods=["POST"])
@login_required
def match_upload():
    """
    Handle Job Matcher: upload resume PDF + raw job description text.
    Auto-extracts job title, skills, education, and experience from JD.
    """
    # Validate file
    if "resume" not in request.files or request.files["resume"].filename == "":
        flash("Please upload a PDF resume.", "error")
        return redirect(url_for("job_matcher"))

    file = request.files["resume"]
    if not allowed_file(file.filename):
        flash("Only PDF files are accepted.", "error")
        return redirect(url_for("job_matcher"))

    # Get raw job description text
    job_description_raw = request.form.get("job_description", "")
    if not job_description_raw.strip():
        flash("Please paste the job description.", "error")
        return redirect(url_for("job_matcher"))

    # Auto-parse the job description using NLP
    jd_parsed = parse_job_description(job_description_raw)
    job_title = jd_parsed["job_title"]
    job_skills = jd_parsed["skills"]
    job_education = jd_parsed["education"]
    job_experience = jd_parsed["experience"]

    # Save and process PDF
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    try:
        # Extract text from PDF
        pdf_result = parse_pdf(file_path)
        raw_text = pdf_result["text"]

        if not raw_text.strip():
            flash("Could not extract text from this PDF.", "error")
            return redirect(url_for("job_matcher"))

        # NLP processing to extract resume features
        nlp_result = process_text(raw_text)
        features = extract_all_features(
            cleaned_text=nlp_result["cleaned_text"],
            entities=nlp_result["entities"],
            lines=nlp_result["lines"],
            word_count=nlp_result["word_count"],
            pdf_metadata=pdf_result["metadata"],
        )

        # Build resume features for the ML model
        resume_skills = features.get("technical_skills", []) + features.get("soft_skills", [])
        resume_features = {
            "skills": resume_skills,
            "degree": str(features.get("education", {}).get("degree_type", "")),
            "responsibilities": raw_text,
            "experience_years": features.get("years_of_experience", 0) or 0,
            "certifications": features.get("sections", {}).get("has_certifications", False),
            "num_positions": 1,
        }

        # Build job features (auto-extracted from JD)
        job_features = {
            "skills_required": job_skills,
            "edu_requirement": job_education,
            "experience_requirement": job_experience,
            "responsibilities": job_description_raw,
        }

        # Run ML prediction
        result = predict_match(resume_features, job_features)

        return render_template(
            "match_results.html",
            result=result,
            job_title=job_title,
            resume_skills=resume_skills,
            jd_parsed=jd_parsed,
        )

    except Exception as e:
        flash(f"An error occurred: {str(e)}", "error")
        return redirect(url_for("job_matcher"))

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route("/interview-prep")
@login_required
def interview_prep():
    """Interview Prep Page"""
    return render_template("interview_prep.html")

import subprocess
import os
import sys

@app.route("/launch-interview", methods=["POST"])
@login_required
def launch_interview():
    """Prepare the interview session and point the app to the local interview project."""
    interview_project_dir = os.path.join(BASE_DIR, "Project_Smart_Interview")
    report_file_path = os.path.join(interview_project_dir, "Report_Smart_Interview.txt")

    if os.path.exists(report_file_path):
        os.remove(report_file_path)

    duration_minutes = request.form.get('duration', '2')
    speaker_count = request.form.get('speakers', '1')
    interview_language = request.form.get('language', 'English')

    try:
        session['interview_report_path'] = report_file_path
        session['interview_duration'] = duration_minutes
        session['interview_speakers'] = speaker_count
        session['interview_language'] = interview_language
        session['interview_project_dir'] = interview_project_dir
        return redirect(url_for('interview_waiting'))

    except Exception as e:
        flash(f"Error launching interview: {str(e)}", "error")
        return redirect(url_for('interview_prep'))


@app.route("/video_feed")
@login_required
def video_feed():
    from flask import Response
    if stream_interview is None:
        # Return a simple error message if the streamer module couldn't be loaded
        return Response(
            b'Camera module unavailable. Please check server logs.',
            status=503,
            mimetype='text/plain'
        )
    duration_minutes = float(session.get('interview_duration', '2'))
    speaker_count = int(session.get('interview_speakers', 1) or 1)
    interview_language = session.get('interview_language', 'English')
    return Response(
        stream_interview(duration_minutes, num_speakers=speaker_count, language=interview_language),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route("/interview-waiting")
@login_required
def interview_waiting():
    """Waiting page shown while the camera interview is in progress."""
    duration = session.get('interview_duration', '2')
    speaker_count = session.get('interview_speakers', '1')
    interview_language = session.get('interview_language', 'English')
    duration_secs = int(duration) * 60
    # One question every 50 seconds, max 15
    total_questions = min(15, max(1, duration_secs // 50))
    return render_template(
        "interview_waiting.html",
        duration=duration,
        total_questions=total_questions,
        speakers=speaker_count,
        interview_language=interview_language,
    )


@app.route("/interview-check-report")
@login_required
def interview_check_report():
    """AJAX endpoint: check if the interview report file is ready."""
    report_path = session.get('interview_report_path', '')
    if report_path and os.path.exists(report_path):
        # Report is ready! Parse it and store results in session
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                content = f.read()
            parsed_report = parse_interview_report(content)
            parsed_report['interview_language'] = session.get('interview_language', 'English')
            advice = generate_advice(parsed_report)
            duration = parsed_report.get('session_duration', 60)
            # Store in session for the dashboard to read
            session['interview_advice'] = advice
            session['interview_score_duration'] = duration
            return {"ready": True}
        except Exception as e:
            return {"ready": False, "error": str(e)}
    return {"ready": False}


@app.route("/interview-results")
@login_required
def interview_results():
    """Dashboard page shown after interview report is parsed."""
    import os
    from services.interview_advisor import generate_advice
    from services.interview_report_parser import parse_interview_report
    report_path = os.path.join(app.root_path, "Project_Smart_Interview", "Report_Smart_Interview.txt")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
            parsed_report = parse_interview_report(content)
            parsed_report['interview_language'] = session.get('interview_language', 'English')
            session['interview_advice'] = generate_advice(parsed_report)
        
    advice = session.get('interview_advice')
    duration = session.get('interview_score_duration', 60)
    interview_language = session.get('interview_language', 'English')
    if not advice:
        flash("No interview results found. Please run an interview first.", "warning")
        return redirect(url_for('interview_prep'))
    return render_template(
        "interview_dashboard.html",
        advice=advice,
        duration=duration,
        interview_language=interview_language,
    )

@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    """API endpoint for the AI Chatbot."""
    data = request.json
    user_message = data.get("message", "")
    chat_history = data.get("history", [])
    
    report_data = session.get("interview_advice", {})
    interview_language = session.get("interview_language", "English")
    
    if not user_message:
        if interview_language == "Arabic":
            return {"reply": "يرجى إرسال رسالة صالحة."}
        return {"reply": "Please send a valid message."}

    ai_reply = get_ai_coach_response(user_message, report_data, chat_history, language=interview_language)
    return {"reply": ai_reply}

@app.route("/upload", methods=["POST"])
@login_required
def upload():
    """
    Handle file upload and run the full ATS analysis pipeline.
    """
    # --- Validate upload ---
    if "resume" not in request.files:
        flash("No file selected. Please upload a PDF resume.", "error")
        return redirect(url_for("ats_checker"))

    file = request.files["resume"]

    if file.filename == "":
        flash("No file selected. Please upload a PDF resume.", "error")
        return redirect(url_for("ats_checker"))

    if not allowed_file(file.filename):
        flash("Invalid file type. Only PDF files are accepted.", "error")
        return redirect(url_for("ats_checker"))

    # --- Save uploaded file ---
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    try:
        # ── Step 1: Extract text & metadata from PDF ──
        pdf_result = parse_pdf(file_path)
        raw_text = pdf_result["text"]
        pdf_metadata = pdf_result["metadata"]

        if not raw_text.strip():
            flash("Could not extract text from this PDF. It may be image-based. Please use a text-based PDF.", "error")
            return redirect(url_for("ats_checker"))

        # ── Steps 2-4: NLP → Features → Score ──
        features, ats_result = analyze_text(raw_text, pdf_metadata)

        # ── Step 5: Render dashboard ──
        return render_template(
            "dashboard.html",
            filename=filename,
            features=features,
            ats=ats_result,
        )

    except Exception as e:
        flash(f"An error occurred while processing your resume: {str(e)}", "error")
        return redirect(url_for("ats_checker"))

    finally:
        # Clean up uploaded file after processing
        if os.path.exists(file_path):
            os.remove(file_path)


@app.route("/analyze-sample", methods=["POST"])
@login_required
def analyze_sample():
    """
    Analyze a sample resume from the Datasets/test_resumes.json file.
    No PDF upload needed — uses the resume_text directly.
    """
    sample_index = request.form.get("sample_index")

    if sample_index is None:
        flash("No sample selected.", "error")
        return redirect(url_for("ats_checker"))

    try:
        sample_index = int(sample_index)
        samples = load_test_resumes()

        if sample_index < 0 or sample_index >= len(samples):
            flash("Invalid sample selection.", "error")
            return redirect(url_for("ats_checker"))

        sample = samples[sample_index]
        raw_text = sample.get("resume_text", "")
        name = sample.get("name", f"Sample #{sample_index + 1}")

        if not raw_text.strip():
            flash("This sample resume has no text content.", "error")
            return redirect(url_for("ats_checker"))

        # Run analysis pipeline (no PDF metadata since it's text-based)
        features, ats_result = analyze_text(raw_text)

        return render_template(
            "dashboard.html",
            filename=f"Sample: {name}",
            features=features,
            ats=ats_result,
            sample_info=sample,
        )

    except Exception as e:
        flash(f"An error occurred while analyzing the sample: {str(e)}", "error")
        return redirect(url_for("ats_checker"))


# ═══════════════════════════════════════════════════════════════════════════
#  FUTURE: Resume-Job Matching Endpoint (placeholder)
# ═══════════════════════════════════════════════════════════════════════════
#
# @app.route("/match", methods=["POST"])
# def match():
#     """
#     Compare a resume against a job description.
#     Accepts both a resume PDF and a job description text.
#     Returns a match score and keyword analysis.
#     """
#     pass
#


# ═══════════════════════════════════════════════════════════════════════════
#  RUN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000)
