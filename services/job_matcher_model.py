"""
job_matcher_model.py — Machine Learning Pipeline for Resume-Job Matching

Full ML Pipeline:
═══════════════════════════════════════════════════════════════════
Step 1: Data Loading
    → Load resume_data_for_ranking.csv (9,544 rows)

Step 2: NLP Preprocessing
    → Clean text: lowercase, remove special chars, strip whitespace
    → Parse list-like string columns (skills, degree_names, etc.)

Step 3: Feature Engineering (4 Categories)
    ┌─────────────────────┬──────────────────────────────────────────┐
    │  Feature            │  Method                                  │
    ├─────────────────────┼──────────────────────────────────────────┤
    │  Text Similarity    │  TF-IDF on responsibilities + job desc   │
    │                     │  → Cosine Similarity                     │
    │  Skills Overlap     │  Jaccard similarity between resume       │
    │                     │  skills and job required skills           │
    │  Education Match    │  Ordinal encoding of degree level,       │
    │                     │  compare resume vs requirement            │
    │  Experience Match   │  Parse years from resume dates &         │
    │                     │  job requirement, compute difference      │
    │  Certification      │  Binary: has certifications or not       │
    └─────────────────────┴──────────────────────────────────────────┘

Step 4: Model Training
    → Algorithm: Gradient Boosting Regressor
    → Split: 80% Train / 20% Test
    → Target: matched_score (0.0 – 1.0)

Step 5: Evaluation
    → MAE, RMSE, R² Score

Step 6: Save Model
    → joblib.dump() for production use in Flask
═══════════════════════════════════════════════════════════════════
"""

import os
import re
import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(BASE_DIR, "Datasets")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
#  STEP 1: DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def load_data():
    """Load the resume ranking dataset."""
    path = os.path.join(DATASETS_DIR, "resume_data_for_ranking.csv")
    df = pd.read_csv(path)
    print(f"[job_matcher] Loaded {len(df)} rows from resume_data_for_ranking.csv")
    return df


# ═══════════════════════════════════════════════════════════════════
#  STEP 2: NLP PREPROCESSING
# ═══════════════════════════════════════════════════════════════════

def clean_text(text):
    """Normalize and clean text."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s/+#.\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def safe_parse_list(val):
    """Safely parse a stringified Python list like \"['a', 'b']\" into a real list."""
    if pd.isna(val) or not isinstance(val, str):
        return []
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return [str(item).strip().lower() for item in parsed if str(item).strip() and str(item).strip().lower() != 'n/a']
        return []
    except (ValueError, SyntaxError):
        # Fallback: split by comma or pipe
        items = re.split(r'[,|]', val)
        return [item.strip().lower() for item in items if item.strip() and item.strip().lower() != 'n/a']


# ═══════════════════════════════════════════════════════════════════
#  STEP 3: FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════

# -- 3a: Text Similarity (TF-IDF + Cosine) --

def compute_text_similarity(resume_texts, job_texts):
    """
    Compute cosine similarity between resume responsibilities
    and job responsibilities using TF-IDF.
    """
    similarities = []
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')

    for res_text, job_text in zip(resume_texts, job_texts):
        res_clean = clean_text(res_text)
        job_clean = clean_text(job_text)

        if not res_clean or not job_clean:
            similarities.append(0.0)
            continue

        try:
            matrix = tfidf.fit_transform([res_clean, job_clean])
            sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
            similarities.append(float(sim))
        except Exception:
            similarities.append(0.0)

    return similarities


# -- 3b: Skills Overlap (Jaccard Similarity) --

def compute_skills_overlap(resume_skills_col, job_skills_col):
    """
    Compute Jaccard similarity between resume skills and job required skills.
    Returns overlap ratio and counts.
    """
    overlaps = []
    matched_counts = []
    missing_counts = []

    for res_skills, job_skills in zip(resume_skills_col, job_skills_col):
        res_set = set(safe_parse_list(res_skills))
        job_set = set(safe_parse_list(job_skills))

        if not job_set:
            overlaps.append(0.0)
            matched_counts.append(0)
            missing_counts.append(0)
            continue

        matched = res_set.intersection(job_set)
        overlap = len(matched) / len(job_set) if job_set else 0.0
        overlaps.append(overlap)
        matched_counts.append(len(matched))
        missing_counts.append(len(job_set) - len(matched))

    return overlaps, matched_counts, missing_counts


# -- 3c: Education Match --

DEGREE_HIERARCHY = {
    'phd': 5, 'doctorate': 5, 'doctor': 5,
    'master': 4, 'mba': 4, 'm.sc': 4, 'msc': 4, 'm.s.': 4,
    'bachelor': 3, 'b.sc': 3, 'bsc': 3, 'b.tech': 3, 'b.a.': 3, 'b.eng': 3,
    'diploma': 2, 'associate': 2, 'apprenticeship': 2, 'certification': 2,
    'high school': 1, 'secondary': 1, 'hsc': 1, 'ssc': 1,
}


def get_degree_level(text):
    """Map a degree string to an ordinal level."""
    if pd.isna(text) or not isinstance(text, str):
        return 0
    text_lower = text.lower()
    for keyword, level in DEGREE_HIERARCHY.items():
        if keyword in text_lower:
            return level
    return 0


def compute_education_match(resume_degrees, job_edu_req):
    """Compare resume degree level vs job education requirement."""
    scores = []
    for res_deg, job_req in zip(resume_degrees, job_edu_req):
        res_level = get_degree_level(str(res_deg))
        job_level = get_degree_level(str(job_req))

        if job_level == 0:
            scores.append(1.0)  # No requirement specified
        elif res_level >= job_level:
            scores.append(1.0)  # Meets or exceeds
        elif res_level == job_level - 1:
            scores.append(0.5)  # One level below
        else:
            scores.append(0.0)  # Significantly under-qualified

    return scores


# -- 3d: Experience Match --

def parse_experience_years(text):
    """Extract years of experience from text like 'At least 5 year(s)' or dates."""
    if pd.isna(text) or not isinstance(text, str):
        return 0
    match = re.search(r'(\d+)\s*(?:year|yr|years)', text.lower())
    if match:
        return int(match.group(1))
    match = re.search(r'(\d+)', text)
    if match:
        return int(match.group(1))
    return 0


def compute_experience_from_dates(start_dates_str, end_dates_str):
    """Estimate total experience from start/end date columns."""
    starts = safe_parse_list(start_dates_str)
    ends = safe_parse_list(end_dates_str)

    total_years = 0
    for start, end in zip(starts, ends):
        start_year = re.search(r'(\d{4})', start)
        end_year = re.search(r'(\d{4})', end)

        if start_year:
            sy = int(start_year.group(1))
            if end_year:
                ey = int(end_year.group(1))
            else:
                ey = 2026  # "Till Date"
            total_years += max(0, ey - sy)

    return total_years


def compute_experience_match(df):
    """Compare resume experience years vs job requirement."""
    scores = []
    for _, row in df.iterrows():
        resume_years = compute_experience_from_dates(
            row.get('start_dates', ''), row.get('end_dates', '')
        )
        required_years = parse_experience_years(str(row.get('experiencere_requirement', '')))

        if required_years == 0:
            scores.append(1.0)
        elif resume_years >= required_years:
            scores.append(1.0)
        elif resume_years >= required_years * 0.5:
            scores.append(0.5)
        else:
            scores.append(0.0)

    return scores


# -- 3e: Certification Feature --

def compute_certification_feature(cert_col):
    """Binary feature: does the candidate have certifications?"""
    return [1.0 if (not pd.isna(val) and str(val).strip() and str(val).strip().lower() != 'nan') else 0.0 for val in cert_col]


# ═══════════════════════════════════════════════════════════════════
#  BUILD FEATURE MATRIX
# ═══════════════════════════════════════════════════════════════════

def build_features(df):
    """Engineer all features and return X matrix and y target."""
    print("[job_matcher] Engineering features...")

    # 3a: Text similarity
    text_sim = compute_text_similarity(
        df['responsibilities'].fillna(''),
        df['responsibilities.1'].fillna('')
    )

    # 3b: Skills overlap
    skills_overlap, matched_count, missing_count = compute_skills_overlap(
        df['skills'], df['skills_required']
    )

    # 3c: Education match
    edu_match = compute_education_match(
        df['degree_names'], df['educationaL_requirements']
    )

    # 3d: Experience match
    exp_match = compute_experience_match(df)

    # 3e: Certifications
    cert_feature = compute_certification_feature(df['certification_providers'])

    # 3f: Additional numeric features
    num_skills = [len(safe_parse_list(s)) for s in df['skills']]
    num_positions = [len(safe_parse_list(p)) for p in df['positions']]

    # Combine all features
    feature_df = pd.DataFrame({
        'text_similarity': text_sim,
        'skills_overlap': skills_overlap,
        'matched_skills_count': matched_count,
        'missing_skills_count': missing_count,
        'education_match': edu_match,
        'experience_match': exp_match,
        'has_certification': cert_feature,
        'num_resume_skills': num_skills,
        'num_positions': num_positions,
    })

    y = df['matched_score'].values

    print(f"[job_matcher] Feature matrix shape: {feature_df.shape}")
    return feature_df, y


# ═══════════════════════════════════════════════════════════════════
#  STEP 4: MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════

def train_model():
    """Full training pipeline: load → features → train → evaluate → save."""
    # Load
    df = load_data()

    # Build features
    X, y = build_features(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"[job_matcher] Train: {len(X_train)}, Test: {len(X_test)}")

    # Train
    print("[job_matcher] Training Gradient Boosting Regressor...")
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=10,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  Model Evaluation Results")
    print(f"{'='*50}")
    print(f"  MAE  (Mean Absolute Error):  {mae:.4f}")
    print(f"  RMSE (Root Mean Sq Error):   {rmse:.4f}")
    print(f"  R²   (R-Squared Score):      {r2:.4f}")
    print(f"{'='*50}")

    # Feature importance
    print("\n  Feature Importances:")
    for name, imp in sorted(zip(X.columns, model.feature_importances_), key=lambda x: -x[1]):
        print(f"    {name:30s} -> {imp:.4f}")

    # Save model
    model_path = os.path.join(MODEL_DIR, "job_matcher_model.pkl")
    joblib.dump(model, model_path)
    print(f"\n[job_matcher] Model saved to {model_path}")

    return model


# ═══════════════════════════════════════════════════════════════════
#  STEP 5: PREDICTION (Used by Flask)
# ═══════════════════════════════════════════════════════════════════

def load_model():
    """Load the trained model from disk."""
    model_path = os.path.join(MODEL_DIR, "job_matcher_model.pkl")
    if not os.path.exists(model_path):
        print("[job_matcher] No trained model found. Training now...")
        return train_model()
    return joblib.load(model_path)


def predict_match(resume_features: dict, job_features: dict) -> dict:
    """
    Predict match score between a resume and a job description.

    Args:
        resume_features: dict with keys: skills, degree, responsibilities,
                         start_dates, end_dates, certifications
        job_features: dict with keys: skills_required, edu_requirement,
                      experience_requirement, responsibilities

    Returns:
        dict with: score, decision, matched_skills, missing_skills, recommendations
    """
    model = load_model()

    # Parse inputs
    resume_skills = set([s.strip().lower() for s in resume_features.get('skills', []) if s.strip()])
    job_skills = set([s.strip().lower() for s in job_features.get('skills_required', []) if s.strip()])

    matched_skills = sorted(resume_skills.intersection(job_skills))
    missing_skills = sorted(job_skills - resume_skills)

    # Build feature vector
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    res_text = clean_text(resume_features.get('responsibilities', ''))
    job_text = clean_text(job_features.get('responsibilities', ''))

    if res_text and job_text:
        matrix = tfidf.fit_transform([res_text, job_text])
        text_sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    else:
        text_sim = 0.0

    skills_overlap = len(matched_skills) / len(job_skills) if job_skills else 0.0

    res_degree_level = get_degree_level(str(resume_features.get('degree', '')))
    job_degree_level = get_degree_level(str(job_features.get('edu_requirement', '')))
    if job_degree_level == 0:
        edu_match = 1.0
    elif res_degree_level >= job_degree_level:
        edu_match = 1.0
    elif res_degree_level == job_degree_level - 1:
        edu_match = 0.5
    else:
        edu_match = 0.0

    resume_years = resume_features.get('experience_years', 0)
    required_years = parse_experience_years(str(job_features.get('experience_requirement', '')))
    if required_years == 0:
        exp_match = 1.0
    elif resume_years >= required_years:
        exp_match = 1.0
    elif resume_years >= required_years * 0.5:
        exp_match = 0.5
    else:
        exp_match = 0.0

    has_cert = 1.0 if resume_features.get('certifications') else 0.0

    feature_vector = pd.DataFrame([{
        'text_similarity': text_sim,
        'skills_overlap': skills_overlap,
        'matched_skills_count': len(matched_skills),
        'missing_skills_count': len(missing_skills),
        'education_match': edu_match,
        'experience_match': exp_match,
        'has_certification': has_cert,
        'num_resume_skills': len(resume_skills),
        'num_positions': resume_features.get('num_positions', 1),
    }])

    # Predict
    raw_score = model.predict(feature_vector)[0]
    score_pct = round(max(0, min(100, raw_score * 100)), 1)
    decision = "Suitable" if score_pct >= 70 else "Not Suitable"

    # Recommendations
    recommendations = []
    if missing_skills:
        top_missing = missing_skills[:5]
        recommendations.append(f"Add these missing skills to your resume: {', '.join(top_missing)}")
    if edu_match < 1.0:
        recommendations.append("Your education level is below the job requirement. Consider adding relevant certifications.")
    if exp_match < 1.0:
        recommendations.append(f"The job requires {required_years}+ years of experience. Highlight all relevant experience and projects.")
    if text_sim < 0.3:
        recommendations.append("Your resume responsibilities don't closely match the job description. Tailor your resume to use similar keywords.")
    if has_cert == 0:
        recommendations.append("Consider adding professional certifications to strengthen your profile.")
    if not recommendations:
        recommendations.append("Your resume is a strong match for this job! Consider applying.")

    return {
        'score': score_pct,
        'decision': decision,
        'matched_skills': matched_skills,
        'missing_skills': missing_skills,
        'skills_overlap_pct': round(skills_overlap * 100, 1),
        'education_match': edu_match,
        'experience_match': exp_match,
        'text_similarity': round(text_sim * 100, 1),
        'recommendations': recommendations,
    }


# ═══════════════════════════════════════════════════════════════════
#  MAIN — Run Training
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train_model()
