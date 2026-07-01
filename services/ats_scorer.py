"""
ats_scorer.py - Weighted Rule-Based ATS Scoring Engine

Scoring weights (total = 100):
    Contact Information : 15
    Resume Structure    : 25
    Skills              : 25
    Experience          : 20
    Education           : 10
    Formatting          :  5

No Machine Learning — purely rule-based scoring.
Generates per-category scores, an overall ATS score,
detected/missing items, and actionable recommendations.
"""

import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
#  CATEGORY SCORERS
# ═══════════════════════════════════════════════════════════════════════════

def score_contact(contact: dict) -> dict:
    """
    Contact Information — max 15 points.

    Breakdown:
        has_name      : 3
        has_email     : 3
        has_phone     : 3
        has_linkedin  : 2
        has_github    : 2
        has_location  : 2
    """
    breakdown = {
        "has_name":     {"score": 3 if contact["has_name"]     else 0, "max": 3},
        "has_email":    {"score": 3 if contact["has_email"]    else 0, "max": 3},
        "has_phone":    {"score": 3 if contact["has_phone"]    else 0, "max": 3},
        "has_linkedin": {"score": 2 if contact["has_linkedin"] else 0, "max": 2},
        "has_github":   {"score": 2 if contact["has_github"]   else 0, "max": 2},
        "has_location": {"score": 2 if contact["has_location"] else 0, "max": 2},
    }
    total = sum(item["score"] for item in breakdown.values())
    return {"score": total, "max": 15, "breakdown": breakdown}


def score_structure(sections: dict) -> dict:
    """
    Resume Structure — max 25 points.

    Breakdown:
        has_summary        : 3
        has_education      : 4
        has_experience     : 5
        has_skills         : 5
        has_projects       : 3
        has_certifications : 2
        has_languages      : 1.5
        has_achievements   : 1.5
    """
    weights = {
        "has_summary":        3,
        "has_education":      4,
        "has_experience":     5,
        "has_skills":         5,
        "has_projects":       3,
        "has_certifications": 2,
        "has_languages":      1.5,
        "has_achievements":   1.5,
    }
    breakdown = {}
    for key, max_val in weights.items():
        breakdown[key] = {
            "score": max_val if sections.get(key, False) else 0,
            "max": max_val,
        }
    total = sum(item["score"] for item in breakdown.values())
    return {"score": total, "max": 25, "breakdown": breakdown}


def score_skills(features: dict) -> dict:
    """
    Skills — max 25 points.

    Breakdown:
        technical_skills_count (scaled, max 20):
            0 skills      → 0
            1-3 skills    → 8
            4-7 skills    → 15
            8+ skills     → 20
        soft_skills_count (scaled, max 5):
            0             → 0
            1-2           → 2
            3+            → 5
    """
    tech_count = features.get("technical_skills_count", 0)
    soft_count = features.get("soft_skills_count", 0)

    # Technical skills score
    if tech_count >= 8:
        tech_score = 20
    elif tech_count >= 4:
        tech_score = 15
    elif tech_count >= 1:
        tech_score = 8
    else:
        tech_score = 0

    # Soft skills score
    if soft_count >= 3:
        soft_score = 5
    elif soft_count >= 1:
        soft_score = 2
    else:
        soft_score = 0

    breakdown = {
        "technical_skills": {"score": tech_score, "max": 20, "count": tech_count},
        "soft_skills":      {"score": soft_score, "max": 5,  "count": soft_count},
    }
    total = tech_score + soft_score
    return {"score": total, "max": 25, "breakdown": breakdown}


def score_experience(features: dict) -> dict:
    """
    Experience — max 20 points.

    Breakdown:
        has_experience section  : 8
        has_projects section    : 5
        years_of_experience     : 4 (1-2 yrs→2, 3+→4)
        word_count quality      : 3 (200-300→1, 300-600→2, 600+→3)
    """
    sections = features.get("sections", {})
    years = features.get("years_of_experience")
    word_count = features.get("word_count", 0)

    exp_score = 8 if sections.get("has_experience", False) else 0
    proj_score = 5 if sections.get("has_projects", False) else 0

    if years and years >= 3:
        years_score = 4
    elif years and years >= 1:
        years_score = 2
    else:
        years_score = 0

    if word_count >= 600:
        wc_score = 3
    elif word_count >= 300:
        wc_score = 2
    elif word_count >= 200:
        wc_score = 1
    else:
        wc_score = 0

    breakdown = {
        "has_experience":      {"score": exp_score,   "max": 8},
        "has_projects":        {"score": proj_score,  "max": 5},
        "years_of_experience": {"score": years_score, "max": 4, "years": years},
        "content_depth":       {"score": wc_score,    "max": 3, "word_count": word_count},
    }
    total = exp_score + proj_score + years_score + wc_score
    return {"score": total, "max": 20, "breakdown": breakdown}


def score_education(features: dict) -> dict:
    """
    Education — max 10 points.

    Breakdown:
        has_education section : 4
        has_degree            : 3
        degree_type bonus     : 3 (PhD→3, Master→2, Bachelor→1.5, Diploma→1)
    """
    sections = features.get("sections", {})
    edu = features.get("education", {})

    edu_section = 4 if sections.get("has_education", False) else 0
    has_degree_score = 3 if edu.get("has_degree", False) else 0

    degree_type = edu.get("degree_type", "")
    if degree_type == "Phd":
        degree_bonus = 3
    elif degree_type == "Master":
        degree_bonus = 2
    elif degree_type == "Bachelor":
        degree_bonus = 1.5
    elif degree_type == "Diploma":
        degree_bonus = 1
    else:
        degree_bonus = 0

    breakdown = {
        "has_education_section": {"score": edu_section,     "max": 4},
        "has_degree":            {"score": has_degree_score, "max": 3},
        "degree_bonus":          {"score": degree_bonus,     "max": 3, "type": degree_type},
    }
    total = edu_section + has_degree_score + degree_bonus
    return {"score": min(total, 10), "max": 10, "breakdown": breakdown}


def score_formatting(formatting: dict) -> dict:
    """
    Formatting — max 5 points.

    Start with 5 and deduct for ATS-unfriendly formatting:
        has_image            : -1.5
        has_table            : -1
        multi_column_layout  : -1.5
        resume too long      : -1
    """
    score = 5.0

    penalties = {}

    if formatting.get("has_image", False):
        score -= 1.5
        penalties["has_image"] = -1.5

    if formatting.get("has_table", False):
        score -= 1.0
        penalties["has_table"] = -1.0

    if formatting.get("multi_column_layout", False):
        score -= 1.5
        penalties["multi_column_layout"] = -1.5

    length = formatting.get("resume_length_score", "good")
    if length == "too_long":
        score -= 1.0
        penalties["too_long"] = -1.0

    score = max(score, 0)

    breakdown = {
        "base_score": 5,
        "penalties": penalties,
        "page_count": formatting.get("page_count", 0),
        "resume_length": length,
    }
    return {"score": score, "max": 5, "breakdown": breakdown}


# ═══════════════════════════════════════════════════════════════════════════
#  RECOMMENDATIONS ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def generate_recommendations(features: dict, category_scores: dict) -> list:
    """
    Generate actionable recommendations based on missing or weak features.
    """
    recs = []
    contact = features.get("contact", {})
    sections = features.get("sections", {})
    formatting = features.get("formatting", {})

    # --- Contact ---
    if not contact.get("has_name"):
        recs.append("Add your full name at the top of your resume.")
    if not contact.get("has_email"):
        recs.append("Include a professional email address.")
    if not contact.get("has_phone"):
        recs.append("Add a phone number for recruiters to reach you.")
    if not contact.get("has_linkedin"):
        recs.append("Add your LinkedIn profile URL.")
    if not contact.get("has_github"):
        recs.append("Add your GitHub profile to showcase your work (especially for tech roles).")
    if not contact.get("has_location"):
        recs.append("Include your city/country location.")

    # --- Sections ---
    section_labels = {
        "has_summary":        "a Summary / Objective section",
        "has_education":      "an Education section",
        "has_experience":     "a Work Experience section",
        "has_skills":         "a Skills section",
        "has_projects":       "a Projects section to showcase practical work",
        "has_certifications": "a Certifications section",
        "has_languages":      "a Languages section",
        "has_achievements":   "an Achievements / Awards section",
    }
    for key, label in section_labels.items():
        if not sections.get(key, False):
            recs.append(f"Add {label}.")

    # --- Skills ---
    tech_count = features.get("technical_skills_count", 0)
    soft_count = features.get("soft_skills_count", 0)

    if tech_count < 4:
        recs.append(f"Add more technical skills — only {tech_count} detected. Aim for 8+.")
    if soft_count == 0:
        recs.append("Include soft skills like Communication, Leadership, and Teamwork.")

    # --- Content depth ---
    word_count = features.get("word_count", 0)
    if word_count < 200:
        recs.append("Your resume seems too short. Add more detail to your experience and projects.")
    elif word_count < 300:
        recs.append("Consider adding more content to better showcase your qualifications.")

    # --- Formatting ---
    if formatting.get("has_image"):
        recs.append("Remove images — most ATS systems cannot read them.")
    if formatting.get("has_table"):
        recs.append("Avoid using tables — they may cause ATS parsing errors.")
    if formatting.get("multi_column_layout"):
        recs.append("Switch to a single-column layout for better ATS compatibility.")
    if formatting.get("resume_length_score") == "too_long":
        recs.append(f"Your resume is {formatting.get('page_count', '?')} pages. Consider trimming to 1-2 pages.")

    # --- Education ---
    if not features.get("education", {}).get("has_degree"):
        recs.append("Explicitly mention your degree (e.g., Bachelor of Science in Computer Science).")

    return recs


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN SCORING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def calculate_ats_score(features: dict) -> dict:
    """
    Main entry point — calculate the full ATS score from extracted features.

    Returns:
        {
            "total_score":     float (0-100),
            "total_max":       100,
            "percentage":      int,
            "grade":           str ("Excellent" | "Good" | "Fair" | "Poor"),
            "categories": {
                "contact":    { score, max, breakdown },
                "structure":  { score, max, breakdown },
                "skills":     { score, max, breakdown },
                "experience": { score, max, breakdown },
                "education":  { score, max, breakdown },
                "formatting": { score, max, breakdown },
            },
            "recommendations": [str, ...],
            "summary_df":      pd.DataFrame (for tabular display),
        }
    """
    # Calculate each category
    categories = {
        "contact":    score_contact(features.get("contact", {})),
        "structure":  score_structure(features.get("sections", {})),
        "skills":     score_skills(features),
        "experience": score_experience(features),
        "education":  score_education(features),
        "formatting": score_formatting(features.get("formatting", {})),
    }

    # Total
    total_score = sum(cat["score"] for cat in categories.values())
    total_max = sum(cat["max"] for cat in categories.values())
    percentage = round((total_score / total_max) * 100) if total_max > 0 else 0

    # Grade
    if percentage >= 85:
        grade = "Excellent"
    elif percentage >= 70:
        grade = "Good"
    elif percentage >= 50:
        grade = "Fair"
    else:
        grade = "Poor"

    # Recommendations
    recommendations = generate_recommendations(features, categories)

    # Summary DataFrame (for optional pandas display / export)
    summary_data = []
    category_labels = {
        "contact":    "Contact Information",
        "structure":  "Resume Structure",
        "skills":     "Skills",
        "experience": "Experience",
        "education":  "Education",
        "formatting": "Formatting",
    }
    for key, label in category_labels.items():
        cat = categories[key]
        summary_data.append({
            "Category":  label,
            "Score":     cat["score"],
            "Max":       cat["max"],
            "Result":    f"{cat['score']}/{cat['max']}",
        })
    summary_data.append({
        "Category": "TOTAL",
        "Score":    total_score,
        "Max":      total_max,
        "Result":   f"{total_score}/{total_max}",
    })
    summary_df = pd.DataFrame(summary_data)

    return {
        "total_score":      total_score,
        "total_max":        total_max,
        "percentage":       percentage,
        "grade":            grade,
        "categories":       categories,
        "recommendations":  recommendations,
        "summary_df":       summary_df,
    }
