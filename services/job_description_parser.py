"""
job_description_parser.py — Extract structured info from raw Job Description text.

Given a single block of text (the full job description), this module extracts:
    1. Job Title
    2. Required Skills
    3. Education Requirement
    4. Experience Requirement
"""

import re
import os
import json
import csv

# ═══════════════════════════════════════════════════════════════════
#  LOAD DATASETS (same as feature_extractor)
# ═══════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(BASE_DIR, "Datasets")


def _load_all_skills():
    """Load all known skills from datasets."""
    skills = []

    # From skills_database.json
    path = os.path.join(DATASETS_DIR, "skills_database.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            db = json.load(f)
        for cat, cat_skills in db.items():
            for s in cat_skills:
                if s not in skills:
                    skills.append(s)
    except FileNotFoundError:
        pass

    # From skills_list.csv
    path = os.path.join(DATASETS_DIR, "skills_list.csv")
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("Skill Name", "").strip()
                if name and name not in skills:
                    skills.append(name)
    except FileNotFoundError:
        pass

    # From job_roles.csv
    path = os.path.join(DATASETS_DIR, "job_roles.csv")
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for skill in row.get("Required Skills", "").split("|"):
                    skill = skill.strip()
                    if skill and skill not in skills:
                        skills.append(skill)
    except FileNotFoundError:
        pass

    return skills


def _load_job_titles():
    """Load all job titles from job_roles.csv."""
    titles = []
    path = os.path.join(DATASETS_DIR, "job_roles.csv")
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                title = row.get("Job Role", "").strip()
                if title and title not in titles:
                    titles.append(title)
    except FileNotFoundError:
        pass
    return titles


# Load once at module level
ALL_SKILLS = _load_all_skills()
ALL_JOB_TITLES = _load_job_titles()

print(f"[jd_parser] Loaded {len(ALL_SKILLS)} skills and {len(ALL_JOB_TITLES)} job titles for JD parsing")


# ═══════════════════════════════════════════════════════════════════
#  EXTRACTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def extract_job_title_from_jd(text):
    """
    Extract the job title from the JD text.
    Strategy:
        1. Look for patterns like "Job Title: ...", "Position: ...", "Role: ..."
        2. Match against known job titles from dataset
        3. Check the first few lines (often the title is at the top)
    """
    # Strategy 1: Explicit label patterns
    label_patterns = [
        r'(?:job\s*title|position|role|designation)\s*[:\-]\s*(.+)',
        r'(?:hiring|looking\s+for|seeking)\s+(?:a\s+)?(.+?)(?:\.|,|\n|$)',
        r'(?:we\s+are\s+looking\s+for\s+(?:a\s+|an\s+)?)(.+?)(?:\.|,|\n|$)',
    ]
    for pattern in label_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            if 2 <= len(candidate.split()) <= 6:
                return candidate.title()

    # Strategy 2: Match against dataset
    text_lower = text.lower()
    best_match = None
    best_len = 0
    for title in ALL_JOB_TITLES:
        if title.lower() in text_lower and len(title) > best_len:
            best_match = title
            best_len = len(title)
    if best_match:
        return best_match

    # Strategy 3: First non-empty short line with Job Title keywords
    lines = text.strip().split('\n')
    title_keywords = ["engineer", "developer", "manager", "designer", "architect", 
                      "analyst", "specialist", "director", "consultant", "administrator", 
                      "scientist", "technician", "lead", "head", "executive", "officer", 
                      "coordinator", "representative"]
    
    for line in lines[:8]:
        line = line.strip()
        line_lower = line.lower()
        if not line or len(line.split()) > 6 or re.search(r'[@:/]', line):
            continue
            
        # Only accept this line if it actually contains a word typical for job titles
        if any(keyword in line_lower for keyword in title_keywords):
            return line.title()

    return None


def extract_skills_from_jd(text):
    """
    Extract required skills from the JD text by matching against the skills database.
    """
    found = []
    for skill in ALL_SKILLS:
        escaped = re.escape(skill)
        pattern = r'(?<![a-zA-Z])' + escaped + r'(?![a-zA-Z])'
        if re.search(pattern, text, re.IGNORECASE):
            if skill not in found:
                found.append(skill)
    return found


def extract_education_from_jd(text):
    """
    Extract education requirement from the JD text.
    """
    degree_patterns = {
        "PhD": [r"ph\.?d", r"doctorate", r"doctor\s+of\s+philosophy"],
        "Master's Degree": [r"master'?s?\s+(?:degree)?", r"m\.?s\.?c?\.?", r"mba", r"m\.?eng"],
        "Bachelor's Degree": [r"bachelor'?s?\s+(?:degree)?", r"b\.?s\.?c?\.?", r"b\.?tech", r"b\.?eng", r"b\.?a\.?"],
        "Diploma": [r"diploma", r"associate\s+degree"],
    }

    text_lower = text.lower()
    for degree_name, patterns in degree_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                # Try to capture the full context
                full_pattern = r'(?:' + pattern + r')[\s,]*(?:in\s+)?([A-Za-z\s&]+?)(?:\.|,|\n|or|with|\d|$)'
                match = re.search(full_pattern, text_lower)
                if match:
                    field = match.group(1).strip().title()
                    if len(field) > 2 and len(field) < 50:
                        return f"{degree_name} in {field}"
                return degree_name

    return ""


def extract_experience_from_jd(text):
    """
    Extract experience requirement from the JD text.
    Looks for patterns like '3+ years', 'at least 5 years', 'minimum 2 years experience'.
    """
    patterns = [
        r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)',
        r'(?:at\s+least|minimum|min\.?)\s*(\d+)\s*(?:years?|yrs?)',
        r'(?:experience|exp)\s*(?:of\s+)?(\d+)\+?\s*(?:years?|yrs?)',
        r'(\d+)\s*(?:\-|to)\s*\d+\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)',
    ]

    max_years = None
    for p in patterns:
        matches = re.findall(p, text, re.IGNORECASE)
        for m in matches:
            years = int(m)
            if max_years is None or years > max_years:
                max_years = years

    if max_years is not None:
        return f"At least {max_years} years"
    return ""


# ═══════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def parse_job_description(raw_text):
    """
    Parse a raw job description text and return structured data.

    Returns:
        dict with: job_title, skills, education, experience
    """
    job_title = extract_job_title_from_jd(raw_text)
    skills = extract_skills_from_jd(raw_text)
    education = extract_education_from_jd(raw_text)
    experience = extract_experience_from_jd(raw_text)

    return {
        "job_title": job_title,
        "skills": skills,
        "education": education,
        "experience": experience,
    }
