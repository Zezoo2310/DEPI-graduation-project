"""
feature_extractor.py - Comprehensive Resume Feature Extraction

Extracts features across 8 ATS categories:
    1. Contact Information (name, email, phone, linkedin, github, location)
    2. Resume Structure   (section headers detection)
    3. Content Quality    (word_count, skills_count, projects_count, etc.)
    4. Technical Skills   (loaded from Datasets/skills_database.json)
    5. Soft Skills        (loaded from Datasets/skills_database.json)
    6. Action Verbs       (developed, implemented, etc.)
    7. Education Analysis (degree type and major)
    8. Formatting         (comes from pdf_parser metadata)

Skills are loaded dynamically from the Datasets folder instead of hardcoded lists.
"""

import re
import os
import json
import csv

# ═══════════════════════════════════════════════════════════════════════════
#  LOAD SKILLS FROM DATASETS
# ═══════════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(BASE_DIR, "Datasets")


def _load_skills_database():
    """
    Load skills from Datasets/skills_database.json.
    Returns a dict of { category: [skills] } and flat lists for
    technical / soft skills.
    """
    path = os.path.join(DATASETS_DIR, "skills_database.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            db = json.load(f)
    except FileNotFoundError:
        print(f"[feature_extractor] WARNING: {path} not found. Using fallback.")
        db = {}
    return db


def _load_skills_csv():
    """
    Load additional skills from Datasets/skills_list.csv.
    Returns a list of dicts: [{"Skill Name": ..., "Category": ...}, ...]
    """
    path = os.path.join(DATASETS_DIR, "skills_list.csv")
    skills = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Skill Name", "").strip():
                    skills.append({
                        "name": row["Skill Name"].strip(),
                        "category": row.get("Category", "").strip(),
                    })
    except FileNotFoundError:
        print(f"[feature_extractor] WARNING: {path} not found.")
    return skills


def _load_job_roles():
    """
    Load job roles from Datasets/job_roles.csv.
    Returns a list of dicts with job info.
    """
    path = os.path.join(DATASETS_DIR, "job_roles.csv")
    roles = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                roles.append(row)
    except FileNotFoundError:
        print(f"[feature_extractor] WARNING: {path} not found.")
    return roles


# ── Load data at module level (once) ──
_SKILLS_DB = _load_skills_database()
_SKILLS_CSV = _load_skills_csv()
_JOB_ROLES = _load_job_roles()

# ── Build merged skill lists from Datasets ──
_SOFT_SKILL_CATEGORIES = {"Soft Skills"}
_TECHNICAL_CATEGORIES = {
    "Programming", "Web Development", "Mobile Development",
    "Cloud & DevOps", "Data Science & Analytics", "Database",
    "Design", "Cybersecurity", "Professional", "Business",
}

# Technical skills: merge from JSON + CSV
TECHNICAL_SKILLS = []
for cat, skills in _SKILLS_DB.items():
    if cat not in _SOFT_SKILL_CATEGORIES:
        for s in skills:
            if s not in TECHNICAL_SKILLS:
                TECHNICAL_SKILLS.append(s)
for entry in _SKILLS_CSV:
    if entry["category"] not in _SOFT_SKILL_CATEGORIES:
        if entry["name"] not in TECHNICAL_SKILLS:
            TECHNICAL_SKILLS.append(entry["name"])

# Also add extra skills that might appear in job_roles but not in the DB
for role in _JOB_ROLES:
    required = role.get("Required Skills", "")
    for skill in required.split("|"):
        skill = skill.strip()
        if skill and skill not in TECHNICAL_SKILLS:
            TECHNICAL_SKILLS.append(skill)

# Soft skills: from JSON + CSV
SOFT_SKILLS = []
for cat, skills in _SKILLS_DB.items():
    if cat in _SOFT_SKILL_CATEGORIES:
        for s in skills:
            if s not in SOFT_SKILLS:
                SOFT_SKILLS.append(s)
for entry in _SKILLS_CSV:
    if entry["category"] in _SOFT_SKILL_CATEGORIES:
        if entry["name"] not in SOFT_SKILLS:
            SOFT_SKILLS.append(entry["name"])

# Print stats on load
print(f"[feature_extractor] Loaded {len(TECHNICAL_SKILLS)} technical skills from Datasets")
print(f"[feature_extractor] Loaded {len(SOFT_SKILLS)} soft skills from Datasets")
print(f"[feature_extractor] Loaded {len(_JOB_ROLES)} job roles from Datasets")


# ═══════════════════════════════════════════════════════════════════════════
#  STATIC DATABASES — Action Verbs, Section Headers, Degrees, Majors
# ═══════════════════════════════════════════════════════════════════════════

# (Action Verbs removed based on user request)

# Section header keywords (matched case-insensitively)
SECTION_HEADERS = {
    "summary":        ["summary", "objective", "profile", "about me", "professional summary", "career objective"],
    "education":      ["education", "academic", "academics", "qualification", "qualifications"],
    "experience":     ["experience", "work experience", "professional experience", "employment", "work history", "career history"],
    "skills":         ["skills", "technical skills", "core competencies", "competencies", "technologies", "tech stack"],
    "projects":       ["projects", "personal projects", "academic projects", "key projects", "portfolio"],
    "certifications": ["certifications", "certificates", "licenses", "professional certifications", "credentials"],
    "languages":      ["languages", "language proficiency", "language skills"],
    "achievements":   ["achievements", "awards", "honors", "accomplishments", "recognition"],
}

DEGREE_KEYWORDS = {
    "phd":       ["ph.d", "phd", "doctorate", "doctor of philosophy"],
    "master":    ["master", "msc", "m.sc", "m.s.", "ma", "m.a.", "mba", "m.b.a", "meng", "m.eng"],
    "bachelor":  ["bachelor", "bsc", "b.sc", "b.s.", "ba", "b.a.", "beng", "b.eng"],
    "diploma":   ["diploma", "associate", "associate degree"],
}

MAJOR_KEYWORDS = [
    "Computer Science", "Information Technology", "Information Systems",
    "Software Engineering", "Computer Engineering", "Data Science",
    "Artificial Intelligence", "Cybersecurity", "Electrical Engineering",
    "Mechanical Engineering", "Civil Engineering", "Business Administration",
    "Marketing", "Finance", "Accounting", "Economics", "Mathematics",
    "Statistics", "Physics", "Chemistry", "Biology", "Medicine",
    "Pharmacy", "Nursing", "Law", "Architecture", "Graphic Design",
    "Communications", "Media", "Journalism", "Psychology", "Sociology",
    "Political Science", "Engineering",
]


# ═══════════════════════════════════════════════════════════════════════════
#  EXTRACTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

# ---------- 1. Contact Information ----------

def extract_email(text: str) -> list:
    """Extract email addresses using regex."""
    pattern = r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'
    return list(set(re.findall(pattern, text)))


def extract_phone(text: str) -> list:
    """Extract phone numbers using multiple regex patterns."""
    patterns = [
        r'\+?\d{1,3}[\s\-]?\(?\d{1,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{3,4}',
        r'\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}',
        r'\+?\d{10,15}',
    ]
    phones = []
    for p in patterns:
        found = re.findall(p, text)
        for phone in found:
            cleaned = re.sub(r'[\s\-\(\)]', '', phone)
            if 7 <= len(cleaned.replace('+', '')) <= 15:
                if phone.strip() not in phones:
                    phones.append(phone.strip())
    return phones


def extract_linkedin(text: str) -> str | None:
    """Extract LinkedIn profile URL."""
    pattern = r'(?:https?://)?(?:www\.)?linkedin\.com/in/[\w\-]+'
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(0) if match else None


def extract_github(text: str) -> str | None:
    """Extract GitHub profile URL."""
    pattern = r'(?:https?://)?(?:www\.)?github\.com/[\w\-]+'
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(0) if match else None


def extract_name(entities: dict, lines: list) -> str | None:
    """
    Extract candidate name reliably.
    Strategy: Check the first few non-empty lines for something that looks like a name
    (1-4 words, alphabetic, title case). If that fails, fallback to spaCy PERSON entities.
    """
    # Look at the top 3 non-empty lines
    for line in lines[:5]:
        line = line.strip()
        if not line: continue
        # A name is typically 1 to 4 words, no numbers, no special chars (except hyphens/apostrophes)
        if 1 <= len(line.split()) <= 4 and re.match(r"^[A-Za-z\s\-']+$", line):
            # Exclude obvious non-names
            ignore_words = ["resume", "curriculum", "vitae", "cv", "developer", "engineer", "designer", "manager", "profile", "contact", "email", "phone"]
            if not any(w in line.lower() for w in ignore_words):
                return line.title()
    
    # Fallback to spaCy
    persons = entities.get("PERSON", [])
    if persons:
        return persons[0].title()
        
    return None

def extract_job_title(lines: list) -> str | None:
    """
    Extract expected job title from the top 10 lines.
    Checks against the loaded job roles dataset or common title keywords.
    """
    job_roles_lower = [role.get("Job Role", "").lower() for role in _JOB_ROLES if role.get("Job Role")]
    
    for line in lines[:10]:
        line_lower = line.strip().lower()
        if not line_lower or len(line_lower.split()) > 6:
            continue
            
        # 1. Match from dataset
        for role in job_roles_lower:
            if role and (role == line_lower or role in line_lower):
                return line.strip().title()
                
        # 2. Heuristic for common title patterns
        title_keywords = ["developer", "engineer", "manager", "specialist", "designer", "consultant", "analyst", "architect", "administrator", "lead"]
        if any(kw in line_lower for kw in title_keywords):
             if not re.search(r'[@:/]', line_lower): # Ensure it's not a URL or email
                 return line.strip().title()
                 
    return None


def extract_location(entities: dict) -> str | None:
    """Extract location from spaCy GPE entities, filtering false positives."""
    locations = entities.get("GPE", [])
    false_positives = ["AI", "IT", "ML", "CS", "Computer Science", "Information Technology", "Data Science", "B.Sc", "M.Sc"]
    
    # Also filter out any word that exists in our technical skills database
    tech_skills_lower = {skill.lower() for skill in TECHNICAL_SKILLS}
    
    for loc in locations:
        loc_clean = loc.strip()
        loc_lower = loc_clean.lower()
        
        # Ignore very short words or known tech false positives
        if len(loc_clean) > 2 and loc_clean not in false_positives and loc_lower not in tech_skills_lower:
            return loc_clean
            
    return None


def get_contact_info(text: str, entities: dict, lines: list) -> dict:
    """Aggregate all contact information."""
    emails = extract_email(text)
    phones = extract_phone(text)
    linkedin = extract_linkedin(text)
    github = extract_github(text)
    name = extract_name(entities, lines)
    location = extract_location(entities)
    job_title = extract_job_title(lines)

    return {
        "name":         name,
        "job_title":    job_title,
        "emails":       emails,
        "phones":       phones,
        "linkedin":     linkedin,
        "github":       github,
        "location":     location,
        "has_name":     name is not None,
        "has_job_title": job_title is not None,
        "has_email":    len(emails) > 0,
        "has_phone":    len(phones) > 0,
        "has_linkedin": linkedin is not None,
        "has_github":   github is not None,
        "has_location": location is not None,
    }


# ---------- 2. Resume Structure (Section Detection) ----------

def detect_sections(lines: list) -> dict:
    """
    Detect which standard resume sections are present
    by matching line text against known header keywords.
    """
    sections_found = {f"has_{key}": False for key in SECTION_HEADERS}

    for line in lines:
        line_lower = line.lower().strip()
        line_clean = re.sub(r'[:\-|_•●►▪▸]', '', line_lower).strip()

        for section_key, keywords in SECTION_HEADERS.items():
            if sections_found[f"has_{section_key}"]:
                continue
            for kw in keywords:
                if line_clean == kw or line_clean.startswith(kw + " "):
                    sections_found[f"has_{section_key}"] = True
                    break

    return sections_found


# ---------- 3 & 4. Technical Skills ----------

def extract_technical_skills(text: str) -> list:
    """
    Match text against the technical skills database
    (loaded from Datasets/skills_database.json + skills_list.csv + job_roles.csv).
    """
    found = []
    for skill in TECHNICAL_SKILLS:
        escaped = re.escape(skill)
        pattern = r'(?<![a-zA-Z])' + escaped + r'(?![a-zA-Z])'
        if re.search(pattern, text, re.IGNORECASE):
            if skill not in found:
                found.append(skill)
    return found


# ---------- 5. Soft Skills ----------

def extract_soft_skills(text: str) -> list:
    """Match text against soft skills database (loaded from Datasets)."""
    found = []
    for skill in SOFT_SKILLS:
        escaped = re.escape(skill)
        pattern = r'(?i)(?<![a-zA-Z])' + escaped + r'(?![a-zA-Z])'
        if re.search(pattern, text):
            if skill not in found:
                found.append(skill)
    return found


# (Action Verbs removed based on user request)


# ---------- 7. Education Analysis ----------

def extract_education(text: str) -> dict:
    """
    Detect highest degree and major/field of study.
    """
    text_lower = text.lower()

    detected_degree = None
    degree_hierarchy = ["phd", "master", "bachelor", "diploma"]
    for degree_level in degree_hierarchy:
        for keyword in DEGREE_KEYWORDS[degree_level]:
            if keyword in text_lower:
                detected_degree = degree_level.capitalize()
                break
        if detected_degree:
            break

    detected_major = None
    for major in MAJOR_KEYWORDS:
        if major.lower() in text_lower:
            detected_major = major
            break

    return {
        "has_degree":    detected_degree is not None,
        "degree_type":   detected_degree,
        "major":         detected_major,
    }


# ---------- 8. Years of Experience (heuristic) ----------

def estimate_years_of_experience(text: str) -> int | None:
    """
    Try to estimate years of experience from explicit mentions.
    """
    patterns = [
        r'(\d{1,2})\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)',
        r'(?:experience|exp)\s*(?:of)?\s*(\d{1,2})\+?\s*(?:years?|yrs?)',
        r'[Ee]xperience:\s*(\d{1,2})\s*(?:years?|yrs?)',
    ]
    max_years = None
    for p in patterns:
        matches = re.findall(p, text, re.IGNORECASE)
        for m in matches:
            years = int(m)
            if max_years is None or years > max_years:
                max_years = years
    return max_years


# ---------- 9. Skill Category Breakdown (from Datasets) ----------

def get_skills_by_category(found_skills: list) -> dict:
    """
    Categorize found skills using skills_database.json categories.
    Returns { "Programming": ["Python", "Java"], "Web Development": ["React"], ... }
    """
    categorized = {}
    for cat, db_skills in _SKILLS_DB.items():
        matched = [s for s in found_skills if s in db_skills]
        if matched:
            categorized[cat] = matched
    return categorized


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def extract_all_features(cleaned_text: str, entities: dict, lines: list,
                         word_count: int, pdf_metadata: dict) -> dict:
    """
    Extract ALL features from the resume and return a comprehensive dict.
    """
    # 1. Contact Information
    contact = get_contact_info(cleaned_text, entities, lines)

    # 2. Resume Structure
    sections = detect_sections(lines)

    # 3 & 4. Technical Skills
    technical_skills = extract_technical_skills(cleaned_text)

    # 5. Soft Skills
    soft_skills = extract_soft_skills(cleaned_text)

    # (Action Verbs removed)

    # 7. Education
    education = extract_education(cleaned_text)

    # 8. Years of experience
    years_exp = estimate_years_of_experience(cleaned_text)

    # 9. Skills by category (from Datasets)
    all_found = technical_skills + soft_skills
    skills_by_category = get_skills_by_category(all_found)

    # Compile everything
    features = {
        # Contact
        "contact": contact,

        # Sections
        "sections": sections,

        # Skills
        "technical_skills":       technical_skills,
        "technical_skills_count": len(technical_skills),
        "soft_skills":            soft_skills,
        "soft_skills_count":      len(soft_skills),

        "skills_by_category":     skills_by_category,

        # Education
        "education": education,

        # Content quality
        "word_count":              word_count,
        "years_of_experience":     years_exp,

        # Formatting (from PDF metadata)
        "formatting": pdf_metadata,
    }

    return features


def get_job_roles():
    """Return loaded job roles for use by other modules (e.g., future job matching)."""
    return _JOB_ROLES
