"""
nlp_processor.py - NLP Text Preprocessing & Entity Recognition

Responsibilities:
    - Load the spaCy language model (en_core_web_sm).
    - Clean and normalize raw text extracted from PDFs.
    - Perform Named Entity Recognition (NER) to identify:
        • PERSON  → candidate name
        • GPE     → location / city / country
        • ORG     → organisations, universities, companies
        • DATE    → dates for experience / education
    - Tokenize text and compute word_count.
    - Split text into lines for section-header detection.
"""

import re
import spacy

# ---------------------------------------------------------------------------
# Load spaCy model once at module level for performance
# ---------------------------------------------------------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If the model is not installed, download it automatically
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def clean_text(raw_text: str) -> str:
    """
    Clean and normalize raw PDF text.

    Steps:
        1. Replace multiple whitespace / newlines with single space.
        2. Remove non-printable characters.
        3. Strip leading / trailing whitespace.
    """
    # Remove non-printable characters (keep newlines for section splitting)
    text = re.sub(r'[^\x20-\x7E\n\r\t\u0600-\u06FF]', ' ', raw_text)
    # Collapse multiple spaces (but keep newlines)
    text = re.sub(r'[ \t]+', ' ', text)
    # Collapse multiple blank lines into one
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def get_spacy_doc(text: str):
    """
    Process cleaned text through spaCy and return the Doc object.
    """
    # spaCy has a max length limit; truncate very long texts
    max_len = 100_000
    return nlp(text[:max_len])


def extract_entities(doc) -> dict:
    """
    Extract named entities grouped by label.

    Returns:
        {
            "PERSON":  ["John Doe"],
            "GPE":     ["Cairo", "Egypt"],
            "ORG":     ["MIT", "Google"],
            "DATE":    ["2020", "Jan 2019"],
            ...
        }
    """
    entities = {}
    for ent in doc.ents:
        label = ent.label_
        if label not in entities:
            entities[label] = []
        value = ent.text.strip()
        if value and value not in entities[label]:
            entities[label].append(value)
    return entities


def get_word_count(text: str) -> int:
    """
    Count meaningful words in the text (ignoring very short tokens).
    """
    words = re.findall(r'\b[a-zA-Z]{2,}\b', text)
    return len(words)


def get_lines(text: str) -> list:
    """
    Split text into individual lines (useful for section-header detection).
    """
    return [line.strip() for line in text.split('\n') if line.strip()]


def process_text(raw_text: str) -> dict:
    """
    Main entry point — clean text, run NLP, and return all results.

    Returns:
        {
            "cleaned_text": str,
            "doc": spacy.tokens.Doc,
            "entities": { label: [values] },
            "word_count": int,
            "lines": [str, ...]
        }
    """
    cleaned = clean_text(raw_text)
    doc = get_spacy_doc(cleaned)
    entities = extract_entities(doc)
    word_count = get_word_count(cleaned)
    lines = get_lines(cleaned)

    return {
        "cleaned_text": cleaned,
        "doc": doc,
        "entities": entities,
        "word_count": word_count,
        "lines": lines,
    }
