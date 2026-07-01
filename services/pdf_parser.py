"""
pdf_parser.py - PDF Text & Metadata Extraction Service

Responsibilities:
    - Extract raw text from uploaded PDF resumes using pdfplumber.
    - Detect formatting issues that ATS systems penalize:
        • Images (has_image)
        • Tables (has_table)
        • Multi-column layouts (multi_column_layout)
    - Calculate page_count and resume_length_score.
"""

import pdfplumber


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract all text from a PDF file, page by page.

    Args:
        file_path: Absolute path to the PDF file.

    Returns:
        A single string containing all extracted text.
    """
    full_text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
    except Exception as e:
        print(f"[pdf_parser] Error extracting text: {e}")
    return full_text.strip()


def get_pdf_metadata(file_path: str) -> dict:
    """
    Analyse the PDF for formatting issues and metadata.

    Returns a dict with:
        - page_count (int): Total number of pages.
        - has_image (bool): True if any page contains images.
        - has_table (bool): True if any page contains tables.
        - multi_column_layout (bool): Heuristic check for multi-column layouts.
        - resume_length_score (str): 'good' | 'too_short' | 'too_long'
    """
    metadata = {
        "page_count": 0,
        "has_image": False,
        "has_table": False,
        "multi_column_layout": False,
        "resume_length_score": "good",
    }

    try:
        with pdfplumber.open(file_path) as pdf:
            metadata["page_count"] = len(pdf.pages)

            for page in pdf.pages:
                # --- Image detection ---
                if page.images and len(page.images) > 0:
                    metadata["has_image"] = True

                # --- Table detection ---
                # pdfplumber's find_tables() is too aggressive and often misidentifies standard CV formatting
                # (like right-aligned dates or spaced columns) as tables, causing false positive ATS penalties.
                # tables = page.find_tables()
                # if tables and len(tables) > 0:
                #     metadata["has_table"] = True

                # --- Multi-column heuristic ---
                words = page.extract_words()
                if words:
                    page_width = page.width
                    mid = page_width / 2
                    
                    left_words = 0
                    right_words = 0
                    crossing_words = 0
                    
                    for w in words:
                        x0, x1 = float(w["x0"]), float(w["x1"])
                        if x1 < mid - 20:
                            left_words += 1
                        elif x0 > mid + 20:
                            right_words += 1
                        else:
                            # Word crosses or is inside the middle gutter
                            crossing_words += 1
                            
                    # A true multi-column layout has substantial text on both sides
                    # and an empty gutter in the middle (very few words crossing it).
                    # A single-column layout will have many words crossing the middle.
                    if left_words > 30 and right_words > 30 and crossing_words < 10:
                        metadata["multi_column_layout"] = True

            # --- Resume length scoring ---
            page_count = metadata["page_count"]
            if page_count == 0:
                metadata["resume_length_score"] = "too_short"
            elif page_count <= 2:
                metadata["resume_length_score"] = "good"
            elif page_count <= 3:
                metadata["resume_length_score"] = "acceptable"
            else:
                metadata["resume_length_score"] = "too_long"

    except Exception as e:
        print(f"[pdf_parser] Error reading PDF metadata: {e}")

    return metadata


def parse_pdf(file_path: str) -> dict:
    """
    Main entry point — extracts text AND metadata from a PDF.

    Returns:
        {
            "text": str,
            "metadata": { page_count, has_image, has_table, ... }
        }
    """
    text = extract_text_from_pdf(file_path)
    metadata = get_pdf_metadata(file_path)
    return {
        "text": text,
        "metadata": metadata,
    }
