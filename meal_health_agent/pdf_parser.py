"""
Meal & Health Agent — pdf_parser.py
Extracts nutritional marker values from a bloodwork PDF.

Strategy:
    1. Extract raw text with pypdf
    2. Send text to OpenAI with a structured extraction prompt
    3. Return a clean dict of marker → value pairs

Markers we look for (extendable):
    Iron, Ferritin, B12, Vitamin D, Folate, Hemoglobin,
    Calcium, Magnesium, Zinc, Cholesterol (LDL/HDL), Glucose, HbA1c
"""

import os
import json
from openai import OpenAI

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader  # fallback for older installs


# Markers the agent cares about nutritionally
NUTRITIONAL_MARKERS = [
    "Iron", "Ferritin", "Vitamin B12", "Vitamin D", "Folate",
    "Hemoglobin", "Calcium", "Magnesium", "Zinc",
    "LDL Cholesterol", "HDL Cholesterol", "Total Cholesterol",
    "Glucose", "HbA1c", "Sodium", "Potassium",
]

EXTRACTION_PROMPT = """
You are a medical data extractor. Below is raw text from a bloodwork report.

Extract ONLY the following markers (if present) and return a JSON object
with marker names as keys and their values (including units) as strings.
If a marker is not found, omit it from the output.

Markers to extract:
{markers}

Return ONLY valid JSON — no explanation, no markdown, no code fences.

Report text:
{text}
"""


def parse_bloodwork_pdf(path: str) -> dict:
    """
    Parse a bloodwork PDF and return a dict of nutritional markers.
    Raises FileNotFoundError if the path doesn't exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF not found: {path}")

    raw_text = _extract_pdf_text(path)
    if not raw_text.strip():
        return {"error": "PDF appears to be empty or image-only (no extractable text)."}

    markers = _extract_markers_with_llm(raw_text)
    return markers


def _extract_pdf_text(path: str) -> str:
    """Extract all text from a PDF file using pypdf."""
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def _extract_markers_with_llm(raw_text: str) -> dict:
    """Use OpenAI to pull structured marker data out of messy PDF text."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Truncate if very long (most bloodwork reports are < 3000 tokens)
    truncated = raw_text[:8000]

    prompt = EXTRACTION_PROMPT.format(
        markers="\n".join(f"- {m}" for m in NUTRITIONAL_MARKERS),
        text=truncated,
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500,
    )

    content = response.choices[0].message.content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Best-effort: return the raw string if JSON parse fails
        return {"raw_extraction": content}


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_parser.py <path_to_bloodwork.pdf>")
    else:
        result = parse_bloodwork_pdf(sys.argv[1])
        print(json.dumps(result, indent=2))
