"""resume_processor.py
Utility functions to extract text, parse candidate resumes, and rank candidates
using the OpenAI API (Chat Completions).
"""

import os
import json
import time
import random
import logging
from typing import List, Dict, Any

import openai
from pdfminer.high_level import extract_text as extract_pdf_text

# Read your API key from the environment or configure it some other secure way
openai.api_key = os.getenv("OPENAI_API_KEY")
# Removed printing the API key to avoid leaking secrets

# Feel free to change the model to gpt-4o or another
MODEL_NAME = "gpt-4o"  # Use a valid OpenAI model name


# -----------------------------------------------------------
# Text extraction
# -----------------------------------------------------------

def extract_text(file_path: str) -> str:
    """Return raw text from a PDF or plaintext resume."""
    if file_path.lower().endswith(".pdf"):
        try:
            return extract_pdf_text(file_path)
        except Exception as exc:
            logging.error(f"[extract_text] PDF extraction failed: {exc}")
            return ""
    elif file_path.lower().endswith(".txt"):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
                return fh.read()
        except Exception as exc:
            logging.error(f"[extract_text] Reading .txt failed: {exc}")
            return ""
    else:
        logging.error("[extract_text] Unsupported file type.")
        return ""


# -----------------------------------------------------------
# Resume parsing
# -----------------------------------------------------------

def _call_openai(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> str:
    """Wrapper around the Chat Completion endpoint with sensible defaults.

    Adds retries with exponential backoff + jitter and logs failures.
    Returns an empty string after exhausting retries so callers can handle gracefully.
    """
    if not openai.api_key:
        # Treat missing API key as non-retryable
        raise RuntimeError("OPENAI_API_KEY is not set. Please configure it in your environment.")

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = openai.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            last_err = exc
            err_name = exc.__class__.__name__
            logging.warning(
                f"[_call_openai] Attempt {attempt}/{max_retries} failed with {err_name}: {exc}"
            )
            # Decide whether to retry: for known client errors, don't bother
            non_retryable = {"BadRequestError", "AuthenticationError"}
            if err_name in non_retryable:
                logging.error(f"[_call_openai] Non-retryable error: {err_name}. Aborting retries.")
                break
            if attempt < max_retries:
                sleep_s = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                time.sleep(sleep_s)
    logging.error(f"[_call_openai] OpenAI call failed after {max_retries} attempts: {last_err}")
    return ""


def parse_resume_with_openai(resume_text: str) -> Dict[str, Any]:
    """Parse a resume into structured JSON using the OpenAI ChatCompletion API.

    The function attempts to coerce the model output into a JSON object.
    """
    system_prompt = (
        "You are an Applicant Tracking System (ATS) resume parser. "
        "Extract key structured data from the raw resume text and reply with **ONLY** a JSON object. "
        "Do not expose any personal information (PII) such as name, email, phone, or address. "
        "Anonymize all candidate data."
    )

    user_prompt = (
        "Resume Text:\n" + resume_text +
        "\n\nReturn a JSON object with these exact keys: \n"
        "  years_of_experience  – integer\n"
        "  skills               – array of strings\n"
        "  projects             – array of objects with keys name & description\n"
        "  anonymized_id        – string (create a unique id, do not expose PII)\n"
    )

    raw = _call_openai(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.15,
    )

    # Make a best‑effort attempt to coerce into JSON
    import logging
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            return json.loads(raw[start:end])
        except Exception:
            logging.error("[parse_resume_with_openai] Failed to decode JSON.")
            return {}


# -----------------------------------------------------------
# Candidate ranking
# -----------------------------------------------------------

def rank_candidates_with_openai(
    job_description: str,
    custom_prompt: str,
    candidate_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Rank candidates against a job description and optional custom prompt.

    Returns a list of objects: {resume_id, fit_score, reason}
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior technical recruiter. "
                "For each candidate, provide a fit_score (float, 0-100) and a specific, meaningful reason for the score. "
                "Return ONLY a JSON array of objects: {resume_id, fit_score, reason}. Reason must be unique and relevant for each candidate."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Job Description:\n{job_description}\n\n" +
                (f"Additional instructions:\n{custom_prompt}\n\n" if custom_prompt else "") +
                "Candidate data (JSON):\n" + json.dumps(candidate_data)
            ),
        },
    ]

    raw = _call_openai(messages, temperature=0.3)

    import logging
    try:
        rankings = json.loads(raw)
        # Ensure every ranking has a reason
        for r in rankings:
            if not r.get('reason'):
                r['reason'] = 'No reason provided.'
        return rankings
    except json.JSONDecodeError:
        try:
            start = raw.index("[")
            end = raw.rindex("]") + 1
            rankings = json.loads(raw[start:end])
            for r in rankings:
                if not r.get('reason'):
                    r['reason'] = 'No reason provided.'
            return rankings
        except Exception:
            logging.error("[rank_candidates_with_openai] Failed to decode JSON.")
            return []
