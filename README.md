# HireMind
A generative AI-based hiring tool that intelligently analyzes resumes, ranks candidates based on job fit, and provides a professional dashboard, designed to attract recruiters with its efficiency and precision.


## Debugging missing Experience/Skills/Projects

Quickly diagnose where the pipeline drops fields using the smoke script:

```
python scripts/smoke_debug.py resumes/<one-file-from-your-upload>
```

It prints:
- raw text length (extraction)
- redacted text length (sanitization)
- parser keys + years_of_experience/tech/impacts and sample JSON

If raw_len is ~0 => extraction issue; if redacted_len is tiny => over-redaction; if parser keys empty => API/schema issue.

## Test suite (offline)

Run with a virtualenv:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

What tests cover:
- extraction smoke for .txt
- redaction stability (doesn’t nuke content)
- parser contract with mocked OpenAI (returns schema + _redacted_text)
- UI flattening logic (years_of_experience/skills/projects rendering)
