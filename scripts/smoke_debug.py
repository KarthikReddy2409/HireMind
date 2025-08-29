import os, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resume_processor import extract_text, pre_redact_text, parse_resume_with_openai


def main(p):
    print("file:", p)
    raw = extract_text(p)
    print("raw_len:", len(raw))
    red = pre_redact_text(raw, blind=True)
    print("redacted_len:", len(red))
    out = parse_resume_with_openai(raw, blind=True)
    keys = list(out.keys())
    print("keys:", keys)
    print("yoe:", out.get("years_of_experience"))
    print("tech:", out.get("tech"))
    print("impacts:", out.get("impacts"))
    print("_redacted_text_len:", len(out.get("_redacted_text", "")))
    print("sample_json:", json.dumps(out)[:500])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/smoke_debug.py <path-to-resume>")
        sys.exit(1)
    main(sys.argv[1])
