Test suite overview

What these tests cover
- Text extraction smoke: confirms .txt path works (fast, no external deps)
- Redaction stability: ensures we don’t nuke the content density
- Parser contract: mocks OpenAI to validate our JSON schema & redacted body exposure
- UI flattening: validates format_candidate_data produces non-empty skills/projects/yoe

Run locally
1) (Recommended) create a venv and install requirements
2) Run tests

Example (zsh)
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

Notes
- The parser test monkeypatches the OpenAI client so it runs offline.
- Add PDF/DOCX fixtures later if needed; for extraction, .txt is sufficient to catch the pipeline regression you saw.
