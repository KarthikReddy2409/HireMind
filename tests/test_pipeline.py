import os
import json
import types
import pytest

# Ensure project can import local modules
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import resume_processor as rp
from resume_processor import extract_text, pre_redact_text, parse_resume_with_openai
from app import format_candidate_data

SAMPLE_TXT = (
    "Experience\n"
    "Senior Data Scientist at [ORG] (2019-01 to 2023-06)\n"
    "- Built ML pipelines in Python and Spark.\n"
    "- Improved conversion by 25% QoQ.\n"
    "Skills: Python, SQL, TensorFlow, Gurobi, AWS\n"
    "Education: [SCHOOL]\n"
)

@pytest.fixture(autouse=True)
def _env():
    os.environ.setdefault("BLIND_FIRST_PASS", "true")
    yield


def test_extract_text_handles_txt(tmp_path):
    p = tmp_path / "resume.txt"
    p.write_text(SAMPLE_TXT, encoding="utf-8")
    out = extract_text(str(p))
    assert isinstance(out, str)
    assert len(out) >= len("Skills")


def test_pre_redact_text_preserves_density():
    red = pre_redact_text(SAMPLE_TXT, blind=True)
    assert len(red) > len(SAMPLE_TXT) * 0.5  # not nuked
    assert "[REDACTED]" in red or "[ORG]" in red or "[SCHOOL]" in red


def _mock_openai_resume(monkeypatch):
    class FakeMsg:
        def __init__(self, content):
            self.content = content
    class FakeChoice:
        def __init__(self, content):
            self.message = FakeMsg(content)
    class FakeResp:
        def __init__(self, content):
            self.choices = [FakeChoice(content)]
    class FakeChat:
        def __init__(self):
            self.completions = types.SimpleNamespace(create=self._create)
        def _create(self, **kwargs):
            # Return deterministic structured output
            data = {
                "years_of_experience": 4.5,
                "tech": {
                    "languages": ["Python", "SQL"],
                    "ml": ["TensorFlow"],
                    "or": ["Gurobi"],
                    "cloud": ["AWS"],
                },
                "impacts": [{"metric": "conversion", "delta": 0.25, "period": "quarterly", "evidence": "Improved conversion by 25% QoQ"}],
                "confidence": 0.9,
                "evidence_spans": [{"type": "impact", "start": 10, "end": 30}],
            }
            content = json.dumps(data)
            return FakeResp(content)
    fake_openai = types.SimpleNamespace(chat=FakeChat())
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    # Patch the module-level openai object inside resume_processor
    monkeypatch.setattr(rp, 'openai', fake_openai, raising=True)


def test_parse_resume_with_openai_structured(monkeypatch):
    _mock_openai_resume(monkeypatch)
    out = parse_resume_with_openai(SAMPLE_TXT, blind=True)
    assert out.get("years_of_experience") == 4.5
    tech = out.get("tech") or {}
    assert "Python" in tech.get("languages", [])
    assert out.get("_redacted_text")  # persisted body for spans


def test_format_candidate_data_flattens_skills(monkeypatch):
    # Fake candidate ORM record
    class C:
        def __init__(self):
            self.fit_score = 87.12
            self.secondary_index = "Candidate_1234"
            self.resume_id = "rid-1"
            self.reason = ""
    c = C()
    # Fake resume object
    class R:
        def __init__(self):
            self.file_path = "/tmp/rid-1_resume.pdf"
    r = R()
    data = {
        "years_of_experience": 3.0,
        "tech": {
            "languages": ["Python", "SQL"],
            "ml": ["TensorFlow"],
            "or": ["Gurobi"],
            "cloud": ["AWS"],
        },
        "projects": [{"name": "ML Pipeline"}]
    }
    res = format_candidate_data(1, c, r, data)
    assert res["years_of_experience"] == 3.0
    assert "Python" in res["skills"]
    assert "ML Pipeline" in res["projects"]
