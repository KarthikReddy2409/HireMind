import sqlite3
from pathlib import Path

# Simple SQLite migration runner for local/dev use.
# Applies the column additions from migrations/001_add_explainability.py idempotently.


def add_column(cur, table: str, col: str, type_: str):
    cur.execute(f"PRAGMA table_info({table});")
    cols = [r[1] for r in cur.fetchall()]
    if col not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {type_};")


def main():
    db = Path("hiring_tool.db")
    if not db.exists():
        print("Database not found, will be created on first run by the app: hiring_tool.db")
    con = sqlite3.connect(str(db))
    cur = con.cursor()
    # CandidateProfile new columns (keep in sync with models.CandidateProfile)
    add_column(cur, "candidate_profiles", "redacted_text", "TEXT")
    add_column(cur, "candidate_profiles", "redacted_text_hash", "TEXT")
    add_column(cur, "candidate_profiles", "evidence_spans", "TEXT")
    add_column(cur, "candidate_profiles", "evidence_snippets", "TEXT")
    add_column(cur, "candidate_profiles", "subscores", "TEXT")
    add_column(cur, "candidate_profiles", "penalties", "TEXT")
    add_column(cur, "candidate_profiles", "uncertainty", "REAL")
    add_column(cur, "candidate_profiles", "reason", "TEXT")
    add_column(cur, "candidate_profiles", "scoring_version", "TEXT")
    con.commit()
    con.close()
    print("Migration applied to:", db)


if __name__ == "__main__":
    main()
