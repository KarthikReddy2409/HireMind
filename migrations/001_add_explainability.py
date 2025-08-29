import sqlite3


def add_column(cur, table, col, type_):
    cur.execute(f"PRAGMA table_info({table});")
    cols = [r[1] for r in cur.fetchall()]
    if col not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {type_};")


if __name__ == "__main__":
    conn = sqlite3.connect("hiring_tool.db")
    cur = conn.cursor()
    # CandidateProfile new columns
    add_column(cur, "candidate_profiles", "redacted_text", "TEXT")
    add_column(cur, "candidate_profiles", "redacted_text_hash", "TEXT")
    add_column(cur, "candidate_profiles", "evidence_spans", "TEXT")
    add_column(cur, "candidate_profiles", "evidence_snippets", "TEXT")
    add_column(cur, "candidate_profiles", "subscores", "TEXT")
    add_column(cur, "candidate_profiles", "penalties", "TEXT")
    add_column(cur, "candidate_profiles", "uncertainty", "REAL")
    add_column(cur, "candidate_profiles", "reason", "TEXT")
    add_column(cur, "candidate_profiles", "scoring_version", "TEXT")
    conn.commit()
    conn.close()
    print("Migration complete.")
