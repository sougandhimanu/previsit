"""Tiny SQLite layer. Demo-grade: one file, no ORM."""
import os
import shutil
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "data")).resolve()
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "previsit.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id TEXT PRIMARY KEY,
    token_hash TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL DEFAULT '',
    dob TEXT NOT NULL DEFAULT '',
    profile TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS reports (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    mime TEXT NOT NULL,
    path TEXT,
    extracted TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'draft',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS shares (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    token TEXT NOT NULL UNIQUE,
    doctor_name TEXT NOT NULL,
    sections TEXT NOT NULL,
    report_ids TEXT NOT NULL DEFAULT '[]',
    expires_at REAL NOT NULL,
    revoked INTEGER NOT NULL DEFAULT 0,
    summary TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS access_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    share_id TEXT NOT NULL REFERENCES shares(id) ON DELETE CASCADE,
    patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    at REAL NOT NULL,
    action TEXT NOT NULL,
    device TEXT NOT NULL DEFAULT ''
);
"""


def init() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with connect() as c:
        c.executescript(SCHEMA)


@contextmanager
def connect():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def one(sql: str, args: tuple = ()):
    with connect() as c:
        return c.execute(sql, args).fetchone()


def many(sql: str, args: tuple = ()):
    with connect() as c:
        return c.execute(sql, args).fetchall()


def run(sql: str, args: tuple = ()) -> None:
    with connect() as c:
        c.execute(sql, args)


def purge_old(retention_hours: float) -> int:
    """Delete demo patients (and their files) older than the retention window."""
    cutoff = time.time() - retention_hours * 3600
    rows = many("SELECT id FROM patients WHERE created_at < ?", (cutoff,))
    for r in rows:
        files = many("SELECT path FROM reports WHERE patient_id = ?", (r["id"],))
        for f in files:
            if f["path"]:
                try:
                    Path(f["path"]).unlink(missing_ok=True)
                except OSError:
                    pass
        run("DELETE FROM patients WHERE id = ?", (r["id"],))
    return len(rows)


def wipe_all() -> None:
    """Used by tests only."""
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    init()
