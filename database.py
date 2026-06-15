"""
database.py
-----------
SQLite pipeline for the Vision Defect Inspection API.
Logs every prediction request with full metadata.
Provides query and aggregation methods for the /logs and /stats endpoints.
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional

DB_PATH = os.environ.get("DB_PATH", "db/predictions.db")


def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection with row factory enabled."""
    parent = os.path.dirname(DB_PATH)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the predictions table if it does not already exist."""
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT    NOT NULL,
            filename     TEXT,
            label        TEXT    NOT NULL,
            confidence   REAL    NOT NULL,
            defect_prob  REAL    NOT NULL,
            pass_prob    REAL    NOT NULL,
            processing_ms INTEGER
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_label ON predictions(label)
    """)
    conn.commit()
    conn.close()


def log_prediction(
    label: str,
    confidence: float,
    defect_prob: float,
    pass_prob: float,
    filename: Optional[str] = None,
    processing_ms: Optional[int] = None,
) -> int:
    """
    Insert a prediction record into the database.
    Returns the new row ID.
    """
    conn = get_connection()
    cursor = conn.execute(
        """
        INSERT INTO predictions
            (timestamp, filename, label, confidence, defect_prob, pass_prob, processing_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.utcnow().isoformat(),
            filename,
            label,
            confidence,
            defect_prob,
            pass_prob,
            processing_ms,
        ),
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def get_recent_logs(limit: int = 50, label_filter: Optional[str] = None) -> list[dict]:
    """
    Retrieve the most recent prediction logs.

    Args:
        limit:        Max number of rows to return (default 50, max 200).
        label_filter: Optional — 'pass' or 'fail' to filter by result.

    Returns:
        List of dicts with all prediction fields.
    """
    limit = min(limit, 200)
    conn = get_connection()

    if label_filter:
        rows = conn.execute(
            """
            SELECT * FROM predictions
            WHERE label = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (label_filter.lower(), limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT * FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    conn.close()
    return [dict(row) for row in rows]


def get_stats() -> dict:
    """
    Compute aggregate statistics across all logged predictions.

    Returns:
        {
            total, pass_count, fail_count, pass_rate, fail_rate,
            avg_confidence, avg_defect_prob, avg_processing_ms,
            last_prediction_at
        }
    """
    conn = get_connection()

    row = conn.execute("""
        SELECT
            COUNT(*)                          AS total,
            SUM(CASE WHEN label='pass' THEN 1 ELSE 0 END) AS pass_count,
            SUM(CASE WHEN label='fail' THEN 1 ELSE 0 END) AS fail_count,
            ROUND(AVG(confidence), 4)         AS avg_confidence,
            ROUND(AVG(defect_prob), 4)        AS avg_defect_prob,
            ROUND(AVG(processing_ms), 1)      AS avg_processing_ms,
            MAX(timestamp)                    AS last_prediction_at
        FROM predictions
    """).fetchone()

    conn.close()
    total = row["total"] or 0
    return {
        "total":              total,
        "pass_count":         row["pass_count"] or 0,
        "fail_count":         row["fail_count"] or 0,
        "pass_rate":          round((row["pass_count"] or 0) / total, 4) if total else 0,
        "fail_rate":          round((row["fail_count"] or 0) / total, 4) if total else 0,
        "avg_confidence":     row["avg_confidence"],
        "avg_defect_prob":    row["avg_defect_prob"],
        "avg_processing_ms":  row["avg_processing_ms"],
        "last_prediction_at": row["last_prediction_at"],
    }
