import json
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any


def initialize_database(database_path: Path) -> None:
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_date TEXT NOT NULL,
                created_at TEXT NOT NULL,
                source_name TEXT NOT NULL,
                target_label TEXT NOT NULL,
                transcript TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                analysis_json TEXT NOT NULL
            )
            """
        )


def save_report(
    database_path: Path,
    session_date: date,
    source_name: str,
    target_label: str,
    transcript: str,
    metrics: dict[str, Any],
    analysis: dict[str, Any],
) -> int:
    with sqlite3.connect(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO reports (
                session_date, created_at, source_name, target_label,
                transcript, metrics_json, analysis_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_date.isoformat(),
                datetime.now().astimezone().strftime("%H:%M"),
                source_name,
                target_label,
                transcript,
                json.dumps(metrics, ensure_ascii=True),
                json.dumps(analysis, ensure_ascii=True),
            ),
        )
        return int(cursor.lastrowid)


def list_reports(database_path: Path, limit: int = 200) -> list[dict[str, Any]]:
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM reports ORDER BY session_date DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {
            "id": row["id"],
            "session_date": row["session_date"],
            "created_at": row["created_at"],
            "source_name": row["source_name"],
            "target_label": row["target_label"],
            "transcript": row["transcript"],
            "metrics": json.loads(row["metrics_json"]),
            "analysis": json.loads(row["analysis_json"]),
        }
        for row in rows
    ]