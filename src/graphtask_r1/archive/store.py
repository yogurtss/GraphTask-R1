from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from graphtask_r1.schema import TaskCertificate


class TaskArchive:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA busy_timeout=30000")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS tasks "
            "(signature TEXT PRIMARY KEY, task_id TEXT UNIQUE, payload TEXT)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS task_features "
            "(signature TEXT PRIMARY KEY, ngram_count INTEGER NOT NULL)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS text_ngrams "
            "(ngram TEXT NOT NULL, signature TEXT NOT NULL, PRIMARY KEY(ngram, signature))"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_text_ngrams_signature ON text_ngrams(signature)"
        )
        self.connection.commit()

    def add(self, task: TaskCertificate) -> bool:
        try:
            ngrams = {"\x1f".join(value) for value in _trigrams(task.question)}
            self.connection.execute(
                "INSERT INTO tasks(signature, task_id, payload) VALUES (?, ?, ?)",
                (task.program_signature, task.task_id, task.model_dump_json()),
            )
            self.connection.execute(
                "INSERT INTO task_features(signature, ngram_count) VALUES (?, ?)",
                (task.program_signature, len(ngrams)),
            )
            self.connection.executemany(
                "INSERT INTO text_ngrams(ngram, signature) VALUES (?, ?)",
                ((ngram, task.program_signature) for ngram in ngrams),
            )
            self.connection.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def all(self) -> list[TaskCertificate]:
        rows = self.connection.execute("SELECT payload FROM tasks ORDER BY task_id").fetchall()
        return [TaskCertificate.model_validate(json.loads(row[0])) for row in rows]

    def contains_signature(self, signature: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM tasks WHERE signature = ?", (signature,)
        ).fetchone()
        return row is not None

    def novelty(self, signature: str, question: str) -> tuple[float, float]:
        structural = 0.0 if self.contains_signature(signature) else 1.0
        tokens = {"\x1f".join(value) for value in _trigrams(question)}
        task_count = int(self.connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0])
        if task_count == 0:
            return structural, 1.0
        maximum = 0.0
        if tokens:
            placeholders = ",".join("?" for _ in tokens)
            rows = self.connection.execute(
                f"SELECT n.signature, COUNT(*) AS overlap, f.ngram_count "
                f"FROM text_ngrams n JOIN task_features f ON f.signature = n.signature "
                f"WHERE n.ngram IN ({placeholders}) GROUP BY n.signature, f.ngram_count",
                tuple(sorted(tokens)),
            ).fetchall()
            for _, overlap, other_count in rows:
                union_count = len(tokens) + int(other_count) - int(overlap)
                maximum = max(maximum, int(overlap) / union_count if union_count else 1.0)
        return structural, 1.0 - maximum

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> TaskArchive:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _trigrams(text: str) -> set[tuple[str, ...]]:
    tokens = re.findall(r"[a-z0-9_.]+", text.casefold())
    if len(tokens) < 3:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index : index + 3]) for index in range(len(tokens) - 2)}
