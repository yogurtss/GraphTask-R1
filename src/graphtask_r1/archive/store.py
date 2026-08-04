from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from graphtask_r1.schema import TaskCertificate


class TaskArchive:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS tasks "
            "(signature TEXT PRIMARY KEY, task_id TEXT UNIQUE, payload TEXT)"
        )
        self.connection.commit()

    def add(self, task: TaskCertificate) -> bool:
        try:
            self.connection.execute(
                "INSERT INTO tasks(signature, task_id, payload) VALUES (?, ?, ?)",
                (task.program_signature, task.task_id, task.model_dump_json()),
            )
            self.connection.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def all(self) -> list[TaskCertificate]:
        rows = self.connection.execute("SELECT payload FROM tasks ORDER BY task_id").fetchall()
        return [TaskCertificate.model_validate(json.loads(row[0])) for row in rows]

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> TaskArchive:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
