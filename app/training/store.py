"""Transactional training records. The local adapter uses only the standard library."""

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3

from .data import encoded


class SQLiteRecords:
    def __init__(self, connection):
        self.connection = connection

    def get(self, kind, key):
        row = self.connection.execute(
            "SELECT value FROM records WHERE kind=? AND id=?", (kind, key)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def list(self, kind):
        return [
            json.loads(row[0])
            for row in self.connection.execute(
                "SELECT value FROM records WHERE kind=? ORDER BY rowid", (kind,)
            )
        ]

    def put(self, kind, key, value):
        self.connection.execute(
            "INSERT INTO records VALUES(?,?,?) ON CONFLICT(kind,id) DO UPDATE SET value=excluded.value",
            (kind, key, encoded(value).decode()),
        )

    def delete(self, kind, key):
        self.connection.execute("DELETE FROM records WHERE kind=? AND id=?", (kind, key))


class LocalRepository:
    def __init__(self, directory):
        self.root = Path(directory) / "training"
        self.root.mkdir(exist_ok=True, mode=0o700)
        self.database = self.root / "metadata.sqlite3"
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS records(kind TEXT,id TEXT,value TEXT NOT NULL,PRIMARY KEY(kind,id))"
            )
        self.database.chmod(0o600)

    @contextmanager
    def transaction(self):
        connection = sqlite3.connect(self.database, timeout=30)
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield SQLiteRecords(connection)
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
