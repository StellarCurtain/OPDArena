"""Store immutable evaluation records and raw model responses in SQLite."""

import json
import sqlite3
from pathlib import Path


class Results:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS records (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )

    def get(self, key):
        row = self.connection.execute("SELECT value FROM records WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key, value, immutable=False):
        value = json.loads(
            json.dumps(value, default=lambda x: x.tolist() if hasattr(x, "tolist") else x.item())
        )
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
        old = self.get(key)
        if immutable and old is not None and old != value:
            raise ValueError("Existing result differs: " + key)
        self.connection.execute("INSERT OR REPLACE INTO records VALUES (?,?)", (key, encoded))
        self.connection.commit()

    def publish(self):
        self.connection.commit()

    def close(self):
        self.connection.close()
