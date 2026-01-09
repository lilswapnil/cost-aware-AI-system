"""SQLite-backed exact prompt cache."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class CacheEntry:
    key: str
    response: Dict[str, Any]


class PromptCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prompt_cache (
                key TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.commit()

    def _make_key(self, prompt: str, constraints: Dict[str, Any]) -> str:
        payload = json.dumps({"prompt": prompt, "constraints": constraints}, sort_keys=True)
        return sha256(payload.encode("utf-8")).hexdigest()

    def get(self, prompt: str, constraints: Dict[str, Any]) -> Optional[CacheEntry]:
        key = self._make_key(prompt, constraints)
        cursor = self._conn.execute(
            "SELECT response_json FROM prompt_cache WHERE key = ?", (key,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        response = json.loads(row[0])
        return CacheEntry(key=key, response=response)

    def set(self, prompt: str, constraints: Dict[str, Any], response: Dict[str, Any]) -> None:
        key = self._make_key(prompt, constraints)
        self._conn.execute(
            "INSERT OR REPLACE INTO prompt_cache (key, prompt, response_json) VALUES (?, ?, ?)",
            (key, prompt, json.dumps(response)),
        )
        self._conn.commit()
