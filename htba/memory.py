"""Deterministic cross-game memory keyed by Core-Knowledge signatures."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


def content_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class CrossGameMemory:
    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def key_for(self, primitive_set: dict[str, Any], action_signature: list[str]) -> str:
        return content_hash(
            {
                "primitive_set": primitive_set,
                "action_signature": sorted(action_signature),
            }
        )

    def load(self, key: str) -> dict[str, Any] | None:
        path = self.root / f"{key}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def append(self, key: str, record: dict[str, Any]) -> Path:
        path = self.root / f"{key}.json"
        existing = self.load(key) or {"key": key, "records": []}
        existing["records"].append(record)
        path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")
        return path
