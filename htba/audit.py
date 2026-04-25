"""Static and runtime audit helpers for the offline package."""

from __future__ import annotations

from dataclasses import dataclass, field
import html
import json
from pathlib import Path
import re
from typing import Any


AUDIT_PATTERNS = {
    "runtime_install": re.compile(r"(%pip|!pip|pip\s+install)", re.IGNORECASE),
    "hosted_api": re.compile(r"\b(openai|anthropic|cohere|googleapis)\b", re.IGNORECASE),
    "external_url": re.compile(r"https?://[^\s>)`]+", re.IGNORECASE),
    "network_client": re.compile(r"\b(requests\.|urllib\.request|aiohttp)\b", re.IGNORECASE),
}

TEXT_EXTENSIONS = {".py", ".ipynb", ".md", ".txt", ".toml", ".yml", ".yaml", ".json"}
ROOT_TEXT_FILES = {"Makefile"}


@dataclass
class AuditResult:
    checks: dict[str, bool] = field(default_factory=dict)
    findings: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(self.checks.values()) and not self.findings

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": self.checks,
            "findings": self.findings,
        }


def iter_text_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        if path.name in ROOT_TEXT_FILES or path.suffix.lower() in TEXT_EXTENSIONS:
            if ".ipynb_checkpoints" not in path.parts and "out" not in path.parts:
                files.append(path)
    return files


def run_static_audit(root: Path | str = ".") -> AuditResult:
    root_path = Path(root).resolve()
    result = AuditResult()
    result.checks["seed_documented"] = _contains(root_path, "0xA6C16E26")
    result.checks["notebook_present"] = (root_path / "HTBA_ARC_AGI3_PaperTrack.ipynb").exists()
    result.checks["package_present"] = (root_path / "htba" / "agent.py").exists()
    result.checks["cover_present"] = (root_path / "ARC_Prize_2026_Cover.png").exists()

    for path in iter_text_files(root_path):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, pattern in AUDIT_PATTERNS.items():
            for match in pattern.finditer(text):
                if _allowed_match(path, name, match.group(0)):
                    continue
                result.findings.append(
                    {
                        "path": str(path.relative_to(root_path)),
                        "check": name,
                        "match": match.group(0),
                    }
                )
    return result


def _contains(root: Path, needle: str) -> bool:
    for path in iter_text_files(root):
        try:
            if needle in path.read_text(encoding="utf-8", errors="ignore"):
                return True
        except OSError:
            continue
    return False


def _allowed_match(path: Path, name: str, match: str) -> bool:
    if name == "runtime_install" and path.parts[-2:] == ("htba", "audit.py"):
        return True
    if name == "hosted_api" and path.parts[-2:] == ("htba", "audit.py"):
        return True
    if name == "network_client" and path.parts[-2:] == ("htba", "audit.py"):
        return True
    if name == "external_url" and path.parts[-2:] == ("htba", "audit.py"):
        return True
    if name == "external_url" and match.rstrip("/").lower() == (
        "https://github.com/naylinnaunghoodedu/arc-prize-2026-paper-track"
    ):
        return True
    return False


def write_audit_html(result: AuditResult, path: Path | str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = html.escape(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    status = "PASS" if result.ok else "FAIL"
    output_path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset=\"utf-8\"><title>HTBA Audit</title></head>",
                "<body>",
                f"<h1>HTBA Offline Audit: {status}</h1>",
                "<pre>",
                payload,
                "</pre>",
                "</body></html>",
            ]
        ),
        encoding="utf-8",
    )
    return output_path
