from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from htba.audit import run_static_audit, write_audit_html


def main() -> int:
    root = ROOT
    result = run_static_audit(root)
    out_dir = root / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scorecard.json").write_text(
        json.dumps({"static_audit": result.to_dict()}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "reasoning_trace.json").write_text(
        json.dumps(
            {
                "source": "static_audit_only",
                "records": [],
                "note": "Official reasoning traces are written by the notebook evaluation path.",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    write_audit_html(result, out_dir / "audit.html")
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
