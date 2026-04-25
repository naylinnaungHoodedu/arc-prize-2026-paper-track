from pathlib import Path

from htba.audit import run_static_audit, write_audit_html


def test_static_audit_result_shape(tmp_path):
    root = tmp_path
    (root / "htba").mkdir()
    (root / "htba" / "agent.py").write_text("SEED = 0xA6C16E26\n", encoding="utf-8")
    (root / "HTBA_ARC_AGI3_PaperTrack.ipynb").write_text("{}", encoding="utf-8")
    (root / "ARC_Prize_2026_Cover.png").write_bytes(b"png")

    result = run_static_audit(root)
    output = write_audit_html(result, root / "out" / "audit.html")

    assert isinstance(result.checks, dict)
    assert Path(output).exists()
