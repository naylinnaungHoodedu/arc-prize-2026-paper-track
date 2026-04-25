import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "HTBA_ARC_AGI3_PaperTrack.ipynb"


def _load_notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_notebook_contains_required_a_to_h_sections():
    text = "\n".join("".join(cell.get("source", [])) for cell in _load_notebook()["cells"])
    for section in [
        "## A. Executive Summary",
        "## B. ARC-AGI-3 Task Understanding",
        "## C. Reasoning Framework Design",
        "## D. Model / Agent Architecture",
        "## E. Implementation Details",
        "## F. Reproducibility & Execution Notes",
        "## G. Evaluation & Sanity Checks",
        "## H. Limitations & Future Work",
    ]:
        assert section in text


def test_notebook_code_cells_parse():
    for index, cell in enumerate(_load_notebook()["cells"], start=1):
        if cell.get("cell_type") != "code":
            continue
        ast.parse("".join(cell.get("source", [])), filename=f"cell-{index}")


def test_notebook_has_no_runtime_installs_or_network_clients():
    text = "\n".join("".join(cell.get("source", [])) for cell in _load_notebook()["cells"])
    forbidden = [
        rf"(%{'pip'}|!{'pip'}|{'pip'}\s+{'install'})",
        rf"\b({'requests'}\.|{'urllib'}\.{'request'}|{'aio'}{'http'})\b",
    ]
    for pattern in forbidden:
        assert re.search(pattern, text, flags=re.IGNORECASE) is None
