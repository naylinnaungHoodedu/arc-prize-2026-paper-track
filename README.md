# HTBA ARC-AGI-3 Paper Track Package

Repository: <https://github.com/naylinnaungHoodedu/arc-prize-2026-paper-track>

This package implements the local strategic blueprint for a Paper Track
proof-of-concept: a Hybrid Theory-Based Agent with object-centric perception,
Bayesian/MDL hypothesis maintenance, information-gain action selection, reward
inference from WIN-linked deltas, and auditable reasoning traces.

The notebook is `HTBA_ARC_AGI3_PaperTrack.ipynb`. It is written so a reviewer
can understand the approach before running code. The Kaggle Paper Track body is
`ARC_Prize_2026_Writeup_Final.md`, with a rendered Word copy at
`ARC_Prize_2026_Writeup_Final.docx`.

For a professional record of completed work, validation evidence, and known
limitations, see `ACTIVITY_LOG.md`.

## Competition Execution

The Kaggle ARC-AGI-3 competition runtime is expected to force the official
toolkit into competition mode. This package therefore pins the public toolkit
version verified during final preparation:

- `arc-agi==0.9.7`

Execution intentionally requires official ARC-AGI-3 resources:

- `ARC_AGI_TOOLKIT_DIR` may point to a vendored local `arc_agi` toolkit; the
  adapter also checks recognized Kaggle input paths before relying on an
  already-importable package.
- When `arc_agi.Arcade` and `OperationMode` are available, the adapter uses
  `OperationMode.COMPETITION`, `get_environments()`, one `make()` call per
  environment, toolkit `GameAction` conversion, and `env.step(action, data=...)`.
- `ARC_AGI_DATA_DIR` remains supported for local offline resources, but the
  official competition path does not require a synthetic local data fallback.
- The optional socket guard is for local audits only. It is not enabled during
  the official competition-mode run because ARC-managed endpoints must remain
  reachable when the toolkit requires them.

There is no synthetic fallback for the Kaggle execution path. Unit tests use
small arrays only to validate reasoning primitives.

Official notebook execution writes:

- `out/scorecard.json`
- `out/reasoning_trace.json`
- `out/audit.html`

## Commands

```bash
python -m pytest -m "not integration"
python scripts/run_audit.py
```

On systems with `make`:

```bash
make test-all
```

The audit writes `out/scorecard.json`, `out/reasoning_trace.json`, and
`out/audit.html`.

## Submission Checklist

1. Run `python -m pytest -m "not integration"` and `python scripts/run_audit.py`.
2. Commit and push the final package, then tag the submitted revision.
3. Create the ARC-AGI-3 Kaggle code submission from
   `HTBA_ARC_AGI3_PaperTrack.ipynb`.
4. Paste the resulting Kaggle code submission URL into the Paper Track writeup
   before pressing Submit.

## License

Authored code is MIT-0. Documentation, writeups, generated narrative artifacts,
and visual assets are CC0-1.0. Upstream dependencies and official ARC/Kaggle
resources retain their own licenses.
