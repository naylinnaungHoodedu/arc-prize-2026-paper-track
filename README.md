# HTBA ARC-AGI-3 Paper Track Package

Repository: <https://github.com/naylinnaungHoodedu/arc-prize-2026-paper-track>

This package implements the local strategic blueprint for a Paper Track
proof-of-concept: a Hybrid Theory-Based Agent with object-centric perception,
Bayesian/MDL hypothesis maintenance, information-gain action selection, reward
inference from WIN-linked deltas, and auditable reasoning traces.

The notebook is `HTBA_ARC_AGI3_PaperTrack.ipynb`. It is written so a reviewer
can understand the approach before running code.

For a professional record of completed work, validation evidence, and known
limitations, see `ACTIVITY_LOG.md`.

## Offline Execution

Execution intentionally requires official local ARC-AGI-3 resources:

- `ARC_AGI_DATA_DIR` must point to the local official data, or data must exist
  at `./data/arc_agi_3`.
- The local `arc_agi` toolkit must be importable.
- The preflight guard blocks non-localhost socket connections.

There is no synthetic fallback for the Kaggle execution path. Unit tests use
small arrays only to validate reasoning primitives.

## Commands

```bash
python -m pytest -m "not integration"
python scripts/run_audit.py
```

On systems with `make`:

```bash
make test-all
```

The audit writes `out/scorecard.json` and `out/audit.html`.
