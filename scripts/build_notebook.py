from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "HTBA_ARC_AGI3_PaperTrack.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


def main() -> int:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    nb.cells = [
        md(
            """
            # Hybrid Theory-Based Agents for ARC-AGI-3

            **ARC Prize 2026 - Paper Track proof-of-concept package**

            ![HTBA cover](ARC_Prize_2026_Cover.png)

            Source of truth: `ARC_Prize_2026_Writeup_Final.md`.
            Author metadata is preserved: Nay Linn Aung, Hood College,
            M.S. Computer Science (AI / Data Science), Kaggle Team: Nay Linn Aung.

            This notebook is intentionally readable before execution. Running it
            requires the official ARC-AGI-3 toolkit, pinned here as
            `arc-agi==0.9.7` unless the logged-in Kaggle runtime exposes a newer
            official version. There is no synthetic fallback on the execution
            path.
            """
        ),
        md(
            """
            ## A. Executive Summary

            ARC-AGI-3 is an interactive reasoning benchmark: the agent chooses
            what to observe, updates its model from action-conditioned frame
            deltas, and is scored by action efficiency. The blueprint frames
            this as skill acquisition under a tight action budget, not as static
            grid completion.

            The implemented proof-of-concept follows the blueprint's Hybrid
            Theory-Based Agent (HTBA). The agent represents frames as objects,
            keeps a bounded Bayesian posterior over symbolic transition
            programs, penalizes complexity with a Minimum Description Length
            prior, explores by expected information gain, and records each
            decision as an auditable trace.

            Why this approach works for the Paper Track: the code is not meant
            to claim leaderboard dominance. It verifies a precise reasoning
            architecture whose claims can be inspected step by step: observation,
            hypothesis, transformation, and validation are separate data fields,
            not prose added after the action.

            Assumptions cited from the prompt and blueprint:

            - "All files" means every readable file in this working directory.
            - The blueprint defines the intended architecture, narrative, and
              evaluation criteria.
            - ARC-AGI-3 tasks require generalization, not memorization or
              brute-force search.
            - The Kaggle notebook is a proof of concept, not a
              leaderboard-optimized solution.
            """
        ),
        md(
            """
            ## B. ARC-AGI-3 Task Understanding

            The key change from ARC-AGI-1/2 is control over information access.
            A static solver maps input grids to output grids. An ARC-AGI-3 agent
            maps history `(frame, action, next_frame)` to the next action. Wasted
            actions matter because RHAE squares action inefficiency.

            Failure modes addressed by the package:

            - Overfitting to public ARC patterns: the DSL is restricted to
              Core-Knowledge primitives such as objectness, geometry, numerosity,
              and action-conditioned causality.
            - Post-hoc reasoning: the trace schema stores the observation and
              hypothesis before validation is written.
            - Reviewer overload: the notebook introduces the formal model first,
              then shows the minimal code surface needed to reproduce it.

            Constraints:

            - Offline execution only.
            - No external API calls.
            - The official toolkit must be importable from the Kaggle runtime,
              `ARC_AGI_TOOLKIT_DIR`, or an attached Kaggle input.
            - The adapter uses official ARC competition mode when available.
            - Claims are bounded to what the code and blueprint support.
            """
        ),
        md(
            """
            ## C. Reasoning Framework Design

            ### Representation

            The frame encoder maps a raw 2-D color grid or RGB frame into a set
            of connected components. Each object has color, size, bounding box,
            centroid, shape signature, and optional motion delta from the
            previous frame. This is the implemented C1 layer.

            ### Transformation Logic

            Each hypothesis is a symbolic rule program:

            ```
            h = (primitive, trigger_action, parameters, description_length)
            ```

            The posterior score is:

            ```
            log P(h | history) = cumulative_log_likelihood(history | h)
                                 - description_length(h) / tau
                                 - normalizer
            ```

            The likelihood is computed from symbolic mismatch between the
            predicted object set and the observed next object set.

            ### Generalization Strategy

            The DSL starts deliberately small: identity, action-triggered
            translation, color-conditioned translation, deletion, and largest
            object selection. The point is not coverage of every possible game;
            the point is an inspectable inductive bias. More primitives can be
            added only if their rationale is documented and their description
            length is explicit.
            """
        ),
        md(
            """
            ## D. Model / Agent Architecture

            Implemented components:

            - C1 Frame Encoder: deterministic connected-component extraction.
            - C2 Action-Space Probe: reads alive actions from official toolkit
              metadata and handles RESET, ACTION1-7, and ACTION6 coordinates.
            - C3 Hypothesis Manager: K=64 Bayesian/MDL beam.
            - C4 Goal Inferer: stores only WIN-linked frame deltas.
            - C5 Planner: explores by expected information gain and uses bounded
              depth-6 lookahead for expected progress under the inferred reward.
            - C6 Cross-Game Memory: SHA-256 content-addressed records.
            - C7 Validation Harness: unit tests, static audit, scorecard output.

            Design rationale:

            - The deterministic object extractor is used because no local CNN
              weights are present. This is documented instead of inventing a
              learned perception capability.
            - Symbolic hypotheses are used after perception because they make
              action rationale auditable.
            - The beam is bounded to keep search controlled.

            Rejected alternatives:

            - Brute-force search: contradicts the RHAE action-efficiency target.
            - Game-specific leaderboard heuristics: weak private-set transfer.
            - Hosted language-model policies: not offline and not necessary for
              the proof-of-concept claim.
            - Opaque end-to-end policies: hard to audit against the Paper Track
              theory and completeness criteria.
            """
        ),
        md(
            """
            ## E. Implementation Details

            The public interface is:

            ```
            HTBAAgent(
                seed=0xA6C16E26,
                beam_width=64,
                eig_samples=8,
                entropy_threshold=theta,
                plan_depth=6
            )
            ```

            Required methods:

            - `reset(game)`
            - `act(frame)`
            - `observe(action, next_frame, win)`
            - `scorecard()`
            - `reasoning_trace()`

            Runtime flow:

            ```
            preflight official toolkit resources
            create arc_agi.Arcade(operation_mode=OperationMode.COMPETITION)
            iterate available official environments
            for each environment:
                reset agent and official environment
                encode frame into objects
                initialize or update posterior beam
                choose action by EIG if entropy is high
                otherwise choose bounded expected-progress action
                observe next frame and WIN signal
                write structured trace record
            save out/scorecard.json, out/reasoning_trace.json, and out/audit.html
            ```
            """
        ),
        code(
            """
            from pathlib import Path
            import json

            from htba import HTBAAgent, offline_preflight, run_required_official_evaluation
            from htba.audit import run_static_audit, write_audit_html

            SEED = 0xA6C16E26
            ROOT = Path.cwd()
            OUT = ROOT / "out"
            OUT.mkdir(exist_ok=True)
            """
        ),
        md(
            """
            ## F. Reproducibility & Execution Notes

            This cell is the hard gate. It requires the official ARC toolkit and
            leaves ARC competition-mode networking unblocked so the Kaggle
            runtime can use its required official mechanism. The adapter checks
            `ARC_AGI_TOOLKIT_DIR` and Kaggle input toolkit locations before
            relying on an installed package. If the toolkit is absent, the
            failure is intentional and should be fixed by attaching official
            resources or using the Kaggle runtime.
            """
        ),
        code(
            """
            preflight = offline_preflight(
                root=ROOT,
                require_toolkit=True,
                require_data=False,
                block_network=False,
            )
            preflight
            """
        ),
        md(
            """
            The next cell runs the proof-of-concept agent through the official
            ARC toolkit competition-mode adapter. It uses
            `Arcade(operation_mode=OperationMode.COMPETITION)`,
            `get_environments()`, one `make()` per environment, and toolkit
            action conversion. If the toolkit API changes, the adapter falls
            back only to official toolkit loader shapes; it never uses synthetic
            tasks on the submission path.
            """
        ),
        code(
            """
            agent = HTBAAgent(
                seed=SEED,
                beam_width=64,
                eig_samples=8,
                entropy_threshold=0.25,
                plan_depth=6,
                memory_dir=OUT / "memory",
            )

            scorecard = run_required_official_evaluation(
                agent=agent,
                root=ROOT,
                game_count=None,
                max_actions_per_game=1000,
            )
            print(json.dumps(scorecard, indent=2, sort_keys=True))
            print(f"reasoning trace: {OUT / 'reasoning_trace.json'}")
            """
        ),
        md(
            """
            ## G. Evaluation & Sanity Checks

            Unit-level validation covers:

            - MDL preference for shorter equivalent programs.
            - Posterior normalization.
            - Beam pruning without resurrection.
            - EIG equals zero when all hypotheses predict the same outcome.
            - Deterministic action choice for fixed seed.
            - Reward inference from WIN-linked deltas.
            - Required trace fields.

            Task-level execution is intentionally limited to official
            ARC-AGI-3 toolkit resources. The static audit checks that the notebook,
            package, seed, and cover asset are present and scans source files for
            runtime installs, hosted API markers, external URLs, and network
            clients.
            """
        ),
        code(
            """
            audit = run_static_audit(ROOT)
            write_audit_html(audit, OUT / "audit.html")
            print(json.dumps(audit.to_dict(), indent=2, sort_keys=True))
            if not audit.ok:
                raise RuntimeError("Offline static audit failed; inspect out/audit.html.")
            """
        ),
        md(
            """
            ## H. Limitations & Future Work

            Limitations:

            - The local folder did not contain official ARC-AGI-3 data or the
              official `arc_agi` toolkit at build time, so the execution path is
              a strict adapter with hard preflight rather than a demonstrated
              official run.
            - C1 is a deterministic object extractor because no CNN weights were
              supplied locally.
            - The DSL is intentionally minimal. It verifies the Bayesian/MDL
              reasoning scaffold but does not claim complete ARC-AGI-3 coverage.

            Required mitigations:

            - Risk: overfitting reasoning to ARC patterns. Mitigation: keep
              primitives Core-Knowledge-aligned and test across distinct official
              task families.
            - Risk: reasoning appears post-hoc. Mitigation: store observation,
              hypothesis, transformation, and validation as separate trace fields.
            - Risk: notebook too complex for reviewers. Mitigation: progressive
              disclosure: narrative first, public interface second, execution
              cells last.

            Future work must add primitives only with explicit description
            lengths, tests, and documented rationale.
            """
        ),
    ]
    nbf.write(nb, NOTEBOOK)
    print(f"Wrote {NOTEBOOK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
