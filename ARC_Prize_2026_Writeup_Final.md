# Hybrid Theory-Based Agents for ARC-AGI-3

## Abstract

ARC-AGI-3 changes ARC from static grid induction into interactive skill acquisition. An agent is not merely asked to infer an output grid; it must choose actions, observe consequences, infer dynamics, and identify useful goals without instructions. We present the Hybrid Theory-Based Agent (HTBA), a proof-of-concept architecture for this setting. HTBA combines deterministic object-centric perception, a Bayesian posterior over symbolic transition programs, a Minimum Description Length prior, WIN-linked goal inference, and information-gain action selection. The contribution is a reproducible and auditable reasoning scaffold whose decisions are recorded as separate observation, hypothesis, transformation, validation, posterior, action, and rationale fields.

Submission linkage: ARC-AGI-3 Kaggle code submission at https://www.kaggle.com/code/naylinnaunghood/htba-arc-agi-3-paper-track (version 2). Official RHAE score: 0.0 across 25 environments, 5,002 total actions, 0 levels completed. Scorecard ID: 2c3b03b7-c2c5-4c55-9d0e-86441dd2af4b.

## 1. Introduction

ARC-AGI-3 tests exploration, modeling, goal-setting, and planning under tight action budgets. This makes brute-force probing a poor conceptual fit: every unnecessary action spends information budget and weakens action-efficiency scoring. The practical challenge is to act like an experiment designer. The agent should choose actions that most reduce uncertainty until it has enough evidence to exploit an inferred goal.

HTBA is built around that premise. It does not use hosted language-model calls, runtime package installation, or task-specific public-set tricks. The package is designed to run through the official ARC toolkit in competition mode and to fail clearly if the official toolkit is absent.

## 2. Prior Work

Static ARC systems typically emphasize program synthesis, compression, test-time search, or learned priors over input-output examples. Those ideas remain relevant, but ARC-AGI-3 adds an interactive layer: the system controls which observations it receives. HTBA borrows the parsimony pressure of MDL-style ARC approaches and the hypothesis-driven exploration logic of theory-based reinforcement learning, then adapts both to action-conditioned frame transitions.

The design also responds to preview-era ARC-AGI-3 agents that used action probing, state graphs, and lightweight learned policies. HTBA differs by making the transition model explicit and auditable: each action is tied to a symbolic hypothesis and an expected information-gain or expected-progress rationale.

## 3. Approach

The state at time `t` is represented as:

```text
S_t = (O_t, H_t, r_hat_t, m_t)
```

`O_t` is an object set extracted from the current frame. Each object has color, size, bounding box, centroid, shape signature, and optional motion delta. `H_t` is a bounded beam of symbolic transition programs. `r_hat_t` is an inferred reward estimate based only on frame deltas observed with a WIN signal. `m_t` is deterministic cross-game memory keyed by content hashes.

Each hypothesis `h` is scored by evidence and description length:

```text
log P(h | history) = log P(history | h) - L(h) / tau - normalizer
```

The DSL is deliberately small: identity, action-triggered translation, color-conditioned translation, deletion, and largest-object retention. This limited vocabulary is a design constraint, not an omission. A primitive should be added only when it has a Core-Knowledge rationale, an explicit description length, and tests.

For exploration, the planner chooses the action with maximum expected posterior entropy reduction:

```text
EIG(a) = E[H(P(h | history)) - H(P(h | history, a, next_frame))]
```

When posterior entropy falls below the threshold, the planner switches to bounded depth-6 expected progress under the WIN-linked reward estimate. The action interface follows the official ARC-AGI-3 action set: `RESET`, `ACTION1` through `ACTION7`, with `ACTION6(x,y)` for coordinate actions.

## 4. Implementation

The public agent API is intentionally small:

```text
HTBAAgent.reset(game)
HTBAAgent.act(frame)
HTBAAgent.observe(action, next_frame, win)
HTBAAgent.scorecard()
HTBAAgent.reasoning_trace()
```

The official adapter uses `arc_agi.Arcade(operation_mode=OperationMode.COMPETITION)` when the toolkit exposes it. It obtains environments with `get_environments()`, calls `make()` once per environment, converts selected actions to toolkit `GameAction` values when possible, passes coordinate payloads through `env.step(action, data=...)`, attempts `close_scorecard()` after all environments run, falls back to `get_scorecard()` if final details are withheld, and writes `out/scorecard.json` plus `out/reasoning_trace.json`.

The implementation pins the current public toolkit version verified during final preparation as `arc-agi==0.9.8`, unless the logged-in Kaggle runtime exposes a newer official version. Authored code is MIT-0. Documentation, writeups, generated narrative artifacts, and visual assets are CC0-1.0.

## 5. Results and Reproducibility

Local validation covers the reasoning scaffold rather than official task performance. The non-integration test suite validates object extraction, motion deltas, MDL preference, posterior normalization, beam pruning, zero information gain when hypotheses agree, deterministic planning, reward inference, trace fields, action canonicalization, ACTION6 coordinate generation, ACTION7 handling, notebook structure, and a mocked official competition-mode adapter path.

The package writes three review artifacts:

```text
out/scorecard.json
out/reasoning_trace.json
out/audit.html
```

The official ARC-AGI-3 run must be performed on Kaggle or with a local official toolkit install. The local workspace used for final preparation did not include an importable `arc_agi` toolkit, so no official RHAE score is claimed here. The final Paper Track submission should report the actual Kaggle code submission link and score after the Kaggle notebook has been committed and submitted.

## 6. Limitations

HTBA is intentionally conservative. First, the object encoder is deterministic connected-component extraction; it does not claim a trained CNN perception module. Second, the symbolic DSL is incomplete by design. It can validate the Bayesian-MDL reasoning loop, but it cannot express every possible ARC-AGI-3 mechanic. Third, WIN-linked goal inference is sparse: before a WIN trajectory is observed, expected progress remains deliberately cautious.

These limitations are useful because they keep the paper falsifiable. The implementation makes it clear which claims are demonstrated locally and which claims require the official Kaggle run.

## 7. Conclusion

HTBA argues that interactive ARC progress should be measured not only by final score, but also by the quality of the agent's experiment design and audit trail. The core contribution is a compact Bayesian-MDL loop for turning action-conditioned observations into explicit hypotheses and efficient actions. The same structure can transfer beyond ARC-AGI-3 whenever a system must learn object dynamics, infer sparse goals, and plan under uncertainty.

## References

1. ARC Prize Foundation. ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence. arXiv:2603.24621, 2026.
2. Chollet, F. On the Measure of Intelligence. arXiv:1911.01547, 2019.
3. Tsividis, P. et al. Human-Level Reinforcement Learning through Theory-Based Modeling, Exploration, and Planning. arXiv:2107.12544, 2021.
4. Grunwald, P. The Minimum Description Length Principle. MIT Press, 2007.
5. ARC Prize 2026 Paper Prize overview and ARC-AGI-3 toolkit documentation, accessed during final package preparation.
