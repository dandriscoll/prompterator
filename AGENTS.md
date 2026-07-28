# AGENTS.md — prompterator

Repo constraints from prior work here.

## Hard-won constraints

- **`main` carries pre-existing test failures.** As of 2026-04-19, 11 tests fail on clean `main` (improver/tuner/eval_runner, `ValueError: too many values to unpack`). Leading hypothesis: markback 0.2 migration. Before interpreting test failures during a job, establish the baseline with `git stash && pytest` so regressions are distinguishable from pre-existing drift.
- **`consolidate_feedback` discards anchor data when `existing_issues` is re-fed.** The re-feed extracts `(source, feedback)` tuples and drops prior-run `instance`/`confidence`. If re-run quality matters, carry anchor text forward in the re-feed prompt. Not yet a confirmed drift source — flagged preemptively.