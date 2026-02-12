# Prompterator Playground Walkthrough

## Phase 0 — Setup

**Goal:** Ensure LLM credentials and config are in place.

**Config file:** `prompterator.yaml`
- Set `stacks.default` to your LLM provider (Anthropic, OpenAI, or Azure OpenAI)
- Export the required API key environment variable

**Command:**
```
prompterator status
```

**Verify:** Status reports no errors, stack is reachable.

---

## Phase 1 — Feedback

**Goal:** Parse `.mb` feedback files and confirm observations are read correctly.

**Input:** `*.mb` files (free-form observations in markback format)
**Output:** Terminal display of parsed entries

**Command:**
```
prompterator feedback
```

**Verify:**
- Each `.mb` file appears with its correct `prompt_ref`
- Entry text matches what you wrote (no mangled parsing)
- Entry count per file is correct

---

## Phase 2 — Issues

**Goal:** Cluster feedback into problem-oriented issues via LLM.

**Input:** `*.mb` files
**Output:** `issues/<name>.issue.yaml`

**Commands:**
```
prompterator issues --dry-run   # preview clusters without writing
prompterator issues             # write issue files
```

**Verify:**
- Clusters represent distinct real problems (not generic categories)
- Evidence entries are assigned to the right cluster
- Severity levels make sense (high = appears in most sources)
- No junk or duplicate clusters

---

## Phase 3 — Evals

**Goal:** Generate eval specs with testable criteria from issue evidence.

**Input:** `issues/<name>.issue.yaml`
**Output:** `evals/<name>.eval.yaml`

**Command:**
```
prompterator evals
```

**Verify:**
- One eval per issue
- Criteria are specific and derived from evidence text
- High-severity issues use `all_required` scoring; medium/low use `any_required`

---

## Phase 4 — Baseline

**Goal:** Prove the evals catch the known problems by testing the original prompt.

**Input:** Original prompt file + `evals/<name>.eval.yaml`
**Output:** `results/<name>.results.yaml`

**Command:**
```
prompterator test <prompt> -v
```

**Verify:**
- Most evals should **fail** — this confirms the evals are working
- Any eval that passes should make sense (the prompt already handles that problem)
- Record the baseline score; this is the "before" snapshot

---

## Phase 5 — Tune

**Goal:** Iteratively improve the prompt and measure progress.

**Input:** Prompt + `issues/<name>.issue.yaml` + `evals/<name>.eval.yaml`
**Output:** `.prompterator/tune/tune-report.yaml` (iteration history) + `.prompterator/tune/<prompt>` (final prompt)

**Command:**
```
prompterator tune <prompt> --max-iterations 5
```

**Verify:**
- Scores trend upward across iterations in the metric table
- Final prompt is a genuine improvement — read it and confirm changes are sensible
- No over-fitting to eval wording (prompt should read naturally)
- Compare baseline score (Phase 4) against final score
