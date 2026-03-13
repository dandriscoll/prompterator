# Structured Edit Formalism

The improve/tune loop uses a **structured edit formalism** to modify prompts.
Instead of asking an LLM to output a complete rewritten prompt (which causes
the LLM to *execute* the prompt rather than edit it), we ask for a structured
edit instruction that Python applies mechanically.

## The Edit Format

The editor LLM must respond in exactly this format:

```
RATIONALE: one sentence explaining the change
ACTION: REPLACE | APPEND | PREPEND
FIND: (REPLACE only) exact text to find in the prompt
REPLACE_WITH: (REPLACE only) text to replace it with
APPEND_TEXT: (APPEND only) text to add at the end
PREPEND_TEXT: (PREPEND only) text to add at the beginning
```

## Why a Formalism?

Asking an LLM to "improve this prompt" and output the full result fails for
several reasons:

1. **Execution vs editing** — The LLM sees the prompt text and follows it
   instead of modifying it. A prompt saying "You are a productivity assistant"
   causes the LLM to act as a productivity assistant rather than edit the text.

2. **Separator collisions** — Using `---` or `===` to separate rationale from
   prompt content collides with markdown in the prompt itself.

3. **Commentary injection** — The LLM embeds its own commentary ("I've
   improved the greeting rules") inside the prompt text.

The structured edit format solves all three: the LLM never outputs prompt text
directly. It outputs a *recipe* (FIND this, REPLACE WITH that) which Python
applies via string operations.

## Edit Application Pipeline

```
LLM response → _parse_field() → _apply_edit() → edited prompt
```

### Parse

Fields are extracted by regex: each known field name (`RATIONALE:`,
`ACTION:`, `FIND:`, etc.) marks a boundary. Multi-line values are supported.

Post-processing:
- Surrounding quotes are stripped (LLMs often wrap values in `"..."`)
- Literal `\n` sequences are unescaped to real newlines

### Apply

**REPLACE**: Find exact substring in prompt, replace first occurrence.
Fallbacks: (1) try with unescaped `\n`, (2) fuzzy match normalizing whitespace.
If no match found, the edit is **rejected** (prompt unchanged).

**APPEND**: Add text after the prompt. Before applying, a **similarity check**
compares the new text against existing paragraphs. If word overlap exceeds 60%,
the edit is rejected as redundant.

**PREPEND**: Add text before the prompt. No additional checks.

### Rejection

Edits are rejected (return original prompt unchanged) when:
- FIND text doesn't match anything in the prompt
- APPEND text is too similar to existing content
- The edit could not be parsed

Rejected edits are recorded in the edit history so the editor LLM can see
what failed and try a different approach.

## Guardrails

The formalism enables guardrails that would be impossible with free-text
rewriting:

### Redundancy Detection

Before applying an APPEND, `_is_similar_to_existing()` checks word overlap
against every paragraph in the current prompt. This prevents the common
failure mode where the LLM appends the same rule 5+ times with minor
rewording.

### Duplicate Line Detection

`_find_duplicate_lines()` scans the prompt for near-identical lines. When
found, the editor LLM is shown the specific duplicate lines and told to use
REPLACE to consolidate them.

### Stall Escalation

The tuner tracks `stall_count` (iterations without score improvement) and
passes it to the editor. Warnings escalate:

- **stall_count >= 3**: Suggest trying a different approach, different eval
  target, or different action type.
- **stall_count >= 6 or 4+ consecutive APPENDs**: Require REPLACE. Block
  APPEND entirely (via the similarity check becoming more aggressive and
  the warning becoming a directive).

### Edit History

The last 8 edit attempts (rationale, action type, accepted/rejected) are
shown to the editor LLM. This prevents the model from proposing the same
edit repeatedly and lets it learn from what worked.

## Self-Review via the Formalism

Because edits are structured data, they can be **reviewed by a second LLM
pass** before application. The reviewer sees:

- The current prompt text
- The proposed edit (ACTION, FIND, REPLACE_WITH / APPEND_TEXT)
- The failing evals

And can reject or revise the edit before `_apply_edit()` runs. This is
possible *because* the edit is inspectable structured data, not an opaque
rewritten prompt.

## Line Numbers

The prompt is shown to the editor with line numbers:

```
  1| You are a productivity assistant.
  2|
  3| Here is the to-do list:
```

This helps the editor reference specific locations without copying text
verbatim (which small models often get wrong).
