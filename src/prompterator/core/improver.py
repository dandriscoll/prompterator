"""Prompt improvement logic - generate improved prompts via LLM.

Uses a single LLM call to produce a structured edit (FIND/REPLACE or APPEND),
then applies it programmatically. This avoids a second LLM call that would
see the prompt text and try to execute it instead of editing it.
"""

import re
from pathlib import Path

from prompterator.models.issue import IssueFile
from prompterator.models.result import EvalResult
from prompterator.runners.llm import LLMClient


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_EDIT_SYSTEM = """\
You are a prompt editor. You edit PROMPT TEMPLATES — text that will be sent \
to an LLM to control its behavior. You are NOT the LLM that will follow the \
prompt. You are editing the prompt text to make it better.

ISSUES describe problems with the LLM's output when it follows the prompt. \
EVALS check whether the prompt contains the right instructions to prevent \
those problems. To fix a failing eval, ADD or MODIFY instructions in the \
prompt that tell the LLM what to do or not do.

IMPORTANT: Do NOT delete or shorten the existing prompt. Instead, ADD new \
rules and constraints. For example, if the LLM's output has unwanted \
preamble, do NOT remove text from the prompt — instead ADD an instruction \
like "Output only the improved list with no introduction or commentary."

Respond in EXACTLY this format — no other text:

RATIONALE: one sentence saying what instruction you are adding and why
ACTION: REPLACE | APPEND | PREPEND
FIND: (REPLACE only) exact text copied from the prompt — no quotes
REPLACE_WITH: (REPLACE only) replacement text — no quotes
APPEND_TEXT: (APPEND only) text to add at the end
PREPEND_TEXT: (PREPEND only) text to add at the beginning

Rules:
- FIND must be an EXACT substring copied verbatim from the prompt.
- Do NOT wrap any field values in quotation marks.
- Add specific, concrete instructions. Vague edits do not help.
- GOOD: APPEND_TEXT that says "Do not add introductory text before the list."
- BAD: Removing "You are a productivity assistant" from the prompt.
- Prefer APPEND to add new constraints. Use REPLACE only to modify existing \
instructions, not to delete them.
- Do not regress passing evals.
- Each edit should address exactly one failing eval."""


# ---------------------------------------------------------------------------
# Build prompt
# ---------------------------------------------------------------------------

def _add_line_numbers(text: str) -> str:
    """Add line numbers to each line of the prompt text."""
    lines = text.split('\n')
    numbered = [f"{i:3d}| {line}" for i, line in enumerate(lines, 1)]
    return '\n'.join(numbered)


def _build_edit_prompt(
    original_prompt: str,
    issue_file: IssueFile,
    eval_results: list[EvalResult] | None = None,
    iteration: int | None = None,
    edit_history: list[dict] | None = None,
    stall_count: int = 0,
) -> str:
    """Build the prompt for the edit LLM call."""
    issues_text = []
    for issue in issue_file.issues:
        issues_text.append(
            f"- [{issue.severity.upper()}] {issue.id} ({issue.category}): {issue.summary}"
        )
    issues_section = "\n".join(issues_text) if issues_text else "No specific issues identified."

    numbered_prompt = _add_line_numbers(original_prompt)

    prompt = f"""PROMPT TO EDIT (with line numbers):
{numbered_prompt}

ISSUES:
{issues_section}"""

    if eval_results:
        failing = [r for r in eval_results if not r.passed]
        passing = [r for r in eval_results if r.passed]
        if failing:
            fail_text = []
            for r in failing:
                line = f"- FAIL {r.eval_id} (score={r.score:.2f})"
                if r.details:
                    line += f" — {r.details}"
                fail_text.append(line)
            prompt += f"\n\nFAILING EVALS:\n{chr(10).join(fail_text)}"
        if passing:
            pass_text = [f"- PASS {r.eval_id}" for r in passing]
            prompt += f"\n\nPASSING EVALS (do not regress):\n{chr(10).join(pass_text)}"

    if edit_history:
        history_lines = []
        for entry in edit_history[-8:]:  # Last 8 attempts
            status = "ACCEPTED" if entry.get("accepted") else "REJECTED"
            action = entry.get("action", "?")
            line = f"- [{status}] ({action}) {entry['rationale']}"
            history_lines.append(line)
        prompt += f"\n\nPREVIOUS EDIT ATTEMPTS:\n"
        prompt += chr(10).join(history_lines)

    # Detect plateau: score hasn't improved in several iterations
    warnings = []
    if stall_count >= 3:
        # Count consecutive appends in history
        recent_appends = sum(
            1 for e in (edit_history or [])[-5:]
            if e.get("action") == "APPEND"
        )
        if stall_count >= 6 or recent_appends >= 4:
            warnings.append(
                "CRITICAL: The score has not improved over many iterations and "
                "recent edits have been APPEND-only. You MUST use REPLACE to "
                "rewrite or consolidate existing instructions. Do NOT use APPEND."
            )
        else:
            warnings.append(
                "The score has not improved in the last several iterations. "
                "Try a COMPLETELY DIFFERENT approach — target a different "
                "failing eval, use REPLACE instead of APPEND, or phrase the "
                "instruction in a fundamentally new way."
            )

    # Detect repetitive prompt: find concrete duplicate lines
    dupes = _find_duplicate_lines(original_prompt)
    if dupes:
        dupe_examples = "\n".join(
            f"  ({count}x) \"{text[:80]}{'...' if len(text) > 80 else ''}\""
            for text, count in dupes[:3]
        )
        warnings.append(
            "The prompt contains REDUNDANT lines:\n"
            f"{dupe_examples}\n"
            "Use REPLACE to consolidate these into a single clear instruction. "
            "Copy one of the duplicate lines as FIND and write a better version "
            "as REPLACE_WITH."
        )

    if warnings:
        prompt += "\n\n" + "\n".join(f"WARNING: {w}" for w in warnings)

    if iteration is not None:
        prompt += f"\n\nITERATION: {iteration}"

    return prompt


# Keep old name as alias for dry-run in improve command
_build_diagnose_prompt = _build_edit_prompt


# ---------------------------------------------------------------------------
# Redundancy detection
# ---------------------------------------------------------------------------

def _find_duplicate_lines(text: str) -> list[tuple[str, int]]:
    """Find lines in the prompt that are near-duplicates of other lines.

    Returns list of (line_text, count) for lines appearing more than once.
    """
    lines = [ln.strip() for ln in text.split('\n') if len(ln.strip()) > 30]
    normed: dict[str, list[str]] = {}
    for ln in lines:
        key = ' '.join(ln.lower().split())
        normed.setdefault(key, []).append(ln)
    return [(group[0], len(group)) for group in normed.values() if len(group) > 1]


def _is_similar_to_existing(new_text: str, original: str, threshold: float = 0.6) -> bool:
    """Check if new_text is too similar to any existing paragraph in original.

    Uses word overlap ratio as a simple similarity measure.
    """
    new_words = set(new_text.lower().split())
    if len(new_words) < 5:
        return False

    for para in original.split('\n\n'):
        para = para.strip()
        if len(para) < 30:
            continue
        # Check each line in the paragraph too
        for segment in [para] + para.split('\n'):
            segment = segment.strip()
            if len(segment) < 30:
                continue
            seg_words = set(segment.lower().split())
            if not seg_words:
                continue
            overlap = len(new_words & seg_words) / min(len(new_words), len(seg_words))
            if overlap >= threshold:
                return True
    return False


# ---------------------------------------------------------------------------
# Parse and apply edit
# ---------------------------------------------------------------------------

def _strip_surrounding_quotes(text: str) -> str:
    """Strip surrounding double or single quotes that LLMs sometimes add."""
    if len(text) >= 2:
        if (text[0] == '"' and text[-1] == '"') or (text[0] == "'" and text[-1] == "'"):
            return text[1:-1]
    return text


def _unescape_literals(text: str) -> str:
    """Convert literal \\n sequences to actual newlines."""
    return text.replace('\\n', '\n')


def _parse_field(response: str, field: str) -> str | None:
    """Extract a field value from the structured response.

    Handles multi-line values: everything from 'FIELD: value' until the next
    known field or end of string.
    """
    known_fields = (
        "RATIONALE:", "ACTION:", "FIND:", "REPLACE_WITH:",
        "APPEND_TEXT:", "PREPEND_TEXT:",
    )
    pattern = re.compile(
        rf"^{re.escape(field)}\s*(.*?)(?=^(?:{'|'.join(re.escape(f) for f in known_fields)})|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(response)
    if match:
        value = match.group(1).strip()
        value = _strip_surrounding_quotes(value)
        return value
    return None


def _apply_edit(original: str, response: str) -> tuple[str, str, bool, str]:
    """Parse the structured edit response and apply it to the original prompt.

    Returns:
        Tuple of (edited_prompt, rationale, success, action).
    """
    rationale = _parse_field(response, "RATIONALE:") or "No rationale provided"
    action = (_parse_field(response, "ACTION:") or "").upper().strip()

    if action == "REPLACE":
        find_text = _parse_field(response, "FIND:")
        replace_with = _parse_field(response, "REPLACE_WITH:")
        if find_text and replace_with is not None:
            # Try literal escape sequences
            find_unescaped = _unescape_literals(find_text)
            replace_unescaped = _unescape_literals(replace_with)

            # Try exact match with original and unescaped versions
            for ft, rw in [(find_text, replace_with), (find_unescaped, replace_unescaped)]:
                if ft in original:
                    edited = original.replace(ft, rw, 1)
                    return edited, rationale, True, action

            # Try fuzzy match — strip whitespace differences
            find_normalized = re.sub(r'\s+', ' ', find_text.strip())
            for i in range(len(original)):
                for j in range(i + 10, min(i + len(find_text) * 3, len(original) + 1)):
                    candidate = original[i:j]
                    if re.sub(r'\s+', ' ', candidate.strip()) == find_normalized:
                        edited = original[:i] + replace_with + original[j:]
                        return edited, rationale, True, action

            # FIND text not found — do NOT append, return unchanged
            return original, rationale + " (FIND text not matched — edit rejected)", False, action

    elif action == "APPEND":
        append_text = _parse_field(response, "APPEND_TEXT:")
        if append_text:
            append_text = _unescape_literals(append_text)
            # Reject if too similar to existing content
            if _is_similar_to_existing(append_text, original):
                return original, rationale + " (APPEND rejected — too similar to existing text, use REPLACE to consolidate)", False, action
            edited = original.rstrip() + "\n\n" + append_text
            return edited, rationale, True, action

    elif action == "PREPEND":
        prepend_text = _parse_field(response, "PREPEND_TEXT:")
        if prepend_text:
            prepend_text = _unescape_literals(prepend_text)
            edited = prepend_text + "\n\n" + original.lstrip()
            return edited, rationale, True, action

    # Fallback: couldn't parse a valid edit
    return original, rationale + " (edit could not be applied)", False, action or "UNKNOWN"


# ---------------------------------------------------------------------------
# Self-review: let the LLM critique and revise its own edit
# ---------------------------------------------------------------------------

_REVIEW_SYSTEM = """\
You are a prompt edit reviewer. You will see a proposed edit to a prompt \
template and the failing evals it is trying to fix. Your job is to decide \
whether the edit is good, and if not, produce a better one.

Common problems with edits:
- Too specific: targets one example instead of writing a general rule
- Redundant: says the same thing as an existing instruction in different words
- Too vague: says "ensure consistency" without concrete guidance
- Wrong level: tries to fix the prompt text itself instead of adding an \
instruction that controls the LLM's output behavior

Respond in EXACTLY this format:

VERDICT: GOOD | REVISE
REASON: one sentence explaining your verdict
(if REVISE, also include the revised edit in the same structured format:)
ACTION: REPLACE | APPEND | PREPEND
FIND: (REPLACE only) exact text from the prompt — no quotes
REPLACE_WITH: (REPLACE only) replacement text — no quotes
APPEND_TEXT: (APPEND only) text to add at the end
PREPEND_TEXT: (PREPEND only) text to add at the beginning"""


def _build_review_prompt(
    original_prompt: str,
    proposed_edit: str,
    eval_results: list[EvalResult] | None = None,
) -> str:
    """Build a prompt for the self-review step."""
    numbered = _add_line_numbers(original_prompt)

    prompt = f"""CURRENT PROMPT:
{numbered}

PROPOSED EDIT:
{proposed_edit}"""

    if eval_results:
        failing = [r for r in eval_results if not r.passed]
        if failing:
            fail_text = []
            for r in failing:
                line = f"- FAIL {r.eval_id} (score={r.score:.2f})"
                if r.details:
                    line += f" — {r.details}"
                fail_text.append(line)
            prompt += f"\n\nFAILING EVALS THIS EDIT SHOULD FIX:\n{chr(10).join(fail_text)}"

    prompt += """

Review this edit. Is it specific enough? Is it general rather than targeting \
one narrow example? Does it duplicate existing instructions? Would it actually \
fix the failing evals?"""

    return prompt


def _review_edit(
    original: str,
    raw_response: str,
    llm_client: LLMClient,
    eval_results: list[EvalResult] | None = None,
) -> str:
    """Review a proposed edit and optionally revise it.

    Returns the (possibly revised) raw edit response.
    """
    review_prompt = _build_review_prompt(original, raw_response, eval_results)
    review_response = llm_client.generate(
        review_prompt,
        system=_REVIEW_SYSTEM,
        temperature=0.3,
    )

    # Log the review
    from prompterator.runners.llm import _debug_enabled, _debug_dir
    if _debug_enabled and _debug_dir is not None:
        logs = sorted(_debug_dir.glob("debug-*.log"))
        if logs:
            with open(logs[-1], "a") as _f:
                _f.write(f"\n--- REVIEW ---\n{review_response}\n")

    # Parse verdict
    verdict_match = re.search(r'^VERDICT:\s*(GOOD|REVISE)', review_response, re.MULTILINE | re.IGNORECASE)
    if not verdict_match:
        return raw_response  # Can't parse, use original

    verdict = verdict_match.group(1).upper()
    if verdict == "GOOD":
        return raw_response

    # Extract the revised edit from the review response
    # Look for ACTION: in the review response (after VERDICT/REASON)
    action_match = re.search(r'^ACTION:', review_response, re.MULTILINE)
    if action_match:
        # Extract rationale from REASON field, then build a new edit response
        reason = _parse_field(review_response, "REASON:") or "Revised by reviewer"
        revised_body = review_response[action_match.start():]
        return f"RATIONALE: {reason}\n{revised_body}"

    return raw_response


# ---------------------------------------------------------------------------
# Legacy parser (kept for compatibility)
# ---------------------------------------------------------------------------

def _parse_improvement_with_rationale(raw: str) -> tuple[str, str]:
    """Parse LLM output with === or --- separator into (prompt, rationale)."""
    raw = raw.strip()
    for sep in (r'={3,}', r'-{3,}'):
        match = re.search(r'\n\s*' + sep + r'\s*\n', raw)
        if not match:
            match = re.search(r'\s+' + sep + r'\s*\n', raw)
        if match:
            rationale = raw[:match.start()].strip()
            prompt_text = raw[match.end():]
            if prompt_text.strip():
                return prompt_text.strip(), rationale or "No rationale provided"
    return raw.strip(), "No rationale provided"


def _clean_llm_output(text: str) -> str:
    """Strip wrapping artifacts from LLM output."""
    return text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_improved_prompt(
    prompt_path: Path,
    issue_file: IssueFile,
    llm_client: LLMClient,
    eval_results: list[EvalResult] | None = None,
    iteration: int | None = None,
) -> str:
    """Generate an improved version of a prompt.

    Returns:
        Improved prompt text.
    """
    with open(prompt_path) as f:
        original_prompt = f.read()

    improved, _rationale, _raw, _action = generate_improved_prompt_with_rationale(
        original_prompt, issue_file, llm_client, eval_results, iteration,
    )
    return improved


def generate_improved_prompt_with_rationale(
    prompt_text: str,
    issue_file: IssueFile,
    llm_client: LLMClient,
    eval_results: list[EvalResult] | None = None,
    iteration: int | None = None,
    directive: str | None = None,
    edit_history: list[dict] | None = None,
    stall_count: int = 0,
) -> tuple[str, str, str, str]:
    """Generate an improved prompt via structured edit.

    Returns:
        Tuple of (improved_prompt, rationale, raw_llm_output, action).
    """
    edit_prompt = _build_edit_prompt(
        prompt_text, issue_file, eval_results, iteration,
        edit_history=edit_history,
        stall_count=stall_count,
    )

    system = _EDIT_SYSTEM
    if directive:
        system += (
            f"\n\nThe user has given you a specific directive. Focus on this "
            f"above all else:\n{directive}"
        )

    raw_response = llm_client.generate(
        edit_prompt,
        system=system,
        temperature=0.4,
    )

    # Self-review: let the LLM critique and optionally revise its own edit
    raw_response = _review_edit(
        prompt_text, raw_response, llm_client, eval_results,
    )

    improved, rationale, success, action = _apply_edit(prompt_text, raw_response)

    # Debug: log parse results
    from prompterator.runners.llm import _debug_enabled, _debug_dir
    if _debug_enabled and _debug_dir is not None:
        logs = sorted(_debug_dir.glob("debug-*.log"))
        if logs:
            with open(logs[-1], "a") as _f:
                _f.write(f"\n--- APPLY ---\n")
                _f.write(f"before:  {len(prompt_text.encode())} bytes\n")
                _f.write(f"after:   {len(improved.encode())} bytes\n")
                _f.write(f"changed: {prompt_text != improved}\n")
                _f.write(f"success: {success}\n")
                _f.write(f"rationale: {rationale}\n")

    return improved, rationale, raw_response, action


def generate_multiple_variants(
    prompt_path: Path,
    issue_file: IssueFile,
    llm_client: LLMClient,
    num_variants: int = 3,
) -> list[str]:
    """Generate multiple improved variants of a prompt.

    Returns:
        List of improved prompt texts.
    """
    variants = []
    for i in range(num_variants):
        improved = generate_improved_prompt(prompt_path, issue_file, llm_client)
        variants.append(improved)
    return variants


def save_improved_prompt(content: str, path: Path) -> None:
    """Save an improved prompt to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
