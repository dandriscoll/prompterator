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

_IDEATION_SYSTEM = """\
You are a prompt strategist. You analyze PROMPT TEMPLATES — text sent to an \
LLM to control its behavior — and propose improvements.

ISSUES describe problems with the LLM's output. EVALS check whether the \
prompt addresses each issue. Your job is to propose ONE specific change \
that will fix a failing eval without regressing passing ones.

Think creatively about what instruction would fix the issue. Consider:
- What specific LLM behavior is wrong?
- What concrete rule would prevent it?
- Is there an existing instruction to strengthen, or do we need a new one?

Respond in EXACTLY this format:

IDEA: 2-3 sentences describing what to change and why. Be specific about \
the exact wording of the instruction to add or modify.
TARGET_EVAL: which failing eval this addresses
APPROACH: APPEND (add new rule) | REPLACE (modify existing text) | PREPEND

Rules:
- Propose concrete instructions, not vague improvements.
- GOOD: "Add a rule stating: do not add conversational preamble or \
introductory text before the to-do list"
- BAD: "Improve the consistency instructions"
- Do NOT propose deleting or shortening existing instructions.
- Each idea should address exactly one failing eval."""


_EXECUTION_SYSTEM = """\
You are a precise prompt editor. You will receive an edit plan and a prompt \
template. Your job is to produce the EXACT structured edit to implement \
the plan.

You MUST respond in this exact format — no other text:

ACTION: REPLACE | APPEND | PREPEND
FIND: (REPLACE only) exact text copied from the prompt — no quotes
REPLACE_WITH: (REPLACE only) replacement text — no quotes
APPEND_TEXT: (APPEND only) text to add at the end
PREPEND_TEXT: (PREPEND only) text to add at the beginning

Rules:
- FIND must be an EXACT substring copied character-for-character from the prompt.
- Do NOT wrap any field values in quotation marks.
- Do NOT add your own commentary or rationale — just the structured edit.
- Implement the plan precisely as described."""


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
        overall = sum(r.score for r in eval_results) / len(eval_results) if eval_results else 0
        prompt += f"\n\nOVERALL SCORE: {overall:.1f}/10 (target: 9.0/10)"
        if failing:
            fail_text = []
            for r in failing:
                line = f"- FAIL {r.eval_id} ({r.score:.1f}/10)"
                if r.details:
                    line += f" — {r.details}"
                fail_text.append(line)
            prompt += f"\n\nFAILING EVALS:\n{chr(10).join(fail_text)}"
        if passing:
            pass_text = [f"- PASS {r.eval_id} ({r.score:.1f}/10)" for r in passing]
            prompt += f"\n\nPASSING EVALS (do not regress):\n{chr(10).join(pass_text)}"
        if not failing:
            prompt += "\n\nAll evals are passing. Focus on strengthening the weakest eval."

    if edit_history:
        history_lines = []
        for entry in edit_history[-8:]:  # Last 8 attempts
            status = "ACCEPTED" if entry.get("accepted") else "REJECTED"
            action = entry.get("action", "?")
            line = f"- [{status}] ({action}) {entry['rationale']}"
            history_lines.append(line)
        prompt += f"\n\nPREVIOUS EDIT ATTEMPTS (do NOT repeat these):\n"
        prompt += chr(10).join(history_lines)

    # Detect plateau and repetition — escalating tiers
    warnings = []
    n_history = len(edit_history) if edit_history else 0

    # Detect idea repetition
    repeat_detected = False
    if edit_history and len(edit_history) >= 3:
        recent_rationales = [e.get("rationale", "") for e in edit_history[-5:]]
        if len(recent_rationales) >= 3:
            first_words = set(recent_rationales[0].lower().split())
            similar_count = sum(
                1 for r in recent_rationales[1:]
                if first_words and set(r.lower().split())
                and len(first_words & set(r.lower().split())) / min(len(first_words), len(set(r.lower().split()))) > 0.6
            )
            repeat_detected = similar_count >= 2

    # Escalation tiers based on combined signals
    plateau_depth = max(stall_count, n_history // 2 if repeat_detected else 0)

    # Tier 3: Force a specific prescribed strategy (rotate through list)
    _FORCED_STRATEGIES = [
        (
            "REWRITE THE EXAMPLE. The example in the prompt may be teaching "
            "the LLM the wrong behavior. Use REPLACE to change the example "
            "so it demonstrates the correct output for the failing eval."
        ),
        (
            "ADD A NEGATIVE EXAMPLE. Show the LLM what NOT to produce. "
            "APPEND a line like 'BAD example: ...' or 'Do NOT produce output "
            "like: ...' that illustrates the exact failure the eval catches."
        ),
        (
            "SIMPLIFY AND COMBINE. The prompt may have too many weak rules. "
            "Use REPLACE to merge 2-3 existing rules into ONE strong, "
            "specific instruction that directly addresses the failing eval."
        ),
        (
            "REMOVE A CONFUSING RULE. An existing instruction may conflict "
            "with what you want. Use REPLACE to shorten or remove a rule "
            "that might be causing the LLM to produce the wrong output."
        ),
        (
            "CHANGE THE FRAMING. Instead of adding rules, change HOW the "
            "prompt frames the task. Use REPLACE to reword the main "
            "instruction so the LLM naturally avoids the failing behavior."
        ),
        (
            "ADD A CHECKLIST. APPEND a self-check instruction like "
            "'Before outputting, verify: [specific check for failing eval]. "
            "If the check fails, revise your output.'"
        ),
        (
            "REORDER FOR PROMINENCE. LLMs pay most attention to the start "
            "and end of a prompt. Use REPLACE to move the most important "
            "rule (the one the failing eval checks) to the very beginning "
            "of the rules section, or repeat it at the end for emphasis."
        ),
        (
            "BAN SPECIFIC WORDS. Look at the failing eval details to find "
            "exact words or phrases the LLM keeps producing incorrectly. "
            "APPEND a rule that explicitly bans those words, e.g. "
            "'Do NOT reorganize items into priority sections or categories.'"
        ),
    ]

    if plateau_depth >= 8 or (repeat_detected and n_history >= 12):
        # Tier 3: Assign a specific strategy — rotate through the list
        strategy_idx = n_history % len(_FORCED_STRATEGIES)
        strategy = _FORCED_STRATEGIES[strategy_idx]
        warnings.append(
            f"MANDATORY STRATEGY: You have been assigned this specific "
            f"approach. You MUST follow it exactly:\n\n{strategy}\n\n"
            f"Do NOT propose anything else. Do NOT repeat a previous idea."
        )
    elif plateau_depth >= 5 or (repeat_detected and n_history >= 6):
        # Tier 2: Strong redirection with suggestions
        warnings.append(
            "CRITICAL: Your recent ideas are too similar and the score is "
            "not improving. You MUST try something fundamentally different:\n"
            "- Target a DIFFERENT failing eval than your last attempt\n"
            "- Address the root cause instead of the symptom\n"
            "- Use REPLACE to rewrite an existing instruction rather than "
            "adding new ones\n"
            "- Consider whether the example in the prompt contradicts your rule\n"
            "- Think about what the LLM is actually doing wrong and WHY"
        )
    elif plateau_depth >= 3 or repeat_detected:
        # Tier 1: Gentle nudge
        warnings.append(
            "The score has not improved recently. Try a different approach — "
            "target a different failing eval, use REPLACE instead of APPEND, "
            "or phrase the instruction in a fundamentally new way."
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


def _find_similar_lines(text: str, threshold: float = 0.8) -> list[list[str]]:
    """Find groups of lines that are similar to each other by word overlap.

    Returns groups of 2+ similar lines (each group is a list of line texts).
    """
    lines = [ln.strip() for ln in text.split('\n') if len(ln.strip()) > 30]
    groups: list[list[str]] = []
    used: set[int] = set()

    for i, a in enumerate(lines):
        if i in used:
            continue
        a_words = set(a.lower().split())
        group = [a]
        for j, b in enumerate(lines):
            if j <= i or j in used:
                continue
            b_words = set(b.lower().split())
            overlap = len(a_words & b_words) / min(len(a_words), len(b_words))
            if overlap >= threshold:
                group.append(b)
                used.add(j)
        if len(group) > 1:
            used.add(i)
            groups.append(group)

    return groups


_CONSOLIDATE_SYSTEM = """\
You are a prompt editor. The prompt below contains redundant or near-duplicate \
instructions. Your job is to consolidate them into a single clear rule.

You MUST respond in this format:

RATIONALE: one sentence explaining what you consolidated
ACTION: REPLACE
FIND: the first redundant line (copied exactly from the prompt — no quotes)
REPLACE_WITH: a single consolidated instruction that covers all the redundant lines

Rules:
- FIND must be an exact substring from the prompt.
- REPLACE_WITH should be ONE clear rule that replaces ALL the redundant lines.
- Do NOT wrap values in quotation marks.
- After this edit, the other redundant lines will be removed automatically."""


def consolidate_redundant_lines(
    prompt_text: str,
    llm_client: LLMClient,
) -> tuple[str, str, str]:
    """Detect and consolidate redundant lines in a prompt.

    Returns:
        Tuple of (consolidated_prompt, rationale, raw_response).
        If no redundancy found, returns (prompt_text, "", "").
    """
    # Find similar lines
    groups = _find_similar_lines(prompt_text)
    dupes = _find_duplicate_lines(prompt_text)

    # Merge both detection methods
    all_redundant: list[list[str]] = []
    seen_lines: set[str] = set()
    for group in groups:
        key = frozenset(ln.lower() for ln in group)
        if key not in seen_lines:
            all_redundant.append(group)
            seen_lines.add(key)
    for line_text, count in dupes:
        if not any(line_text in g for g in all_redundant):
            all_redundant.append([line_text] * count)

    if not all_redundant:
        return prompt_text, "", ""

    # Show the LLM the prompt and the redundant groups
    numbered = _add_line_numbers(prompt_text)
    redundant_text = []
    for i, group in enumerate(all_redundant[:3], 1):
        redundant_text.append(f"Group {i} ({len(group)} similar lines):")
        for ln in group:
            redundant_text.append(f"  - {ln[:100]}{'...' if len(ln) > 100 else ''}")

    user_prompt = f"""PROMPT TO EDIT:
{numbered}

REDUNDANT LINES TO CONSOLIDATE:
{chr(10).join(redundant_text)}

Produce a REPLACE edit that replaces the first line of the first group with \
a single consolidated instruction, then I will remove the other duplicates."""

    raw_response = llm_client.generate(
        user_prompt,
        system=_CONSOLIDATE_SYSTEM,
        temperature=0.3,
    )

    improved, rationale, success, action = _apply_edit(prompt_text, raw_response)

    if success:
        # Remove the other redundant lines (all except the one we replaced)
        for group in all_redundant:
            for ln in group[1:]:
                # Remove the line if it still exists
                lines = improved.split('\n')
                cleaned = []
                removed = False
                for l in lines:
                    if not removed and l.strip() == ln.strip():
                        removed = True
                        continue
                    cleaned.append(l)
                improved = '\n'.join(cleaned)

        # Clean up any triple+ blank lines
        while '\n\n\n' in improved:
            improved = improved.replace('\n\n\n', '\n\n')

    return improved, rationale, raw_response


_SIMPLIFY_SYSTEM = """\
You are a prompt editor. Your job is to simplify a prompt template by making \
it shorter and clearer WITHOUT changing its meaning or losing any rules.

You will make ONE simplification at a time. Choose the highest-impact change:
- Merge rules that say the same thing in different words
- Remove redundant qualifiers or filler phrases
- Combine multiple bullet points into one concise rule
- Remove instructions that are implied by other instructions

IMPORTANT: Do NOT remove any rule or constraint that is not covered by \
another rule. Every behavior the prompt currently controls must still be \
controlled after your edit.

Respond in EXACTLY this format:

RATIONALE: one sentence explaining what you simplified
ACTION: REPLACE
FIND: exact text from the prompt — no quotes
REPLACE_WITH: simplified version — no quotes

If the prompt cannot be simplified further, respond with exactly:
DONE"""


def simplify_prompt(
    prompt_text: str,
    llm_client: LLMClient,
) -> tuple[str, str, bool]:
    """Apply one simplification step to a prompt.

    Returns:
        Tuple of (simplified_prompt, rationale, changed).
        If no simplification possible, returns (prompt_text, "", False).
    """
    numbered = _add_line_numbers(prompt_text)

    user_prompt = f"""PROMPT TO SIMPLIFY:
{numbered}

Make ONE simplification. If nothing can be simplified, respond with DONE."""

    raw_response = llm_client.generate(
        user_prompt,
        system=_SIMPLIFY_SYSTEM,
        temperature=0.3,
    )

    if raw_response.strip() == "DONE":
        return prompt_text, "", False

    improved, rationale, success, action = _apply_edit(prompt_text, raw_response)
    return improved, rationale, success


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


def _apply_edit(original: str, response: str, *, force: bool = False) -> tuple[str, str, bool, str]:
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
            # Reject if too similar to existing content (unless forced via directive)
            if not force and _is_similar_to_existing(append_text, original):
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

def _execute_idea(
    original: str,
    idea_response: str,
    llm_client: LLMClient,
) -> str:
    """Turn an idea into a precise structured edit.

    The ideation step produces a high-level plan (IDEA/TARGET_EVAL/APPROACH).
    This step produces the exact structured edit (ACTION/FIND/REPLACE_WITH).

    Returns the raw structured edit response.
    """
    numbered = _add_line_numbers(original)

    exec_prompt = f"""PROMPT TO EDIT (with line numbers):
{numbered}

EDIT PLAN:
{idea_response}

Produce the exact structured edit to implement this plan. Copy text from \
the prompt precisely — character for character."""

    exec_response = llm_client.generate(
        exec_prompt,
        system=_EXECUTION_SYSTEM,
        temperature=0.2,
    )

    # Log the execution
    from prompterator.runners.llm import _debug_enabled, _debug_dir
    if _debug_enabled and _debug_dir is not None:
        logs = sorted(_debug_dir.glob("debug-*.log"))
        if logs:
            with open(logs[-1], "a") as _f:
                _f.write(f"\n--- EXECUTE ---\n{exec_response}\n")

    # Carry the rationale from the idea into the edit
    idea_text = _parse_field(idea_response, "IDEA:") or ""
    # Prepend RATIONALE if the executor didn't include one
    if not re.search(r'^RATIONALE:', exec_response, re.MULTILINE):
        exec_response = f"RATIONALE: {idea_text}\n{exec_response}"

    return exec_response


# ---------------------------------------------------------------------------
# Help request generation
# ---------------------------------------------------------------------------

_HELP_SYSTEM = """\
You are a prompt tuning assistant. The automated tuning loop has plateaued — \
it cannot improve the prompt further. Your job is to write a short, concrete \
request for human help.

Respond in this format:

STUCK ON: [which eval(s) are still failing]
TRIED: [brief summary of what edits were attempted]
PROBLEM: [why the edits aren't working — be specific]
SUGGESTION: [what a human could do to help — e.g. rewrite a section, \
add an example, change the eval criteria, or provide a directive]

Keep it concise — 1-2 sentences per field."""


def generate_help_request(
    prompt_text: str,
    issue_file: IssueFile,
    llm_client: LLMClient,
    eval_results: list[EvalResult] | None = None,
    edit_history: list[dict] | None = None,
) -> str:
    """Generate a help request when tuning plateaus.

    Returns:
        Human-readable help request string.
    """
    numbered = _add_line_numbers(prompt_text)

    parts = [f"CURRENT PROMPT:\n{numbered}"]

    if eval_results:
        failing = [r for r in eval_results if not r.passed]
        if failing:
            fail_lines = []
            for r in failing:
                line = f"- FAIL {r.eval_id} (score={r.score:.2f})"
                if r.details:
                    line += f" — {r.details}"
                fail_lines.append(line)
            parts.append(f"STILL FAILING:\n{chr(10).join(fail_lines)}")

    if edit_history:
        history_lines = []
        for entry in edit_history[-10:]:
            status = "ACCEPTED" if entry.get("accepted") else "REJECTED"
            action = entry.get("action", "?")
            history_lines.append(f"- [{status}] ({action}) {entry['rationale']}")
        parts.append(f"EDIT HISTORY:\n{chr(10).join(history_lines)}")

    user_prompt = "\n\n".join(parts)

    try:
        return llm_client.generate(
            user_prompt,
            system=_HELP_SYSTEM,
            temperature=0.3,
        )
    except Exception:
        return "Tuning plateaued but could not generate a help request."


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

    # Step 1: Ideation — creative, higher temperature
    # Temperature ramps up with plateau depth to force divergent thinking.
    n_attempts = len(edit_history) if edit_history else 0
    plateau_signal = max(stall_count, n_attempts // 2)
    ideation_temp = min(0.7 + plateau_signal * 0.06, 1.2)

    system = _IDEATION_SYSTEM
    if directive:
        system += (
            f"\n\nThe user has given you a specific directive. Focus on this "
            f"above all else:\n{directive}"
        )

    idea_response = llm_client.generate(
        edit_prompt,
        system=system,
        temperature=ideation_temp,
    )

    # Step 2: Execution — precise, low temperature
    raw_response = _execute_idea(
        prompt_text, idea_response, llm_client,
    )

    improved, rationale, success, action = _apply_edit(
        prompt_text, raw_response, force=bool(directive),
    )

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
