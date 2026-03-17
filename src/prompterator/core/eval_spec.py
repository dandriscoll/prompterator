"""Eval specification generation from issues."""

from __future__ import annotations

import json
from pathlib import Path

from prompterator.models.eval import Eval, EvalFile, EvalRubric
from prompterator.models.issue import IssueFile
from prompterator.runners.llm import LLMClient


_CRITERIA_SYSTEM = (
    "You convert an issue description into a single evaluation pass-criterion. "
    "The issue describes a PROBLEM. You produce ONE criterion that checks for "
    "the ABSENCE of that problem — it should PASS when the problem is gone.\n\n"
    "You may also see EVIDENCE — actual feedback quotes describing what went "
    "wrong. Use these to make your criterion SPECIFIC. Reference the concrete "
    "words, patterns, or structures from the evidence, not generic categories.\n\n"
    "Frame the criterion as what the output must NOT contain or do. "
    "Do not frame it as what the output SHOULD contain — that overspecifies "
    "and tests for things beyond the original issue.\n\n"
    "Examples:\n"
    "  Issue: Output adds conversational preamble before the list\n"
    "  Evidence: starts with 'Here's your updated to-do list'\n"
    "  Criterion: Output does not begin with a conversational sentence "
    "like 'Here's your...' before the first list item\n\n"
    "  Issue: Output replaces checkboxes with priority-grouped sections\n"
    "  Evidence: replaced [ ] items with priority headings\n"
    "  Criterion: Output does not replace [ ] checkbox items with "
    "priority-grouped sections or headings\n\n"
    "  Issue: Output removes profanity and emotional language\n"
    "  Evidence: sanitized swear words, replaced 'damn' with 'darn'\n"
    "  Criterion: Output does not sanitize, censor, or replace profanity "
    "or emotional language from the original text\n\n"
    "Stay focused on the specific issue. Do not add criteria for problems "
    "that were not raised. Be specific — cite the patterns from evidence.\n\n"
    "Output ONLY a JSON array containing exactly one string. "
    "Do not wrap in markdown fences."
)


# Mapping of issue categories to evaluation criteria
CATEGORY_CRITERIA = {
    "clarity": [
        "Instructions are unambiguous",
        "Language is clear and precise",
        "Examples are provided where helpful",
    ],
    "completeness": [
        "All required information is present",
        "Edge cases are addressed",
        "No missing instructions",
    ],
    "accuracy": [
        "Information is factually correct",
        "Technical details are accurate",
        "Examples are correct",
    ],
    "tone": [
        "Tone is appropriate for audience",
        "Language is professional",
        "Consistent voice throughout",
    ],
    "format": [
        "Structure is logical",
        "Formatting is consistent",
        "Sections are well-organized",
    ],
}


def _generate_eval_id(prompt_ref: str, category: str, index: int) -> str:
    """Generate a unique eval ID."""
    base = Path(prompt_ref).stem.split(".")[0]
    return f"eval-{base}-{category}-{index:02d}"


_MAX_CRITERIA_PER_ISSUE = 5


def _criteria_from_issue(issue, llm_client: LLMClient | None = None, directive: str | None = None) -> list[str]:
    """Generate eval pass-criteria from an issue.

    Uses the LLM to invert the issue (problem description) into
    desired-state criteria that pass when the problem is fixed.
    Falls back to evidence-derived criteria when no LLM is available.
    """
    if llm_client is not None:
        try:
            # Build input with issue summary and concrete evidence
            parts = [f"ISSUE: {issue.summary}"]
            if issue.evidence:
                evidence_texts = [ev.feedback for ev in issue.evidence[:5]]
                parts.append("EVIDENCE:\n" + "\n".join(f"- {t}" for t in evidence_texts))
            user_input = "\n\n".join(parts)
            if directive:
                user_input = (
                    f"IMPORTANT — follow this guidance when writing the criterion:\n"
                    f"{directive}\n\n{user_input}"
                )
            raw = llm_client.generate(user_input, system=_CRITERIA_SYSTEM)
            criteria = json.loads(raw)
            if isinstance(criteria, list) and criteria:
                return [str(c) for c in criteria[:1]]
        except (json.JSONDecodeError, Exception):
            # Fall through to heuristic
            pass

    # Fallback: derive from raw evidence
    details: list[str] = []
    for ev in issue.evidence:
        if ev.feedback:
            details.append(ev.feedback)

    unique = _deduplicate_details(details)
    return [f"Prompt addresses: {d}" for d in unique[:_MAX_CRITERIA_PER_ISSUE]]


def _deduplicate_details(details: list[str]) -> list[str]:
    """Collapse semantically overlapping feedback notes into unique themes.

    Uses a word-overlap heuristic with synonym expansion: if a new detail
    shares more than 40% of its significant words (after synonym
    normalisation) with an already-kept detail, it is treated as a
    duplicate.  The first (longest / most specific) detail wins.
    """
    if not details:
        return []

    # Sort longest-first so the most descriptive note is kept.
    details = sorted(details, key=len, reverse=True)

    # Groups of words that should be treated as identical for overlap.
    _SYNONYM_GROUPS: list[set[str]] = [
        {"preamble", "intro", "introduction", "conversational", "chatty",
         "opening", "framing"},
        {"sign-off", "signoff", "offer", "offering", "offers", "chatbot"},
        {"dashes", "bullets", "bullet", "characters", "checkboxes",
         "checkbox", "formatting"},
        {"priority", "priorities", "groups", "grouped", "sections",
         "tiers", "taxonomy", "headings"},
        {"reorganize", "reorganized", "restructured", "rewrite",
         "re-sorted", "flattened", "stripped"},
        {"structure", "structural", "structured", "format"},
    ]

    # Build a quick lookup: word → canonical representative
    _synonyms: dict[str, str] = {}
    for group in _SYNONYM_GROUPS:
        canonical = sorted(group)[0]  # deterministic pick
        for word in group:
            _synonyms[word] = canonical

    stop = {
        "a", "an", "the", "and", "or", "but", "is", "was", "are", "were",
        "in", "on", "at", "to", "for", "of", "with", "from", "as", "not",
        "that", "this", "it", "its", "into", "than", "which", "every",
        "other", "same", "still", "also", "model", "output", "input",
        "would", "should", "could", "just", "already", "show", "before",
        "after", "actually", "really", "very", "new", "own", "original",
        "list", "content", "rendered", "lines", "top", "end", "bottom",
    }

    def _significant_words(text: str) -> set[str]:
        words = set()
        for w in text.lower().split():
            w = w.strip(".,;:!?\"'()-")
            if len(w) > 2 and w not in stop:
                words.add(_synonyms.get(w, w))
        return words

    kept: list[tuple[str, set[str]]] = []
    for detail in details:
        words = _significant_words(detail)
        if not words:
            continue

        is_duplicate = False
        for _kept_detail, kept_words in kept:
            overlap = words & kept_words
            if len(overlap) / len(words) > 0.4:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append((detail, words))

    return [text for text, _words in kept]


_RECONCILE_SYSTEM = (
    "You are an eval maintenance assistant. The issues that evals were based on "
    "have been reorganized — issues may have been split, merged, renamed, or "
    "renumbered. You will see the NEW issues and the EXISTING eval criteria. "
    "Your job is to map each existing eval's criteria to the new issue it best "
    "matches, so that hand-tuned criteria are preserved.\n\n"
    "For each existing eval, decide:\n"
    "- KEEP: the eval's criteria still match a new issue (give the new issue ID)\n"
    "- DROP: the eval tests for a problem that no longer exists in any issue\n\n"
    "Output ONLY a JSON array of objects, one per existing eval:\n"
    '  {"eval_id": "...", "action": "keep", "new_issue_id": "..."}\n'
    '  {"eval_id": "...", "action": "drop"}\n\n'
    "Rules:\n"
    "- Each new issue should be matched by at most one existing eval.\n"
    "- Match by meaning, not by ID — an eval about 'preamble' maps to whichever "
    "new issue describes the preamble problem, regardless of its ID.\n"
    "- If two existing evals match the same new issue, keep the one with more "
    "specific criteria and drop the other.\n"
    "- Do not wrap in markdown fences."
)


def _reconcile_evals_with_issues(
    existing_evals: list[Eval],
    issue_file: IssueFile,
    llm_client: LLMClient,
) -> dict[str, str]:
    """Map existing evals to new issue IDs after issues were reorganized.

    Returns:
        Dict mapping existing eval_id → new issue_id for evals to keep.
        Evals not in the dict should be dropped.
    """
    # Build the prompt showing new issues and existing evals
    issue_lines = []
    for issue in issue_file.issues:
        issue_lines.append(f"- {issue.id} ({issue.category}): {issue.summary}")

    eval_lines = []
    for ev in existing_evals:
        criteria_str = ""
        if ev.rubric and ev.rubric.criteria:
            criteria_str = "; ".join(ev.rubric.criteria)
        desc = ev.description or ""
        eval_lines.append(
            f"- {ev.id} [issue_ref={ev.issue_ref}]: {desc}"
            + (f"\n    Criteria: {criteria_str}" if criteria_str else "")
        )

    user_prompt = (
        f"NEW ISSUES:\n{chr(10).join(issue_lines)}\n\n"
        f"EXISTING EVALS:\n{chr(10).join(eval_lines)}\n\n"
        "Map each existing eval to the new issue it best matches, or drop it."
    )

    raw = llm_client.generate(user_prompt, system=_RECONCILE_SYSTEM)

    # Parse response
    mapping: dict[str, str] = {}
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                entries = json.loads(raw[start:end])
            except json.JSONDecodeError:
                return {}
        else:
            return {}

    new_issue_ids = {issue.id for issue in issue_file.issues}
    used_issues: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        eval_id = entry.get("eval_id", "")
        action = entry.get("action", "").lower()
        new_issue_id = entry.get("new_issue_id", "")
        if action == "keep" and new_issue_id in new_issue_ids and new_issue_id not in used_issues:
            mapping[eval_id] = new_issue_id
            used_issues.add(new_issue_id)

    return mapping


def generate_evals_from_issues(
    issue_file: IssueFile,
    llm_client: LLMClient | None = None,
    existing_evals: list[Eval] | None = None,
    directive: str | None = None,
) -> EvalFile:
    """Generate evaluation specifications from issues.

    When existing_evals is provided, evals whose issue_ref matches an
    existing eval are kept as-is (preserving hand-tuned criteria). New
    evals are generated only for issues not already covered.

    Args:
        issue_file: IssueFile containing issues to address.
        llm_client: Optional LLM client for generating inverted pass-criteria.
        existing_evals: Previously generated evals to merge with.

    Returns:
        EvalFile with merged evaluations.
    """
    # Index existing evals by issue_ref
    existing_by_issue: dict[str, Eval] = {}
    if existing_evals:
        new_issue_ids = {issue.id for issue in issue_file.issues}

        # Check if issue IDs have changed (reorganized)
        orphaned = [ev for ev in existing_evals if ev.issue_ref and ev.issue_ref not in new_issue_ids]

        if orphaned and llm_client is not None:
            # Issues were reorganized — ask LLM to reconcile
            mapping = _reconcile_evals_with_issues(existing_evals, issue_file, llm_client)
            for ev in existing_evals:
                new_id = mapping.get(ev.id)
                if new_id:
                    # Re-point the eval to its new issue and preserve criteria
                    existing_by_issue[new_id] = ev
        else:
            for ev in existing_evals:
                if ev.issue_ref:
                    existing_by_issue[ev.issue_ref] = ev

    evals = []
    eval_index = 1

    for issue in issue_file.issues:
        # Keep existing eval if it covers this issue
        if issue.id in existing_by_issue:
            kept = existing_by_issue[issue.id]
            # Update the issue_ref to the new ID
            kept_dict = kept.model_dump()
            kept_dict["issue_ref"] = issue.id
            evals.append(Eval(**kept_dict))
            eval_index += 1
            continue
        category = issue.category.lower()

        # Use LLM to invert issue descriptions into pass-criteria.
        # Falls back to generic category criteria when no LLM or no details.
        specific_criteria = _criteria_from_issue(issue, llm_client, directive=directive)
        if specific_criteria:
            criteria = specific_criteria
        else:
            criteria = CATEGORY_CRITERIA.get(category, [f"Addresses {category} concerns"])

        # High severity issues require all criteria
        # Medium/low can pass with any
        scoring = "all_required" if issue.severity == "high" else "any_required"

        eval_spec = Eval(
            id=_generate_eval_id(issue_file.prompt_ref, category, eval_index),
            type="rubric",
            issue_ref=issue.id,
            description=f"Verify {category} improvements: {issue.summary}",
            rubric=EvalRubric(
                criteria=criteria,
                scoring=scoring,
            ),
        )
        evals.append(eval_spec)
        eval_index += 1

    return EvalFile(
        version="1.0",
        prompt_ref=issue_file.prompt_ref,
        evals=evals,
    )


def load_eval_file(path: Path) -> EvalFile:
    """Load an eval file from disk."""
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)

    evals = []
    for eval_data in data.get("evals", []):
        rubric = None
        if "rubric" in eval_data:
            rubric = EvalRubric(
                criteria=eval_data["rubric"]["criteria"],
                scoring=eval_data["rubric"].get("scoring", "all_required"),
                weights=eval_data["rubric"].get("weights"),
            )

        evals.append(
            Eval(
                id=eval_data["id"],
                type=eval_data.get("type", "rubric"),
                issue_ref=eval_data.get("issue_ref"),
                description=eval_data.get("description"),
                rubric=rubric,
                assertion=eval_data.get("assertion"),
            )
        )

    return EvalFile(
        version=data.get("version", "1.0"),
        prompt_ref=data["prompt_ref"],
        evals=evals,
    )


def save_eval_file(eval_file: EvalFile, path: Path) -> None:
    """Save an eval file to disk."""
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(eval_file.to_yaml_dict(), f, default_flow_style=False, sort_keys=False)
