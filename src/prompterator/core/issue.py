"""Issue consolidation logic - aggregate feedback into issues via LLM clustering."""

import json
from pathlib import Path

from prompterator.models.feedback import Feedback
from prompterator.models.issue import Issue, IssueEvidence, IssueFile
from prompterator.runners.llm import LLMClient


_CLUSTER_SYSTEM = (
    "You are a feedback analyst. You receive a numbered list of free-form observations "
    "about LLM-generated outputs. Your job is to cluster the observations by the real "
    "underlying problem they describe. Prefer more clusters over fewer — only group "
    "observations that describe the SAME specific problem together. Two observations "
    "that point to different output problems should be separate clusters even if they "
    "are thematically related. Ignore positive observations (praise, approval, "
    "things that are fine). Do NOT create meta-issues about the feedback itself "
    '(e.g. "feedback is repetitive" or "reviewers agree") — every cluster must '
    "describe a problem with the output. Output ONLY a JSON array of clusters. "
    "Each cluster has:\n"
    '  "label": a short kebab-case tag (e.g. "chatty-preamble", "structural-rewrite"),\n'
    '  "summary": a SPECIFIC description of the problem that references concrete '
    "details from the observations — quote exact words, phrases, or patterns from "
    "the feedback. Do NOT write generic summaries like 'output doesn't follow "
    "instructions' or 'content is not preserved'. Instead cite what actually happened, "
    "e.g. 'Output begins with a conversational sentence like \"Here\\'s your updated "
    "list\" before the actual content' or 'Output replaces [ ] checkbox markers with "
    "bullet-pointed priority sections',\n"
    '  "evidence_indices": list of 0-based indices into the input observations.\n'
    "Do not wrap the JSON in markdown fences."
)


def _split_feedback_entry(text: str) -> list[str]:
    """Split a feedback entry that addresses multiple issues into separate items.

    Reviewers often combine observations with semicolons, e.g.
    "too chatty; incorrect grammar". This splits on semicolons (and
    similar separators) when each part looks like an independent observation.
    """
    # Only split on semicolons — the most common multi-issue separator
    if ";" not in text:
        return [text]

    parts = [p.strip() for p in text.split(";")]
    # Keep only non-trivial parts (at least 2 words)
    parts = [p for p in parts if len(p.split()) >= 2]

    if len(parts) < 2:
        # Splitting didn't produce multiple meaningful items — keep original
        return [text]

    return parts


def _determine_severity(occurrences: int, total_feedback_sources: int) -> str:
    """Determine issue severity based on occurrence frequency."""
    if total_feedback_sources == 0:
        return "medium"

    ratio = occurrences / total_feedback_sources

    if ratio >= 0.7:
        return "high"
    elif ratio >= 0.3:
        return "medium"
    else:
        return "low"


def _generate_issue_id(prompt_ref: str, index: int) -> str:
    """Generate a unique issue ID."""
    base = Path(prompt_ref).stem.split(".")[0]
    return f"issue-{base}-{index:02d}"


def consolidate_feedback(
    feedback_list: list[Feedback],
    prompt_ref: str,
    llm_client: LLMClient,
    min_occurrences: int = 1,
    existing_issues: list[Issue] | None = None,
    directive: str | None = None,
) -> IssueFile:
    """Consolidate multiple feedback entries into issues via LLM clustering.

    When existing_issues are provided, the LLM merges new feedback into
    the existing issue structure — updating, splitting, or creating new
    issues as needed rather than starting from scratch.

    Args:
        feedback_list: List of parsed feedback objects.
        prompt_ref: Reference to the prompt file.
        llm_client: LLM client for clustering.
        min_occurrences: Minimum evidence count to keep an issue.
        existing_issues: Previously identified issues to merge with.

    Returns:
        IssueFile with consolidated issues.
    """
    # 1. Collect all observations: existing evidence + new feedback
    observations: list[tuple[str, str]] = []

    # Include evidence from existing issues so the LLM can re-cluster everything
    if existing_issues:
        seen: set[tuple[str, str]] = set()
        for issue in existing_issues:
            for ev in issue.evidence:
                key = (ev.source, ev.feedback)
                if key not in seen:
                    seen.add(key)
                    observations.append(key)

    # Add new feedback, splitting multi-issue entries
    for feedback in feedback_list:
        for entry in feedback.entries:
            for part in _split_feedback_entry(entry.text):
                observations.append((feedback.source_file, part))

    if not observations:
        return IssueFile(version="1.0", prompt_ref=prompt_ref, issues=[])

    # 2. Build prompt listing every observation
    lines = []
    for idx, (source, text) in enumerate(observations):
        lines.append(f"[{idx}] ({source}) {text}")

    user_prompt = "\n".join(lines)

    # 3. Ask the LLM to cluster
    if directive:
        user_prompt = (
            f"IMPORTANT — follow this guidance when clustering:\n{directive}\n\n"
            + user_prompt
        )
    raw_response = llm_client.generate(user_prompt, system=_CLUSTER_SYSTEM)

    # Parse LLM response
    try:
        clusters = json.loads(raw_response)
    except json.JSONDecodeError:
        # Try to find JSON array in the response
        clusters = []
        start = raw_response.find("[")
        end = raw_response.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                clusters = json.loads(raw_response[start:end])
            except json.JSONDecodeError:
                pass

    # 4. Build issues from clusters
    total_sources = len({source for source, _ in observations})
    issues: list[Issue] = []
    issue_index = 1

    for cluster in clusters:
        label = cluster.get("label", f"issue-{issue_index}")
        summary = cluster.get("summary", "")
        evidence_indices = cluster.get("evidence_indices", [])

        # Build evidence list
        evidence: list[IssueEvidence] = []
        for idx in evidence_indices:
            if 0 <= idx < len(observations):
                source, text = observations[idx]
                evidence.append(IssueEvidence(source=source, feedback=text))

        # Apply min_occurrences threshold
        if len(evidence) < min_occurrences:
            continue

        # Compute severity from evidence count ratio
        severity = _determine_severity(len(evidence), total_sources)

        issue = Issue(
            id=_generate_issue_id(prompt_ref, issue_index),
            category=label,
            severity=severity,
            summary=summary,
            evidence=evidence,
        )
        issues.append(issue)
        issue_index += 1

    return IssueFile(
        version="1.0",
        prompt_ref=prompt_ref,
        issues=issues,
    )


def load_issue_file(path: Path) -> IssueFile:
    """Load an issue file from disk."""
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)

    issues = []
    for issue_data in data.get("issues", []):
        evidence = [
            IssueEvidence(source=e["source"], feedback=e["feedback"])
            for e in issue_data.get("evidence", [])
        ]
        issues.append(
            Issue(
                id=issue_data["id"],
                category=issue_data["category"],
                severity=issue_data["severity"],
                summary=issue_data["summary"],
                evidence=evidence,
            )
        )

    return IssueFile(
        version=data.get("version", "1.0"),
        prompt_ref=data["prompt_ref"],
        issues=issues,
    )


def save_issue_file(issue_file: IssueFile, path: Path) -> None:
    """Save an issue file to disk."""
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(issue_file.to_yaml_dict(), f, default_flow_style=False, sort_keys=False)
