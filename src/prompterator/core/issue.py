"""Issue consolidation logic - aggregate feedback into issues."""

from collections import defaultdict
from pathlib import Path

from prompterator.models.feedback import Feedback
from prompterator.models.issue import Issue, IssueEvidence, IssueFile


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


def _generate_issue_id(prompt_ref: str, category: str, index: int) -> str:
    """Generate a unique issue ID."""
    # Extract just the base name without extension
    base = Path(prompt_ref).stem.split(".")[0]
    return f"issue-{base}-{category}-{index:02d}"


def consolidate_feedback(
    feedback_list: list[Feedback],
    prompt_ref: str,
    categories: list[str],
    min_occurrences: int = 1,
) -> IssueFile:
    """Consolidate multiple feedback entries into issues.

    Args:
        feedback_list: List of parsed feedback objects.
        prompt_ref: Reference to the prompt file.
        categories: Valid category names.
        min_occurrences: Minimum occurrences to create an issue.

    Returns:
        IssueFile with consolidated issues.
    """
    # Group evidence by category
    category_evidence: dict[str, list[IssueEvidence]] = defaultdict(list)
    category_values: dict[str, list[str]] = defaultdict(list)

    for feedback in feedback_list:
        for entry in feedback.entries:
            # Normalize category name
            cat = entry.category.lower()
            if cat not in [c.lower() for c in categories]:
                continue

            evidence = IssueEvidence(
                source=feedback.source_file,
                feedback=f"{entry.category}={entry.value}"
                + (f"; {entry.details}" if entry.details else ""),
            )
            category_evidence[cat].append(evidence)
            category_values[cat].append(entry.value)

    # Create issues for categories meeting threshold
    issues = []
    issue_index = 1

    for cat in categories:
        cat_lower = cat.lower()
        evidence_list = category_evidence.get(cat_lower, [])

        if len(evidence_list) < min_occurrences:
            continue

        # Generate summary from values
        values = category_values.get(cat_lower, [])
        unique_values = list(set(values))
        if len(unique_values) == 1:
            summary = f"{cat.capitalize()} issue: {unique_values[0]}"
        else:
            summary = f"{cat.capitalize()} issues noted: {', '.join(unique_values[:3])}"
            if len(unique_values) > 3:
                summary += f" (+{len(unique_values) - 3} more)"

        issue = Issue(
            id=_generate_issue_id(prompt_ref, cat_lower, issue_index),
            category=cat_lower,
            severity=_determine_severity(len(evidence_list), len(feedback_list)),
            summary=summary,
            evidence=evidence_list,
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
