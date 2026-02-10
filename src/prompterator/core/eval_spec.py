"""Eval specification generation from issues."""

from pathlib import Path

from prompterator.models.eval import Eval, EvalFile, EvalRubric
from prompterator.models.issue import IssueFile


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


def _criteria_from_evidence(issue) -> list[str]:
    """Extract specific eval criteria from issue evidence details.

    Turns feedback like 'note=opens with conversational paragraph' into
    a testable criterion like 'Prompt instructs against opening with
    conversational paragraph'.
    """
    criteria = []
    seen = set()

    for ev in issue.evidence:
        feedback = ev.feedback
        # Extract the note/detail text
        detail = None
        for marker in ("; note=", "; needs=", "; detail="):
            if marker in feedback:
                detail = feedback.split(marker, 1)[1]
                break

        if detail and detail not in seen:
            seen.add(detail)
            criteria.append(f"Prompt addresses: {detail}")

    return criteria


def generate_evals_from_issues(issue_file: IssueFile) -> EvalFile:
    """Generate evaluation specifications from issues.

    Args:
        issue_file: IssueFile containing issues to address.

    Returns:
        EvalFile with generated evaluations.
    """
    evals = []
    eval_index = 1

    for issue in issue_file.issues:
        category = issue.category.lower()

        # Prefer feedback-specific criteria derived from evidence details.
        # Fall back to generic category criteria when no details are available.
        specific_criteria = _criteria_from_evidence(issue)
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
