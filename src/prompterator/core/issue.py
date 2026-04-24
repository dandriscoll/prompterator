"""Issue consolidation logic - aggregate feedback into issues via LLM clustering."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from prompterator.models.feedback import Feedback
from prompterator.models.issue import Issue, IssueEvidence, IssueFile
from prompterator.runners.llm import LLMClient


_CLUSTER_SYSTEM = (
    "You are a feedback analyst. You receive a numbered list of free-form observations "
    "about LLM-generated outputs and a THEMES list. Your job executes three passes "
    "in a single response:\n\n"
    "  PASS 1 — ANCHORS. For each observation, emit an anchor: a concrete instance "
    "of a problem with the output. Do not generalise here. Skip positive observations "
    "(praise, things that are fine). Skip meta-observations about the feedback itself.\n"
    "  PASS 2 — CLUSTERS. Group anchors by the real underlying problem they describe, "
    "WITHIN the scope of a single theme. Do NOT cluster across themes. Prefer more "
    "clusters over fewer — only group anchors that describe the SAME specific problem.\n"
    "  PASS 3 — COVERAGE ANALYSIS. Note anchors that fit no configured theme, and "
    "suggest (non-authoritative) adjustments if the current theme set consistently "
    "strains to fit the anchors.\n\n"
    "THEMES ARE AUTHORITATIVE. When THEMES is non-empty:\n"
    "  - Every cluster's `theme` MUST be one of the configured themes, or the literal "
    "string `unassigned` for anchors that genuinely fit none of them.\n"
    "  - NEVER invent new themes. Candidate new themes go in `analysis.missing_themes` "
    "as suggestions only.\n"
    "  - An anchor may map to multiple themes if it supports more than one — the same "
    "anchor index may appear in multiple clusters (one per matching theme).\n"
    "When THEMES is empty, you are in PROPOSAL MODE: infer candidate themes from the "
    "anchors, use them as cluster `theme` values, AND list them in "
    "`analysis.missing_themes` so the user can adopt them authoritatively.\n\n"
    "CATEGORY CONVENTION. Observations may be written as `Category (specific instance)` "
    "— for example `Style issue (rushed cadence)`. The parenthesised body is the "
    "concrete instance — use it verbatim (or closely paraphrased) as the anchor's "
    "`instance`. The category prefix is a hint for theme assignment but does NOT "
    "override the authoritative THEMES list.\n\n"
    "ANCHORS PRESERVE INTENT. The anchor's `instance` must be grounded in the "
    "observation's concrete detail — quote exact words, phrases, or patterns. Do NOT "
    "write generic summaries like 'output doesn't follow instructions'. Cite what "
    "actually happened, e.g. 'Output begins with a conversational sentence like "
    "\"Here\\'s your updated list\" before the actual content' or 'Output replaces "
    "[ ] checkbox markers with bullet-pointed priority sections'.\n\n"
    "GENERALIZATION IS CONSTRAINED. Each cluster's `summary` generalises only across "
    "its own anchors within its theme. The summary must be explainable from the "
    "anchors it groups.\n\n"
    "Output ONLY a single JSON object with these top-level keys:\n"
    '  "anchors": list of {"index": int, "instance": str, '
    '"confidence": "high"|"medium"|"low", "themes": list of theme names the anchor '
    "supports (or [] if none)}.\n"
    '  "clusters": list of {"theme": one of THEMES or "unassigned" (or a proposed '
    'theme in proposal mode), "failure_mode": short kebab-case tag (e.g. '
    '"chatty-preamble"), "summary": specific description grounded in the anchors, '
    '"anchor_indices": list of 0-based indices into `anchors`}.\n'
    '  "analysis": {"missing_themes": list of suggested theme names (always '
    'suggestions, never authoritative), "theme_adjustments": list of suggested '
    "refinements to existing themes as short strings}.\n"
    "Do not wrap in markdown fences."
)


@dataclass(frozen=True)
class Analysis:
    """Coverage analysis output alongside the issue file.

    Themes and adjustments here are always suggestions — they never
    mutate the configured theme list, only inform the user.
    """

    missing_themes: list[str] = field(default_factory=list)
    theme_adjustments: list[str] = field(default_factory=list)
    unassigned_anchor_count: int = 0
    dropped_invented_theme_count: int = 0


@dataclass(frozen=True)
class ConsolidationResult:
    """Output of `consolidate_feedback`: the issue file plus coverage analysis."""

    issue_file: IssueFile
    analysis: Analysis

    @property
    def issues(self) -> list[Issue]:
        return self.issue_file.issues


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


def _parse_llm_response(raw: str) -> dict | list:
    """Parse the LLM's JSON response, tolerating surrounding text or fences."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Try object first, then array
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = raw.find(open_ch)
        end = raw.rfind(close_ch) + 1
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                continue
    return {}


def _normalize_cluster_response(
    parsed: dict | list,
) -> tuple[list[dict], list[dict], dict]:
    """Coerce legacy (bare cluster array) and new (anchors+clusters+analysis)
    response shapes into a uniform (anchors, clusters, analysis) triple.

    Legacy shape is preserved so responses from the pre-themes prompt — and
    test fixtures still returning bare arrays — keep parsing.
    """
    if isinstance(parsed, list):
        return [], list(parsed), {}
    if not isinstance(parsed, dict):
        return [], [], {}
    return (
        parsed.get("anchors", []) or [],
        parsed.get("clusters", []) or [],
        parsed.get("analysis", {}) or {},
    )


def consolidate_feedback(
    feedback_list: list[Feedback],
    prompt_ref: str,
    llm_client: LLMClient,
    min_occurrences: int = 1,
    existing_issues: list[Issue] | None = None,
    directive: str | None = None,
    themes: list[str] | None = None,
) -> ConsolidationResult:
    """Consolidate feedback into issues and emit coverage analysis.

    When `themes` is non-empty, cluster categories are constrained to those
    themes (plus 'unassigned' for anchors that fit none). When empty or None,
    the LLM proposes candidate themes — surfaced in `ConsolidationResult.analysis`
    for the user to adopt authoritatively.

    `existing_issues`, when provided, is re-fed as observations so the LLM
    re-clusters the full corpus rather than appending.
    """
    configured_themes = list(themes) if themes else []

    observations: list[tuple[str, str]] = []

    if existing_issues:
        seen: set[tuple[str, str]] = set()
        for issue in existing_issues:
            for ev in issue.evidence:
                key = (ev.source, ev.feedback)
                if key not in seen:
                    seen.add(key)
                    observations.append(key)

    for feedback in feedback_list:
        for entry in feedback.entries:
            for part in _split_feedback_entry(entry.text):
                observations.append((feedback.source_file, part))

    if not observations:
        return ConsolidationResult(
            issue_file=IssueFile(version="1.0", prompt_ref=prompt_ref, issues=[]),
            analysis=Analysis(),
        )

    lines = [f"[{idx}] ({source}) {text}" for idx, (source, text) in enumerate(observations)]
    themes_line = (
        "THEMES: " + ", ".join(configured_themes)
        if configured_themes
        else "THEMES: (none configured — propose candidates)"
    )
    user_prompt = themes_line + "\n\nOBSERVATIONS:\n" + "\n".join(lines)
    if directive:
        user_prompt = (
            f"IMPORTANT — follow this guidance when clustering:\n{directive}\n\n"
            + user_prompt
        )

    raw_response = llm_client.generate(user_prompt, system=_CLUSTER_SYSTEM)
    anchors_payload, clusters_payload, analysis_payload = _normalize_cluster_response(
        _parse_llm_response(raw_response)
    )

    # Index anchors by observation index so we can reuse their instance/confidence
    # when building IssueEvidence — gives downstream evals access to the distilled
    # anchor text in addition to the raw feedback.
    anchor_by_index: dict[int, dict] = {}
    for anchor in anchors_payload:
        if not isinstance(anchor, dict):
            continue
        idx = anchor.get("index")
        if isinstance(idx, int):
            anchor_by_index[idx] = anchor

    valid_themes = set(configured_themes)
    dropped_invented = 0

    total_sources = len({source for source, _ in observations})
    issues: list[Issue] = []
    issue_index = 1

    for cluster in clusters_payload:
        if not isinstance(cluster, dict):
            continue
        # New shape: theme + failure_mode. Legacy shape: label only.
        theme = cluster.get("theme")
        failure_mode = cluster.get("failure_mode") or cluster.get("label") or f"issue-{issue_index}"
        category = theme if theme else failure_mode

        if configured_themes and theme and theme not in valid_themes and theme != "unassigned":
            dropped_invented += 1
            continue

        summary = cluster.get("summary", "")
        indices = cluster.get("anchor_indices") or cluster.get("evidence_indices") or []

        evidence: list[IssueEvidence] = []
        for idx in indices:
            if not (isinstance(idx, int) and 0 <= idx < len(observations)):
                continue
            source, text = observations[idx]
            anchor = anchor_by_index.get(idx, {})
            instance = anchor.get("instance") if isinstance(anchor, dict) else None
            confidence = anchor.get("confidence") if isinstance(anchor, dict) else None
            if confidence not in ("high", "medium", "low"):
                confidence = None
            evidence.append(
                IssueEvidence(
                    source=source,
                    feedback=text,
                    instance=instance if isinstance(instance, str) and instance else None,
                    confidence=confidence,
                )
            )

        unique_sources = len({ev.source for ev in evidence})
        if total_sources > 1 and unique_sources < min_occurrences:
            continue

        severity = _determine_severity(unique_sources, total_sources)

        # When themed: category IS the theme — it's the user-authoritative axis.
        # The LLM's failure_mode is informational and lives in the summary.
        if theme and failure_mode and failure_mode not in summary:
            summary = f"[{failure_mode}] {summary}" if summary else failure_mode

        issues.append(
            Issue(
                id=_generate_issue_id(prompt_ref, issue_index),
                category=category,
                severity=severity,
                summary=summary,
                evidence=evidence,
            )
        )
        issue_index += 1

    unassigned_count = sum(
        1
        for anchor in anchors_payload
        if isinstance(anchor, dict) and not (anchor.get("themes") or [])
    )

    analysis = Analysis(
        missing_themes=[t for t in analysis_payload.get("missing_themes", []) if isinstance(t, str)],
        theme_adjustments=[a for a in analysis_payload.get("theme_adjustments", []) if isinstance(a, str)],
        unassigned_anchor_count=unassigned_count,
        dropped_invented_theme_count=dropped_invented,
    )

    return ConsolidationResult(
        issue_file=IssueFile(version="1.0", prompt_ref=prompt_ref, issues=issues),
        analysis=analysis,
    )


def load_issue_file(path: Path) -> IssueFile:
    """Load an issue file from disk."""
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)

    issues = []
    for issue_data in data.get("issues", []):
        evidence = [
            IssueEvidence(
                source=e["source"],
                feedback=e["feedback"],
                instance=e.get("instance"),
                confidence=e.get("confidence"),
            )
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
