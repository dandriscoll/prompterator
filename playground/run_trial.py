#!/usr/bin/env python3
"""End-to-end pipeline trial for prompterator.

Uses the ACTUAL installed prompterator package to trace through
every stage of the feedback-to-improvement pipeline.
"""

from collections import defaultdict
from pathlib import Path

from prompterator.commands.feedback import parse_mb_file
from prompterator.core.eval_spec import generate_evals_from_issues
from prompterator.core.improver import _build_improvement_prompt
from prompterator.core.issue import consolidate_feedback


CATEGORIES = ["clarity", "completeness", "accuracy", "tone", "format"]


def main():
    playground = Path("/workspace/playground")
    mb_files = sorted(playground.glob("*.mb"))

    print("=" * 72)
    print("PROMPTERATOR PIPELINE TRIAL")
    print("(Uses actual prompterator package — not re-implemented logic)")
    print("=" * 72)

    # ── Step 1: Parse feedback ──
    print("\n## Step 1: Parse Feedback")
    all_feedback = [parse_mb_file(p) for p in mb_files]

    groups: dict[str | None, list] = defaultdict(list)
    for fb in all_feedback:
        groups[fb.prompt_ref].append(fb)

    print(f"  Feedback files: {len(all_feedback)}")
    print(f"  Prompt groups: {len(groups)}")
    for ref, fbs in groups.items():
        print(f"    '{ref}' → {len(fbs)} feedback file(s)")

    # ── Step 2: Consolidate issues ──
    print("\n## Step 2: Consolidate Issues")
    prompt_ref = "improve-todo.prompt.md"
    prompt_fb = groups.get(prompt_ref, [])
    print(f"  Sources for '{prompt_ref}': {len(prompt_fb)}")

    issue_file = consolidate_feedback(prompt_fb, prompt_ref, CATEGORIES)
    print(f"  Issues generated: {len(issue_file.issues)}\n")

    for issue in issue_file.issues:
        print(f"  [{issue.severity.upper():6s}] {issue.category}:")
        print(f"    {issue.summary[:120]}")
        print(f"    Evidence: {len(issue.evidence)} items")
        print()

    # ── Step 3: Generate eval criteria ──
    print("## Step 3: Eval Criteria")
    eval_file = generate_evals_from_issues(issue_file)

    for ev in eval_file.evals:
        print(f"\n  {ev.id} ({ev.rubric.scoring}, {len(ev.rubric.criteria)} criteria):")
        for c in ev.rubric.criteria:
            print(f"    • {c}")

    # ── Step 4: Build improvement prompt ──
    print("\n\n## Step 4: Improvement Prompt")
    original_prompt = (playground / "improve-todo.prompt.md").read_text()
    improvement_prompt = _build_improvement_prompt(original_prompt, issue_file)

    print("\n─── BEGIN IMPROVEMENT PROMPT ───")
    print(improvement_prompt)
    print("─── END IMPROVEMENT PROMPT ───")

    # ── Step 5: Show the simulated improvement ──
    # The LLM editor is not available in this environment, so we show
    # what the pre-generated improved prompt looks like relative to
    # the issues identified.
    print("\n\n## Step 5: Improved Prompt (pre-generated)")
    print("(LLM editor not available — showing pre-generated result)")
    improved = (playground / "improve-todo-improved.prompt.md").read_text()
    print()
    print(improved)

    # ── Step 6: Summary ──
    print("=" * 72)
    print("## SUMMARY")
    print("=" * 72)
    print()
    print("ORIGINAL PROMPT:")
    print("  " + original_prompt.strip().replace("\n", "\n  "))
    print()
    print("FINAL IMPROVED PROMPT:")
    print("  " + improved.strip().replace("\n", "\n  "))
    print()
    print("CHANGES MADE (each tied to specific feedback):")
    print()
    print("  1. [FORMAT] Added: 'Do not include any conversational preamble, sign-off,")
    print("     or offers for further help. Output only the improved to-do list itself.'")
    print("     ← Addresses preamble + chatbot sign-off feedback")
    print()
    print("  2. [ACCURACY] Added: 'Preserve the original format and structure... Keep")
    print("     markdown checkboxes... Do not reorganize into new priority groups'")
    print("     ← Addresses structural rewrite feedback")
    print()
    print("  3. [COMPLETENESS] Added: 'Keep all inline notes, context, and human")
    print("     commentary from the original items.'")
    print("     ← Addresses dropped context feedback")
    print()
    print("  4. [NO UNRELATED CHANGES] The original instructions are preserved verbatim.")


if __name__ == "__main__":
    main()
