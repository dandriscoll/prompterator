"""Guide command - show where the user is in the workflow and what to do next."""

from pathlib import Path

import click

from prompterator.config.loader import (
    CONFIG_FILENAME,
    find_config_file,
    get_config_base_dir,
    load_config,
)


def _count(directory: Path, pattern: str) -> int:
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


@click.command("guide")
def guide_cmd() -> None:
    """Show where you are in the workflow and what to do next."""
    config_path = find_config_file()

    if config_path is None:
        click.echo("No prompterator.yaml found.")
        click.echo()
        click.echo("To get started:")
        click.echo("  prompterator init       Interactive setup")
        click.echo("  prompterator howto       Print setup guide (for LLM agents)")
        return

    config = load_config(config_path)
    base_dir = get_config_base_dir(config_path)

    prompts_dir = config.get_dir("prompts", base_dir)
    feedback_dir = config.get_dir("feedback", base_dir)
    issues_dir = config.get_dir("issues", base_dir)
    evals_dir = config.get_dir("evals", base_dir)
    results_dir = config.get_dir("results", base_dir)

    n_prompts = _count(prompts_dir, "*.prompt.txt") + _count(prompts_dir, "*.prompt.md")
    n_feedback = _count(feedback_dir, "**/*.mb")
    n_issues = _count(issues_dir, "*.issue.yaml")
    n_evals = _count(evals_dir, "*.eval.yaml")
    n_results = _count(results_dir, "**/*.results.yaml")

    # Determine workflow stage
    STEPS = [
        ("prompts", n_prompts),
        ("feedback", n_feedback),
        ("issues", n_issues),
        ("evals", n_evals),
        ("results", n_results),
    ]

    # Find the last completed stage (has files) and first incomplete stage
    last_done = -1
    for i, (_, count) in enumerate(STEPS):
        if count > 0:
            last_done = i

    # Print compact status line
    parts = []
    for name, count in STEPS:
        marker = "+" if count > 0 else "."
        parts.append(f"{marker}{name}({count})")
    click.echo(" → ".join(parts))
    click.echo()

    # Determine stage and give guidance
    if n_prompts == 0:
        click.echo("Stage: not started")
        click.echo()
        click.echo("Create prompt files (*.prompt.txt or *.prompt.md) in:")
        click.echo(f"  {prompts_dir}")
    elif n_feedback == 0:
        click.echo(f"Stage: prompts ready ({_plural(n_prompts, 'prompt')})")
        click.echo()
        click.echo("Next: write feedback on prompt outputs using markback (.mb) files.")
        click.echo("  mb new <file>           Create a new .mb feedback file")
        click.echo("  prompterator annotate   Create .mb with boilerplate filled in")
    elif n_issues == 0:
        click.echo(f"Stage: feedback collected ({_plural(n_feedback, 'file')})")
        click.echo()
        click.echo("Next: consolidate feedback into issues.")
        click.echo("  prompterator issues")
    elif n_evals == 0:
        click.echo(f"Stage: issues identified ({_plural(n_issues, 'issue')})")
        click.echo()
        click.echo("Next: generate eval criteria from issues.")
        click.echo("  prompterator evals")
        click.echo()
        click.echo("Optional: review issues first in " + str(issues_dir))
    elif n_results == 0:
        click.echo(f"Stage: evals ready ({_plural(n_evals, 'eval')})")
        click.echo()
        click.echo("Next: calibrate evals, then improve and test prompts.")
        click.echo("  prompterator calibrate  Verify evals match human labels")
        click.echo("  prompterator improve    Generate an improved prompt")
        click.echo("  prompterator test       Run evals against a prompt")
        click.echo("  prompterator tune       Full improve→test loop")
    else:
        click.echo(f"Stage: results available ({_plural(n_results, 'result')})")
        click.echo()
        click.echo("Review results, add more feedback, and iterate:")
        click.echo(f"  Results in {results_dir}")
        click.echo("  prompterator tune       Continue improving")
        click.echo("  prompterator status -v  Detailed file listings")
