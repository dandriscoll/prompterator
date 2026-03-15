"""Simplify command - consolidate and shorten a prompt while retaining accuracy."""

from pathlib import Path

import click

from prompterator.config.loader import get_config_base_dir, load_config
from prompterator.commands.resolve import (
    ResolveError,
    resolve_content,
    resolve_prompt_and_evals,
)
from prompterator.core.eval_runner import run_all_evals
from prompterator.core.improver import consolidate_redundant_lines, simplify_prompt
from prompterator.core.run import create_run_dir
from prompterator.runners.llm import LLMClient, LLMError


@click.command("simplify")
@click.argument("prompt", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=False)
@click.option(
    "--evals",
    "evals_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to eval file (default: auto-detect from prompt name)",
)
@click.option(
    "--max-steps",
    type=int,
    default=10,
    show_default=True,
    help="Maximum simplification steps",
)
@click.option(
    "--samples",
    "-s",
    type=int,
    default=None,
    help="Samples per eval for accuracy check (default: from config)",
)
@click.option(
    "--content",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Content file to pair with the prompt (overrides config)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be simplified without applying",
)
def simplify_cmd(
    prompt: Path | None,
    evals_path: Path | None,
    max_steps: int,
    samples: int | None,
    content: Path | None,
    dry_run: bool,
) -> None:
    """Simplify a prompt by consolidating and shortening while retaining accuracy.

    Iteratively applies simplifications (deduplication, merging, trimming)
    and tests after each step. Reverts any change that causes eval regressions.

    PROMPT is the path to the prompt file (optional — derived from config).
    """
    config = load_config()
    base_dir = get_config_base_dir()

    try:
        prompt, evals_path, eval_file = resolve_prompt_and_evals(
            config, base_dir, prompt, evals_path,
        )
    except ResolveError as e:
        click.echo(str(e))
        raise SystemExit(1)

    if not eval_file.evals:
        click.echo("No evaluations found. Cannot verify accuracy during simplification.")
        raise SystemExit(1)

    n_samples = samples if samples is not None else config.critic.samples
    n_ensemble = config.critic.ensemble
    content_texts = resolve_content(config, base_dir, content) or [None]
    n_content = len(content_texts)
    n_evals = len(eval_file.evals)

    n_outputs = n_content * n_samples
    calls_per_test = n_outputs * (1 + n_evals * n_ensemble)
    # Each step: 1 simplify LLM call + 1 test
    max_llm_calls = calls_per_test + max_steps * (1 + calls_per_test)

    click.echo(f"Simplifying: {prompt}")
    click.echo(f"Evals: {n_evals} from {evals_path.name}")
    if n_content > 1 or content_texts[0] is not None:
        click.echo(f"Content files: {n_content}")
    click.echo(f"Max steps: {max_steps}")
    click.echo(f"LLM calls: up to {max_llm_calls}")
    click.echo()

    if dry_run:
        click.echo("[dry-run] Would simplify with accuracy checks after each step.")
        return

    # Initialize LLMs
    try:
        editor_llm = LLMClient(**config.resolve_role("editor"))
        author_llm = LLMClient(**config.resolve_role("author"))
        critic_llm = None
        critic_script = None
        critic_script_timeout = config.critic.script_timeout
        if config.critic.mode == "script":
            critic_script = config.critic.script
        else:
            critic_llm = LLMClient(**config.resolve_role("critic"))
    except LLMError as e:
        click.echo(f"LLM error: {e}", err=True)
        raise SystemExit(1)

    from prompterator.runners.llm import debug_context

    with open(prompt) as f:
        current_text = f.read()

    original_len = len(current_text)

    # Baseline accuracy
    debug_context("simplify.baseline")
    click.echo("Checking baseline accuracy...")

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".prompt.txt", delete=False) as f:
        f.write(current_text)
        tmp_path = Path(f.name)

    try:
        baseline = run_all_evals(
            eval_file, tmp_path, critic_llm,
            author_llm=author_llm,
            content_texts=content_texts if content_texts != [None] else None,
            samples=n_samples, ensemble=n_ensemble,
            confidence_threshold=config.critic.confidence_threshold,
            script=critic_script, script_timeout=critic_script_timeout,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    baseline_score = baseline.summary.overall_score
    click.echo(f"Baseline: {baseline_score:.1f}/10, {baseline.summary.passed_count}/{len(baseline.results)} evals passing")
    click.echo()

    # Phase 1: Consolidate redundant lines (no accuracy check needed — mechanical)
    click.echo("Phase 1: Consolidating redundant lines...")
    consolidated, con_rationale, _ = consolidate_redundant_lines(current_text, editor_llm)
    if consolidated != current_text:
        click.echo(f"  Consolidated: {con_rationale}")
        current_text = consolidated
    else:
        click.echo("  No redundancies found.")

    # Phase 2: Iterative simplification with accuracy checks
    click.echo()
    click.echo("Phase 2: Simplifying with accuracy checks...")

    accepted = 0
    rejected = 0

    for step in range(1, max_steps + 1):
        debug_context(f"simplify.{step}")
        simplified, rationale, changed = simplify_prompt(current_text, editor_llm)

        if not changed:
            click.echo(f"  Step {step}: No further simplifications possible.")
            break

        # Test accuracy of simplified version
        with tempfile.NamedTemporaryFile(mode="w", suffix=".prompt.txt", delete=False) as f:
            f.write(simplified)
            tmp_path = Path(f.name)

        try:
            result = run_all_evals(
                eval_file, tmp_path, critic_llm,
                author_llm=author_llm,
                content_texts=content_texts if content_texts != [None] else None,
                samples=n_samples, ensemble=n_ensemble,
                confidence_threshold=config.critic.confidence_threshold,
                script=critic_script, script_timeout=critic_script_timeout,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        new_score = result.summary.overall_score

        # Accept if accuracy didn't drop
        noise_margin = 5.0 / max(len(result.results), 1)
        if new_score >= baseline_score - noise_margin:
            current_text = simplified
            accepted += 1
            status = click.style("KEPT", fg="green")
            click.echo(f"  Step {step}: {status} ({new_score:.1f}/10) — {rationale}")
        else:
            rejected += 1
            status = click.style("REVERTED", fg="red")
            click.echo(f"  Step {step}: {status} ({new_score:.1f}/10 < {baseline_score:.1f}/10) — {rationale}")

    # Save result
    with open(prompt, "w") as f:
        f.write(current_text)

    final_len = len(current_text)
    reduction = ((original_len - final_len) / original_len * 100) if original_len > 0 else 0

    click.echo()
    click.echo("=" * 50)
    click.echo("Simplification Complete")
    click.echo("=" * 50)
    click.echo(f"Steps: {accepted} accepted, {rejected} reverted")
    click.echo(f"Size: {original_len} → {final_len} chars ({reduction:.0f}% reduction)")
    click.echo(f"Prompt updated: {prompt}")
