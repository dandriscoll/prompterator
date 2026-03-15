"""Test command - run evals against a prompt."""

from pathlib import Path

import click

from prompterator.config.loader import get_config_base_dir, load_config
from prompterator.core.eval_runner import map_content_to_evals, run_all_evals, save_result_file
from prompterator.core.run import create_run_dir
from prompterator.commands.resolve import (
    ResolveError,
    resolve_content_with_paths,
    resolve_feedback,
    resolve_issues,
    resolve_prompt_and_evals,
)
from prompterator.runners.critic_script import CriticScriptError
from prompterator.runners.llm import LLMClient, LLMError


@click.command("test")
@click.argument("prompt", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=False)
@click.option(
    "--evals",
    "evals_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to eval file (default: auto-detect from prompt name)",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output path for results file (default: auto-generated)",
)
@click.option(
    "--samples",
    "-s",
    type=int,
    default=None,
    help="Author outputs per content file (default: from config critic.samples)",
)
@click.option(
    "--ensemble",
    "-e",
    type=int,
    default=None,
    help="Critic evaluations per output (default: from config critic.ensemble)",
)
@click.option(
    "--content",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Content file to pair with the prompt (overrides config)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed results",
)
@click.option(
    "--all-evals",
    is_flag=True,
    help="Run all evals for every content file (skip feedback-based filtering)",
)
def test_cmd(
    prompt: Path | None,
    evals_path: Path | None,
    output: Path | None,
    samples: int | None,
    ensemble: int | None,
    content: Path | None,
    verbose: bool,
    all_evals: bool,
) -> None:
    """Run evaluations against a prompt and report results.

    PROMPT is the path to the prompt file to test (optional — can be
    derived from eval files).
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
        click.echo("No evaluations to run.")
        raise SystemExit(1)

    n_samples = samples if samples is not None else config.critic.samples
    n_ensemble = ensemble if ensemble is not None else config.critic.ensemble

    # Resolve content files (with paths for feedback mapping)
    content_pairs = resolve_content_with_paths(config, base_dir, content)
    if content_pairs:
        content_paths = [p for p, _ in content_pairs]
        content_texts = [t for _, t in content_pairs]
    else:
        content_paths = []
        content_texts = [None]

    n_content = len(content_texts)
    n_evals = len(eval_file.evals)

    # --- Feedback-based eval filtering ------------------------------------
    content_eval_map: dict[int, list[str]] | None = None

    if not all_evals and content_paths:
        # Try to build content → eval mapping from feedback chain
        try:
            _, issue_file = resolve_issues(config, base_dir, prompt)
            feedback_list = resolve_feedback(
                config, base_dir, eval_file.prompt_ref,
            )
            if feedback_list:
                content_eval_map = map_content_to_evals(
                    content_paths, feedback_list, issue_file, eval_file,
                )
        except (ResolveError, Exception):
            # Issues or feedback not available — run all evals
            pass

    if content_eval_map is not None:
        # Report the optimization
        total_pairs = sum(len(eids) for eids in content_eval_map.values())
        full_pairs = n_content * n_evals
        click.echo(
            f"Feedback filtering: {total_pairs}/{full_pairs} content-eval pairs "
            f"(skipping evals without feedback on each content)"
        )
        click.echo("  Use --all-evals to override and run every eval for every content file.")
        n_critic = sum(len(eids) for eids in content_eval_map.values()) * n_samples * n_ensemble
    else:
        n_critic = n_content * n_samples * n_evals * n_ensemble

    n_outputs = n_samples * n_content
    n_author = n_outputs
    n_llm = n_author + n_critic

    click.echo(f"Testing: {prompt}")
    click.echo(f"Evals: {n_evals} from {evals_path.name}")
    if n_content > 1 or content_texts[0] is not None:
        click.echo(f"Content files: {n_content}")
    click.echo(f"Samples: {n_samples}, ensemble: {n_ensemble}, threshold: {config.critic.confidence_threshold:.1f}/10")
    click.echo(f"LLM calls: {n_llm} ({n_author} author + {n_critic} critic)")
    click.echo()

    # Initialize LLMs
    try:
        author_llm = LLMClient(**config.resolve_role("author"))
    except LLMError as e:
        click.echo(f"Author LLM error: {e}", err=True)
        raise SystemExit(1)

    llm = None
    script = None
    script_timeout = config.critic.script_timeout

    if config.critic.mode == "script":
        script = config.critic.script
        click.echo(f"Critic mode: script ({script})")
    else:
        click.echo("Critic mode: llm")
        try:
            llm = LLMClient(**config.resolve_role("critic"))
        except LLMError as e:
            click.echo(f"Critic LLM error: {e}", err=True)
            raise SystemExit(1)

    # Run evaluations
    from prompterator.runners.llm import debug_context
    debug_context("test")
    click.echo("Running evaluations...")
    try:
        result_file = run_all_evals(
            eval_file, prompt, llm,
            author_llm=author_llm,
            content_texts=content_texts if content_texts != [None] else None,
            samples=n_samples,
            ensemble=n_ensemble,
            confidence_threshold=config.critic.confidence_threshold,
            script=script, script_timeout=script_timeout,
            content_eval_map=content_eval_map,
        )
    except LLMError as e:
        click.echo(f"LLM error during evaluation: {e}", err=True)
        raise SystemExit(1)
    except CriticScriptError as e:
        click.echo(f"Critic script error: {e}", err=True)
        raise SystemExit(1)

    # Display results
    click.echo()
    click.echo("Results:")
    click.echo("-" * 40)

    for result in result_file.results:
        status = "PASS" if result.passed else "FAIL"
        status_color = "green" if result.passed else "red"
        click.echo(
            f"  [{click.style(status, fg=status_color)}] {result.eval_id} "
            f"({result.score:.1f}/10)"
        )
        if verbose and result.details:
            click.echo(f"        {result.details}")

    click.echo("-" * 40)
    verdict_color = {
        "PASS": "green",
        "FAIL": "red",
        "PARTIAL": "yellow",
    }.get(result_file.summary.verdict, "white")

    click.echo(
        f"Verdict: {click.style(result_file.summary.verdict, fg=verdict_color, bold=True)}"
    )
    click.echo(f"Score: {result_file.summary.overall_score:.1f}/10")
    click.echo(
        f"Passed: {result_file.summary.passed_count}/{result_file.summary.passed_count + result_file.summary.failed_count}"
    )

    # Save results
    base_name = prompt.stem.split(".")[0]
    stem_parts = prompt.stem.split(".")
    if len(stem_parts[0]) > 3 and stem_parts[0][3:4].isalpha():
        base_name = stem_parts[0]

    if output is not None:
        run_dir = output if output.is_dir() else output.parent
    else:
        results_dir = config.get_dir("results", base_dir)
        run_dir = create_run_dir(results_dir)

    save_result_file(result_file, run_dir / f"{base_name}.results.yaml")
    if result_file.generated_output:
        (run_dir / f"{base_name}.output.txt").write_text(result_file.generated_output)

    click.echo(f"\nResults saved to: {run_dir}")

    # Exit with error if failed
    if result_file.summary.verdict == "FAIL":
        raise SystemExit(1)
