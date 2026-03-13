"""Test command - run evals against a prompt."""

from pathlib import Path

import click

from prompterator.config.loader import get_config_base_dir, load_config
from prompterator.core.eval_runner import run_all_evals, save_result_file
from prompterator.core.run import create_run_dir
from prompterator.commands.resolve import ResolveError, resolve_prompt_and_evals
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
    help="Samples per eval (default: from config critic.samples)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed results",
)
def test_cmd(
    prompt: Path | None,
    evals_path: Path | None,
    output: Path | None,
    samples: int | None,
    verbose: bool,
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

    click.echo(f"Testing: {prompt}")
    click.echo(f"Evals: {len(eval_file.evals)} from {evals_path.name}")
    click.echo(f"Samples: {n_samples}, confidence: {config.critic.confidence_threshold:.0%}")
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
            samples=n_samples,
            confidence_threshold=config.critic.confidence_threshold,
            script=script, script_timeout=script_timeout,
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
            f"(score: {result.score:.2f})"
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
    click.echo(f"Score: {result_file.summary.overall_score:.2f}")
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
