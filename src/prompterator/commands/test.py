"""Test command - run evals against a prompt."""

from pathlib import Path

import click

from prompterator.config.loader import get_config_base_dir, load_config
from prompterator.core.eval_runner import run_all_evals, save_result_file
from prompterator.core.eval_spec import load_eval_file
from prompterator.runners.llm import LLMClient, LLMError


@click.command("test")
@click.argument("prompt", type=click.Path(exists=True, dir_okay=False, path_type=Path))
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
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed results",
)
def test_cmd(
    prompt: Path,
    evals_path: Path | None,
    output: Path | None,
    verbose: bool,
) -> None:
    """Run evaluations against a prompt and report results.

    PROMPT is the path to the prompt file to test.
    """
    config = load_config()
    base_dir = get_config_base_dir()

    # Find eval file if not specified
    if evals_path is None:
        evals_dir = config.get_dir("evals", base_dir)
        base_name = prompt.stem.split(".")[0]
        evals_path = evals_dir / f"{base_name}.eval.yaml"

        if not evals_path.exists():
            click.echo(f"No eval file found at {evals_path}")
            click.echo("Run 'prompterator evals' first or specify --evals path.")
            raise SystemExit(1)

    try:
        eval_file = load_eval_file(evals_path)
    except Exception as e:
        click.echo(f"Error loading eval file: {e}", err=True)
        raise SystemExit(1)

    if not eval_file.evals:
        click.echo("No evaluations to run.")
        raise SystemExit(1)

    click.echo(f"Testing: {prompt}")
    click.echo(f"Evals: {len(eval_file.evals)} from {evals_path.name}")
    click.echo()

    # Initialize LLM client
    try:
        llm = LLMClient(
            runner=config.llm.runner,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
        )
    except LLMError as e:
        click.echo(f"LLM error: {e}", err=True)
        raise SystemExit(1)

    # Run evaluations
    click.echo("Running evaluations...")
    try:
        result_file = run_all_evals(eval_file, prompt, llm)
    except LLMError as e:
        click.echo(f"LLM error during evaluation: {e}", err=True)
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

    # Save results if output specified or to default location
    if output is None:
        results_dir = config.get_dir("results", base_dir)
        base_name = prompt.stem.split(".")[0]
        # Include variation in filename if present
        stem_parts = prompt.stem.split(".")
        if len(stem_parts[0]) > 3 and stem_parts[0][3:4].isalpha():
            base_name = stem_parts[0]
        output = results_dir / f"{base_name}.results.yaml"

    save_result_file(result_file, output)
    click.echo(f"\nResults saved to: {output}")

    # Exit with error if failed
    if result_file.summary.verdict == "FAIL":
        raise SystemExit(1)
