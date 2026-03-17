"""Improve command - generate improved prompts via LLM."""

from pathlib import Path

import click

from prompterator.config.loader import get_config_base_dir, load_config
from prompterator.core.run import create_run_dir
from prompterator.core.eval_runner import find_latest_results, load_result_file
from prompterator.core.improver import (
    _build_diagnose_prompt,
    generate_improved_prompt,
    save_improved_prompt,
)
from prompterator.commands.resolve import ResolveError, resolve_issues
from prompterator.runners.naming import NamingClient, NamingError
from prompterator.runners.llm import LLMClient, LLMError


@click.command("improve")
@click.argument("prompt", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=False)
@click.option(
    "--issues",
    "issues_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to issue file (default: auto-detect from prompt name)",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output path for improved prompt (default: auto-generated)",
)
@click.option(
    "--samples",
    "-s",
    type=int,
    default=1,
    show_default=True,
    help="Number of test samples per iteration",
)
@click.option(
    "--runs",
    "-r",
    type=int,
    default=1,
    show_default=True,
    help="Number of improve→test iterations",
)
@click.option(
    "--in-place",
    is_flag=True,
    help="Overwrite the original prompt file (git mode)",
)
@click.option(
    "--directive",
    "-d",
    type=str,
    default=None,
    help='Specific change to make, e.g. "explicitly prohibit conversational preamble"',
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show the improved prompt without saving",
)
def improve_cmd(
    prompt: Path | None,
    issues_path: Path | None,
    output: Path | None,
    samples: int,
    runs: int,
    in_place: bool,
    directive: str | None,
    dry_run: bool,
) -> None:
    """Generate an improved prompt based on identified issues.

    PROMPT is the path to the prompt file to improve (optional — can be
    derived from issue files).
    """
    config = load_config()
    base_dir = get_config_base_dir()

    # --- Resolve prompt and issues ----------------------------------------
    from prompterator.commands.resolve import resolve_prompt
    if prompt is None:
        # Try config first
        prompt = resolve_prompt(config, base_dir)

    if prompt is None and issues_path is not None:
        # Derive prompt from issue file's prompt_ref
        from prompterator.core.issue import load_issue_file as _load
        issue_file_tmp = _load(issues_path)
        prompt = config.get_dir("prompts", base_dir) / issue_file_tmp.prompt_ref
        if not prompt.exists():
            click.echo(f"Prompt file not found at {prompt}")
            raise SystemExit(1)
    elif prompt is None:
        # Auto-detect from first .issue.yaml
        issues_dir = config.get_dir("issues", base_dir)
        issue_files = sorted(issues_dir.glob("*.issue.yaml"))
        if not issue_files:
            click.echo(f"No .issue.yaml files found in {issues_dir}")
            click.echo("Run 'prompterator issues' first, or specify a prompt or --issues path.")
            raise SystemExit(1)
        issues_path = issue_files[0]
        from prompterator.core.issue import load_issue_file as _load
        issue_file_tmp = _load(issues_path)
        prompt = config.get_dir("prompts", base_dir) / issue_file_tmp.prompt_ref
        if not prompt.exists():
            click.echo(f"Prompt file not found at {prompt}")
            raise SystemExit(1)

    try:
        issues_path, issue_file = resolve_issues(
            config, base_dir, prompt, issues_path,
        )
    except ResolveError as e:
        click.echo(str(e))
        raise SystemExit(1)

    if not issue_file.issues:
        click.echo("No issues to address in issue file.")
        raise SystemExit(1)

    # Find latest eval results to inform the improvement
    base_name = prompt.stem.split(".")[0]
    results_dir = config.get_dir("results", base_dir)
    latest_results_path = find_latest_results(results_dir, base_name)
    eval_results = None
    if latest_results_path:
        try:
            result_file = load_result_file(latest_results_path)
            eval_results = result_file.results
            click.echo(f"Using eval results from: {latest_results_path.parent.name}")
        except Exception as e:
            click.echo(f"Warning: could not load results: {e}", err=True)

    click.echo(f"Improving: {prompt}")
    click.echo(f"Based on: {len(issue_file.issues)} issues from {issues_path.name}")
    if eval_results:
        failed = [r for r in eval_results if not r.passed]
        click.echo(f"Eval results: {len(eval_results) - len(failed)}/{len(eval_results)} passing")
    click.echo()

    if dry_run:
        # Show the improvement prompt that would be sent to the LLM,
        # without requiring a working LLM connection.
        with open(prompt) as f:
            original_text = f.read()
        diagnose_prompt = _build_diagnose_prompt(original_text, issue_file, eval_results=eval_results)
        click.echo("--- Step 1: Diagnose prompt (would be sent to LLM) ---")
        click.echo(diagnose_prompt)
        click.echo("--- End ---")
        return

    # Initialize Editor LLM client
    try:
        llm = LLMClient(**config.resolve_role("editor"))
    except LLMError as e:
        click.echo(f"Editor LLM error: {e}", err=True)
        raise SystemExit(1)

    # Load eval file if available (needed for multi-run and helpful for single-run)
    from prompterator.core.eval_spec import load_eval_file
    evals_dir = config.get_dir("evals", base_dir)
    evals_path = evals_dir / f"{base_name}.eval.yaml"
    eval_file = None
    if evals_path.exists():
        try:
            eval_file = load_eval_file(evals_path)
        except Exception:
            pass

    # Multi-run mode: delegate to tuning loop
    if runs > 1:
        from prompterator.core.tuner import run_tuning_loop

        if eval_file is None:
            click.echo(f"No eval file found at {evals_path} (needed for multi-run mode)")
            raise SystemExit(1)

        try:
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

        click.echo(f"Running {runs} improve→test iterations...")

        def on_iteration(record):
            verdict_color = {"PASS": "green", "FAIL": "red", "PARTIAL": "yellow"}.get(
                record.summary.verdict, "white"
            )
            click.echo(
                f"  Run {record.iteration}: "
                f"score={record.summary.overall_score:.2f} "
                f"[{click.style(record.summary.verdict, fg=verdict_color)}]"
            )

        report = run_tuning_loop(
            prompt_path=prompt,
            issue_file=issue_file,
            eval_file=eval_file,
            editor_llm=llm,
            critic_llm=critic_llm,
            max_iterations=runs,
            on_iteration=on_iteration,
            author_llm=author_llm,
            samples=config.critic.samples,
            confidence_threshold=config.critic.confidence_threshold,
            critic_script=critic_script,
            critic_script_timeout=critic_script_timeout,
            results_dir=config.get_dir("results", base_dir),
        )

        improved = report.final_prompt
        click.echo(f"\nBest score: {report.final_summary.overall_score:.2f}")
        # Fall through to save logic below
        # Skip the single-improve path
    else:
        # Generate improved prompt (single run)
        from prompterator.core.improver import generate_improved_prompt_with_rationale
        from prompterator.runners.llm import debug_context
        debug_context("improve")
        click.echo("Generating improved prompt...")
        try:
            with open(prompt) as f:
                original_text = f.read()
            improved, rationale, _raw, _action = generate_improved_prompt_with_rationale(
                original_text, issue_file, llm, eval_results=eval_results,
                directive=directive, eval_file=eval_file,
            )
            click.echo(f"Change: {rationale}")
        except LLMError as e:
            click.echo(f"LLM error: {e}", err=True)
            raise SystemExit(1)

    # Check for git mode (from config or --in-place flag)
    use_in_place = in_place or config.workflow.git_mode

    # Determine output path
    if output is not None:
        # Explicit output path provided
        pass
    elif use_in_place:
        # Git mode: overwrite the original file
        output = prompt
        click.echo("(git mode: overwriting original file)")
    else:
        # Normal mode: create a new variation in a run directory
        run_dir = create_run_dir(prompt.parent)
        try:
            naming = NamingClient(
                executable=config.naming.executable,
                timeout=config.naming.timeout,
            )
            naming_config = naming.config()

            # Get the primary prior type
            prior_type = naming_config.prior_types[0] if naming_config.prior_types else "prompt.txt"
            output_str = naming.propose(str(prompt), prior_type)
            output = run_dir / Path(output_str).name
        except NamingError as e:
            # Fallback to simple naming
            click.echo(f"Warning: naming tool error ({e}), using fallback naming", err=True)
            stem = prompt.stem.split(".")[0]
            output = run_dir / f"{stem}a.prompt.txt"

        # Ensure we don't overwrite existing files (only in normal mode)
        while output.exists():
            name = output.name
            if "a" <= name[3:4] <= "z":
                # Increment variation letter
                letter = chr(ord(name[3]) + 1)
                output = output.parent / (name[:3] + letter + name[4:])
            else:
                output = output.parent / (name[:3] + "a" + name[3:])

    save_improved_prompt(improved, output)
    click.echo(f"\nSaved improved prompt to: {output}")
