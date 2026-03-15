"""CLI definition for prompterator."""

from pathlib import Path as _Path

import click
from dotenv import load_dotenv

from prompterator import __version__
from prompterator.commands.annotate import annotate_cmd
from prompterator.commands.calibrate import calibrate_cmd
from prompterator.commands.collect import collect_cmd
from prompterator.commands.evals import evals_cmd
from prompterator.commands.feedback import feedback_cmd
from prompterator.commands.generate import generate_cmd
from prompterator.commands.guide import guide_cmd
from prompterator.commands.howto import howto_cmd
from prompterator.commands.improve import improve_cmd
from prompterator.commands.init import init_cmd
from prompterator.commands.issues import issues_cmd
from prompterator.commands.simplify import simplify_cmd
from prompterator.commands.status import status_cmd
from prompterator.commands.test import test_cmd
from prompterator.commands.tune import tune_cmd


_debug_option = click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Log all LLM input/output to debug.log in the output directory",
    is_eager=True,
    expose_value=False,
    callback=lambda ctx, param, value: (
        __import__("prompterator.runners.llm", fromlist=["enable_debug_log"]).enable_debug_log()
        if value
        else None
    ),
)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="prompterator")
def main() -> None:
    """Prompterator - CLI tool for prompt improvement workflow.

    Manage the prompt improvement workflow:

    \b
    1. feedback   - Parse and display .mb feedback files
    2. issues     - Consolidate feedback into .issue.yaml files
    3. evals      - Generate .eval.yaml files from issues
    4. calibrate  - Verify evals agree with human labels
    5. improve    - Generate improved prompts via LLM
    6. test       - Run evals against prompts
    7. tune       - Run full improve→test loop iteratively
    8. generate   - Run a prompt through the Author LLM

    \b
    Utilities:
    - annotate - Create .mb feedback files without writing boilerplate
    - collect  - Gather mb files with their source and prior files
    - status   - Show workflow state
    - guide    - Show where you are and what to do next
    - howto    - Print setup guide (for LLM agents)

    Run 'prompterator init' to create a configuration file.
    """
    _dotenv_path = _Path.cwd() / ".env"
    if _dotenv_path.is_file():
        load_dotenv(_dotenv_path, override=True)


# Apply --debug to every command that uses LLMs
for _cmd in (
    calibrate_cmd,
    evals_cmd,
    generate_cmd,
    improve_cmd,
    issues_cmd,
    simplify_cmd,
    test_cmd,
    tune_cmd,
):
    _debug_option(_cmd)

# Register commands
main.add_command(init_cmd)
main.add_command(annotate_cmd)
main.add_command(feedback_cmd)
main.add_command(issues_cmd)
main.add_command(evals_cmd)
main.add_command(calibrate_cmd)
main.add_command(improve_cmd)
main.add_command(test_cmd)
main.add_command(simplify_cmd)
main.add_command(status_cmd)
main.add_command(collect_cmd)
main.add_command(tune_cmd)
main.add_command(generate_cmd)
main.add_command(guide_cmd)
main.add_command(howto_cmd)


if __name__ == "__main__":
    main()
