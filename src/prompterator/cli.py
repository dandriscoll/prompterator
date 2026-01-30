"""CLI definition for prompterator."""

import click

from prompterator import __version__
from prompterator.commands.collect import collect_cmd
from prompterator.commands.evals import evals_cmd
from prompterator.commands.feedback import feedback_cmd
from prompterator.commands.improve import improve_cmd
from prompterator.commands.init import init_cmd
from prompterator.commands.issues import issues_cmd
from prompterator.commands.status import status_cmd
from prompterator.commands.test import test_cmd
from prompterator.commands.tune import tune_cmd


@click.group()
@click.version_option(version=__version__, prog_name="prompterator")
def main() -> None:
    """Prompterator - CLI tool for prompt improvement workflow.

    Manage the prompt improvement workflow:

    \b
    1. feedback  - Parse and display .mb feedback files
    2. issues    - Consolidate feedback into .issue.yaml files
    3. evals     - Generate .eval.yaml files from issues
    4. improve   - Generate improved prompts via LLM
    5. test      - Run evals against prompts
    6. tune      - Run full improve→test loop iteratively

    \b
    Utilities:
    - collect  - Gather mb files with their source and prior files
    - status   - Show workflow state

    Run 'prompterator init' to create a configuration file.
    """
    pass


# Register commands
main.add_command(init_cmd)
main.add_command(feedback_cmd)
main.add_command(issues_cmd)
main.add_command(evals_cmd)
main.add_command(improve_cmd)
main.add_command(test_cmd)
main.add_command(status_cmd)
main.add_command(collect_cmd)
main.add_command(tune_cmd)


if __name__ == "__main__":
    main()
