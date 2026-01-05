from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from .config import ConfigError, init_env, load_config
from .llm import LLMError
from .workflow import run_workflow


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Refine prompts with an Editor LLM and execute with an Operator LLM.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create a .env configuration file.")
    init_parser.add_argument("--path", default=".env", help="Path to write the .env file.")
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the .env file if it already exists.",
    )

    run_parser = subparsers.add_parser("run", help="Run the prompt refinement workflow.")
    run_parser.add_argument("--env", default=".env", help="Path to the .env file.")
    run_parser.add_argument("--prompt", help="Initial prompt string.")
    run_parser.add_argument("--prompt-file", help="Path to a file containing the prompt.")
    run_parser.add_argument("--examples", required=True, help="Path to examples file.")
    run_parser.add_argument("--feedback", help="Feedback to apply to the prompt.")
    run_parser.add_argument("--feedback-file", help="Path to a feedback file.")
    run_parser.add_argument(
        "--apply-to",
        help="Write operator output to this file using the configured file mode.",
    )
    run_parser.add_argument(
        "--show-refined",
        action="store_true",
        help="Print the refined prompt.",
    )
    run_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-essential output.",
    )

    args = parser.parse_args(argv)

    if args.command == "init":
        return _handle_init(args)
    if args.command == "run":
        return _handle_run(args)

    parser.print_help()
    return 1


def _handle_init(args: argparse.Namespace) -> int:
    path = Path(args.path)
    try:
        init_env(path, force=args.force)
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Wrote configuration to {path}")
    return 0


def _handle_run(args: argparse.Namespace) -> int:
    if args.prompt and args.prompt_file:
        print("Use --prompt or --prompt-file, not both.", file=sys.stderr)
        return 1
    if args.feedback and args.feedback_file:
        print("Use --feedback or --feedback-file, not both.", file=sys.stderr)
        return 1

    prompt = _read_optional(args.prompt_file, args.prompt)
    feedback = _read_optional(args.feedback_file, args.feedback)
    env_path = Path(args.env)
    try:
        config = load_config(env_path)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    examples_path = Path(args.examples)
    apply_to = Path(args.apply_to) if args.apply_to else None

    try:
        result = run_workflow(
            config=config,
            initial_prompt=prompt,
            examples_path=examples_path,
            feedback=feedback,
            apply_to=apply_to,
            cwd=Path.cwd(),
        )
    except LLMError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not args.quiet:
        _print_result(result, show_refined=args.show_refined)

    return 0


def _read_optional(path: Optional[str], inline: Optional[str]) -> str:
    if path:
        return Path(path).read_text(encoding="utf-8")
    return inline or ""


def _print_result(result, *, show_refined: bool) -> None:
    if show_refined:
        print("Refined prompt:")
        print(result.refined_prompt)
        print("")

    print("Operator output:")
    print(result.operator_output)
    print("")

    evaluation = result.evaluation
    score_text = "n/a" if evaluation.score is None else str(evaluation.score)
    print(f"Evaluation: {evaluation.status} (score: {score_text})")
    if evaluation.details:
        print(f"Details: {evaluation.details}")

    if result.applied_path:
        print("")
        print(
            f"Wrote operator output to {result.applied_path} "
            f"(mode: {result.file_mode})."
        )
