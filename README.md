# Prompterator

Prompterator is a CLI for refining prompts with an Editor LLM, running the refined prompt with an Operator LLM, and evaluating the output against good/bad examples.

## Quick start

1. Initialize configuration:

```bash
prompterator init
```

2. Edit `.env` with your Editor/Operator LLM endpoints.

3. Run the workflow:

```bash
prompterator run --examples path/to/examples.txt --prompt "Initial prompt"
```

## Configuration

The `.env` file stores all configuration, including LLM endpoints, file handling mode, and the examples backend module.
Azure OpenAI endpoints are supported; provide the full deployment URL (including the `api-version` query)
and the API key.

`FILE_MODE` supports:

- `git`: write changes in place.
- `plain`: write to a new file with `OUTPUT_SUFFIX`.
- `auto`: detect a git repo and choose automatically.

## Examples format

Prompterator integrates with an external examples parser/evaluator (configured via `EXAMPLES_MODULE`). If the module is unavailable, the tool will still run but evaluation is skipped.
