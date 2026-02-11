# Prompterator

CLI tool for prompt improvement workflow.

## Installation

```bash
pip install prompterator
```

For LLM support:

```bash
pip install prompterator[anthropic]  # Anthropic Claude
pip install prompterator[openai]     # OpenAI GPT
pip install prompterator[all]        # All LLM providers
```

## Quick Start

```bash
# Initialize configuration
prompterator init

# View feedback from .mb files
prompterator feedback

# Consolidate feedback into issues
prompterator issues

# Generate evaluations from issues
prompterator evals

# Improve a prompt based on issues
prompterator improve 001.prompt.txt

# Test a prompt against evaluations
prompterator test 001a.prompt.txt

# Check workflow status
prompterator status
```

## Conceptual Model

Prompterator uses three distinct LLM roles, each with a specific responsibility:

```
                    ┌─────────┐
                    │  Prior  │
                    └────┬────┘
                         │
                         ▼
                    ┌─────────┐
                    │ Author  │  Takes a prior, produces a source
                    └────┬────┘
                         │
                         ▼
                    ┌─────────┐
    Feedback ──────▶│ Editor  │  Turns feedback into evals,
                    └────┬────┘  makes changes to prompts
                         │
                         ▼
                    ┌─────────┐
                    │ Critic  │  Runs evals
                    └────┬────┘
                         │
                         ▼
                    ┌─────────┐
                    │ Results │
                    └─────────┘
```

### Author

The **Author** takes a prior (context, requirements, examples) and produces a source
(the initial prompt). This is the generative, creative role.

### Editor

The **Editor** has two responsibilities:
1. **Feedback → Evals**: Transforms human feedback into structured evaluation criteria
2. **Prompt Changes**: Improves prompts based on identified issues

The Editor is iterative and refinement-focused, working to address specific concerns.

### Critic

The **Critic** runs evaluations against prompts and produces objective assessments.
It uses a lower temperature (0.3 by default) for consistent, reproducible judgments.

## Workflow

1. Create prompt files (`.prompt.txt` or `.prompt.md`)
2. Annotate with feedback in `.mb` markback files
3. Run `prompterator issues` to consolidate feedback
4. Run `prompterator evals` to generate evaluations
5. Run `prompterator improve <prompt>` to generate improvements (Editor)
6. Run `prompterator test <prompt>` to validate improvements (Critic)

## Git Mode

Git mode enables in-place editing of prompts, useful when you have files committed
in a git repo and want to overwrite them instead of creating new variations.

Enable via config:
```yaml
workflow:
  git_mode: true
```

Or use the `--in-place` flag:
```bash
prompterator improve 001.prompt.txt --in-place
```

In git mode, `prompterator improve` overwrites the original prompt file instead of
creating a new variation (e.g., `001a.prompt.txt`). This allows you to:
1. Commit the original prompt
2. Run improve with git mode
3. Review the diff to see what changed

## Configuration

Configuration is stored in `prompterator.yaml`:

```yaml
version: "1.0"

directories:
  prompts: "."
  feedback: "."
  issues: "issues"
  evals: "evals"
  results: "results"

# LLM roles (see Conceptual Model above)
author:
  runner: "anthropic"
  temperature: 0.7
  max_tokens: 4096

editor:
  runner: "anthropic"
  temperature: 0.7
  max_tokens: 4096

critic:
  runner: "anthropic"
  temperature: 0.3  # Lower for consistent evaluation
  max_tokens: 4096

feedback:
  categories: [clarity, completeness, accuracy, tone, format]
  min_occurrences: 1

ft:
  executable: "ft"
  timeout: 30

workflow:
  git_mode: false  # Set to true for in-place editing
```

Each role can use a different LLM provider or settings. For example, you might use
a larger model for the Author and a faster model for the Critic.

## File Naming Tool (ft)

Prompterator includes a file naming tool (`ft`) that manages filenames according to
a structured naming convention. The tool implements the
[Boutiques](https://boutiques.github.io/) descriptor format for self-documentation
and introspection.

### Boutiques Contract

Tools implementing the Boutiques contract provide:

- `--descriptor` — Machine-readable JSON schema describing the tool
- `--help` — Human-readable usage documentation

```bash
# Get JSON descriptor for programmatic introspection
ft --descriptor

# Get human-readable help
ft --help
```

The descriptor includes:

| Field | Description |
|-------|-------------|
| `name` | Tool identifier |
| `description` | Human-readable description |
| `tool-version` | Semantic version of the tool |
| `schema-version` | Boutiques schema version (0.5) |
| `x-spec-url` | Link to full Boutiques schema |
| `command-line` | Command template with placeholders |
| `inputs` | Array of input parameter definitions |
| `output-files` | Description of tool outputs |
| `groups` | Logical groupings of related inputs |
| `custom` | Tool-specific metadata (operations, patterns) |

For the full Boutiques schema specification, see:
https://github.com/boutiques/boutiques/blob/master/boutiques/schema/descriptor.schema.json

### Implementing a Compatible Tool

To implement a tool compatible with this contract:

1. Embed a JSON descriptor following the Boutiques schema
2. Output the descriptor when invoked with `--descriptor`
3. Output help text when invoked with `--help` or `-h`
4. Document operations in `custom.operations` with examples
5. Use standard exit codes (0 for success, non-zero for failure)

Example descriptor structure:

```json
{
  "name": "mytool",
  "description": "Tool description",
  "tool-version": "1.0.0",
  "schema-version": "0.5",
  "x-spec-url": "https://github.com/boutiques/boutiques/blob/master/boutiques/schema/descriptor.schema.json",
  "command-line": "mytool [OPERATION] [ARGS]",
  "inputs": [
    {
      "id": "operation",
      "name": "Operation",
      "description": "The operation to perform",
      "type": "String",
      "optional": true,
      "value-key": "[OPERATION]",
      "value-choices": ["list", "create", "delete"]
    }
  ],
  "output-files": [
    {
      "id": "stdout_output",
      "name": "Standard output",
      "description": "Operation result",
      "path-template": "-",
      "optional": false
    }
  ],
  "custom": {
    "operations": {
      "list": {
        "description": "List all items",
        "arguments": [],
        "examples": ["mytool list"]
      }
    }
  }
}
```

### Filename Convention

The `ft` tool uses the pattern: `[index][variation?][-name?][.extension]`

- `index` — Zero-padded digits (e.g., `001`, `0042`)
- `variation` — Lowercase letters for alternates (`a`, `b`, ..., `z`, `za`, `zb`, ...)
- `name` — Optional descriptor, hyphen-prefixed
- `extension` — Type-specific extension

### Operations

| Operation | Description |
|-----------|-------------|
| `config` | Print tool configuration |
| `propose <path> <type>` | Propose new filename with target type |
| `ready <path>` | Check if file is ready for transformation |
| `bundles [dir]` | List file bundles in directory |

## File Formats

### Feedback (.mb)

Markback files contain annotations about prompts.

### Issues (.issue.yaml)

```yaml
version: "1.0"
prompt_ref: "001.prompt.txt"
issues:
  - id: "issue-001-clarity-01"
    category: "clarity"
    severity: "high"
    summary: "Error handling instructions are ambiguous"
    evidence:
      - source: "001.mb"
        feedback: "clarity=low; needs=error examples"
```

### Evals (.eval.yaml)

```yaml
version: "1.0"
prompt_ref: "001.prompt.txt"
evals:
  - id: "eval-001-clarity-01"
    type: "rubric"
    rubric:
      criteria: ["Instructions are unambiguous", "Language is clear"]
      scoring: "all_required"
```

### Results (.results.yaml)

```yaml
version: "1.0"
prompt_tested: "001a.prompt.txt"
results:
  - eval_id: "eval-001-clarity-01"
    passed: true
    score: 1.0
summary:
  overall_score: 1.0
  verdict: "PASS"
```

## License

MIT
