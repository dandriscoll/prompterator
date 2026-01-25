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
  issues: ".prompterator/issues"
  evals: ".prompterator/evals"
  results: ".prompterator/results"

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
