# Setting Up Prompterator for Your Project

Use the following prompt with an LLM to help you configure prompterator for a new project. Copy everything below the line.

---

I need help setting up **prompterator** — a CLI tool for iterative prompt improvement using structured human feedback. Walk me through configuring it for my project by asking me the following questions one at a time, waiting for my answer before moving on. After all questions are answered, generate my `prompterator.yaml` config file, a suggested `.env` file, and installation instructions.

## Installation

Prompterator requires **markback** — a markup format and editor for writing structured feedback annotations in `.mb` files. Markback is how human reviewers annotate LLM outputs with inline feedback that prompterator then consolidates into issues and evaluations.

Both packages should be installed together:

```bash
pip install prompterator markback
```

Markback is listed as a dependency of prompterator and will be pulled in automatically, but it's worth calling out because you'll use the `mb` command directly when writing feedback.

## Questions to ask me

### 1. Project context
- What does my project do, and what role do LLM prompts play in it?
- Where is my project root directory?

### 2. LLM providers
Prompterator uses three LLM roles — **Author** (drafts prompts), **Editor** (improves prompts from feedback), and **Critic** (evaluates prompts against rubrics). Each role can use a different provider/model.

Available built-in providers:
- **Anthropic Claude** — requires `ANTHROPIC_API_KEY`
- **OpenAI** — requires `OPENAI_API_KEY`
- **Azure OpenAI** — requires `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT`
- **Custom script** — any executable that reads a prompt on stdin and writes a response to stdout

Ask me:
- Which provider(s) do I want to use?
- Do I want the same model for all three roles, or different ones? (A common pattern is a stronger/slower model for Author and Editor, and a faster/cheaper model for Critic.)
- Which specific model(s)? (e.g. `claude-sonnet-4-20250514`, `gpt-4o`, `gpt-4o-mini`)
- Do I need a custom endpoint or API version for any provider?

### 3. Source data and directory layout
Prompterator works with several file types that can live in different directories:

| File type | Description | Default location |
|-----------|-------------|-----------------|
| **Prompts** (`.prompt.txt` / `.prompt.md`) | The prompts I'm improving | `.` |
| **Feedback** (`.mb` markback files) | Annotated human feedback on prompt outputs | `.` |
| **Issues** (`.issue.yaml`) | Consolidated feedback clusters | `issues/` |
| **Evals** (`.eval.yaml`) | Generated evaluation criteria (rubrics/assertions) | `evals/` |
| **Results** (`.results.yaml`) | Test execution results | `results/` |

Ask me:
- Where are my existing prompt files, or where should they go?
- Where will I store human feedback files?
- Am I happy with the default output directories for issues, evals, and results, or do I want custom paths?

### 4. Feedback authoring with markback
Feedback is written in `.mb` (markback) files — a lightweight markup format for annotating LLM outputs with inline human feedback. Each `.mb` file uses `@source` and `@prior` directives to reference the prompt and its output, then contains annotated feedback entries.

Ask me:
- Am I familiar with markback, or do I need an overview of the `.mb` format?
- How will I collect feedback — solo review, team reviews, or automated?

### 5. Feedback processing
Ask me:
- How many independent feedback entries should mention the same problem before it becomes an issue? (Default: 1 — every piece of feedback becomes an issue. Higher values like 2-3 are more conservative and filter out one-off complaints.)

### 6. Workflow mode
Prompterator can work in two modes when improving prompts:

- **Variation mode** (default): Creates new files for each improvement (`001.prompt.txt` → `001a.prompt.txt` → `001b.prompt.txt`). Good for exploring alternatives side by side.
- **Git mode**: Overwrites the original file in place. Good when you use git to track prompt history via diffs.

Ask me:
- Which mode fits my workflow better?

### 7. Critic mode
The Critic can evaluate prompts in two ways:

- **LLM mode** (default): Uses the Critic's LLM to score prompts against rubrics and assertions. No extra setup needed.
- **Script mode**: Runs an external script for deterministic evaluation (e.g. regex checks, API calls, custom scoring). The script receives eval criteria as YAML on stdin and must output YAML results.

Ask me:
- Do I want LLM-based evaluation, script-based, or both?
- If script-based, what's the path to my evaluation script?

### 8. Temperature settings
Temperature controls creativity vs consistency. Typical defaults:
- **Author**: 0.7 (creative prompt drafting)
- **Editor**: 0.7 (creative improvement)
- **Critic**: 0.3 (consistent, repeatable evaluation)

Ask me:
- Am I happy with these defaults, or do I want to adjust any?

### 9. Max tokens
Default is 4096 for all roles.

Ask me:
- Are my prompts or expected outputs especially long? If so, I may want to increase max_tokens (e.g. 8192).

## After collecting all answers

Generate:

1. A complete `prompterator.yaml` with inline comments explaining each section
2. A `.env` file with the required API key variables (values as placeholders)
3. A quick-start checklist showing the first commands to run:
   - `prompterator init` (or just use the generated config)
   - `prompterator status` to verify setup
   - `prompterator feedback` to check feedback parsing
   - `prompterator issues` to consolidate feedback
   - `prompterator evals` to generate evaluations
   - `prompterator calibrate` to validate evals against feedback
   - `prompterator improve` or `prompterator tune` to start improving

Also mention:
- The typical iterative workflow: **feedback → issues → evals → calibrate → improve → test → repeat**
- That `prompterator tune PROMPT` automates the improve→test loop with up to 20 iterations by default
- That `prompterator status -v` shows current workflow state and suggests next steps
