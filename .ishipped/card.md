---
title: "Prompterator"
summary: "CLI that turns user feedback into better prompts — automatically."
shipped: 2026-03-14
tags: [cli, llm, prompts, python, developer-tools]
links:
  - label: "Install"
    url: "https://pypi.org/project/prompterator/"
    primary: true
  - label: "Source"
    url: "https://github.com/dandriscoll/prompterator"
---

## What is it?

Prompterator is a CLI tool that closes the loop between user feedback and prompt quality. Point it at feedback files and it consolidates issues, generates evaluations, improves your prompts, and tests the results — all driven by LLMs playing distinct roles (Author, Editor, Critic).

## Key Features

- **Automated tune loop** — Iteratively improves prompts until evals pass, with `--focus` to target a specific eval and `--aggressive` to force strategies when progress stalls
- **Feedback → Issues → Evals pipeline** — Clusters raw user feedback into actionable issues, then generates structured evaluation criteria automatically
- **Multi-LLM architecture** — Separates generation (Author), refinement (Editor), and judgment (Critic) into distinct roles with independent model and temperature settings
- **Git mode** — Overwrites prompts in-place so you review improvements as a clean `git diff`
- **Provider-agnostic** — Works with Anthropic, OpenAI, or any supported LLM backend

---

[View on ishipped.io](https://ishipped.io/card/dandriscoll/prompterator)
