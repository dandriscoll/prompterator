"""Shared test fixtures."""

import pytest

from prompterator.models.eval import Eval, EvalFile, EvalRubric
from prompterator.models.feedback import Feedback, FeedbackEntry
from prompterator.models.issue import Issue, IssueEvidence, IssueFile
from prompterator.models.result import EvalResult, ResultSummary
from prompterator.runners.llm import LLMClient


class MockLLMClient:
    """Deterministic mock LLM client for testing."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = list(responses) if responses else []
        self.calls: list[dict] = []
        self._call_index = 0

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int = 300,
    ) -> str:
        self.calls.append({
            "prompt": prompt,
            "system": system,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        if self._call_index < len(self.responses):
            response = self.responses[self._call_index]
            self._call_index += 1
            return response
        return "RESULT: PASS\nOVERALL: PASS\nSCORE: 1.0"


@pytest.fixture
def mock_llm():
    """Create a mock LLM client with default pass-all responses."""
    return MockLLMClient()


@pytest.fixture
def sample_feedback_list():
    """Sample feedback entries from multiple sources."""
    return [
        Feedback(
            source_file="review1.mb",
            prompt_ref="test.prompt.txt",
            entries=[
                FeedbackEntry(text="Instructions are vague and unclear"),
                FeedbackEntry(text="No examples provided"),
            ],
        ),
        Feedback(
            source_file="review2.mb",
            prompt_ref="test.prompt.txt",
            entries=[
                FeedbackEntry(text="Multiple interpretations possible due to ambiguous wording"),
                FeedbackEntry(text="Tone is too formal for the audience"),
            ],
        ),
        Feedback(
            source_file="review3.mb",
            prompt_ref="test.prompt.txt",
            entries=[
                FeedbackEntry(text="Overall structure is confusing"),
            ],
        ),
    ]


@pytest.fixture
def sample_issue_file():
    """Sample issue file with two issues."""
    return IssueFile(
        prompt_ref="test.prompt.txt",
        issues=[
            Issue(
                id="issue-test-01",
                category="unclear-instructions",
                severity="high",
                summary="Instructions are vague and allow multiple interpretations",
                evidence=[
                    IssueEvidence(source="review1.mb", feedback="Instructions are vague and unclear"),
                    IssueEvidence(source="review2.mb", feedback="Multiple interpretations possible due to ambiguous wording"),
                    IssueEvidence(source="review3.mb", feedback="Overall structure is confusing"),
                ],
            ),
            Issue(
                id="issue-test-02",
                category="missing-examples",
                severity="low",
                summary="No examples provided to illustrate expected behavior",
                evidence=[
                    IssueEvidence(source="review1.mb", feedback="No examples provided"),
                ],
            ),
        ],
    )


@pytest.fixture
def sample_eval_file():
    """Sample eval file with rubric evals."""
    return EvalFile(
        prompt_ref="test.prompt.txt",
        evals=[
            Eval(
                id="eval-test-unclear-instructions-01",
                type="rubric",
                issue_ref="issue-test-01",
                description="Verify unclear-instructions improvements",
                rubric=EvalRubric(
                    criteria=[
                        "Prompt addresses: Instructions are vague and unclear",
                        "Prompt addresses: Multiple interpretations possible due to ambiguous wording",
                        "Prompt addresses: Overall structure is confusing",
                    ],
                    scoring="all_required",
                ),
            ),
            Eval(
                id="eval-test-missing-examples-02",
                type="rubric",
                issue_ref="issue-test-02",
                description="Verify missing-examples improvements",
                rubric=EvalRubric(
                    criteria=[
                        "All required information is present",
                        "Edge cases are addressed",
                        "No missing instructions",
                    ],
                    scoring="any_required",
                ),
            ),
        ],
    )


@pytest.fixture
def sample_eval_results():
    """Sample eval results with mixed pass/fail."""
    return [
        EvalResult(eval_id="eval-test-unclear-instructions-01", passed=False, score=0.33, details="1/3 criteria met"),
        EvalResult(eval_id="eval-test-missing-examples-02", passed=True, score=1.0, details="All criteria met"),
    ]


@pytest.fixture
def sample_prompt_text():
    """Sample prompt text for testing."""
    return "You are a helpful assistant. Answer the user's question clearly and concisely."
