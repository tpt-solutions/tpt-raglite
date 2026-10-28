"""Tests for the prompt and citation logic. They use a fake database, so no model is needed.

Run with: python -m pytest templates/docs-chatbot
"""

from dataclasses import dataclass

from chatbot import NO_ANSWER, answer, build_prompt, cited_numbers, find_passages


@dataclass
class FakeResult:
    source: str
    text: str
    score: float


class FakeDB:
    def __init__(self, results):
        self.results = results
        self.asked_top_k = None

    def query(self, text, top_k=5):
        self.asked_top_k = top_k
        return self.results[:top_k]


DOCS = [
    FakeResult("docs/keys.md", "Rotate the API key from the settings page.", 0.9),
    FakeResult("docs/limits.md", "Requests are limited to 100 per minute.", 0.5),
    FakeResult("docs/billing.md", "Invoices are sent on the first day.", 0.1),
]


def test_prompt_numbers_passages_and_names_sources():
    passages = find_passages(FakeDB(DOCS), "rotate key", top_k=2)
    prompt = build_prompt("How do I rotate the key?", passages)

    assert "[1] (from docs/keys.md)" in prompt
    assert "[2] (from docs/limits.md)" in prompt
    assert prompt.rstrip().endswith("Answer:")
    assert "How do I rotate the key?" in prompt


def test_cited_numbers_keeps_first_appearance_order_and_drops_duplicates():
    assert cited_numbers("Use the page [2] and then [1][2].") == [2, 1]
    assert cited_numbers("No citations here.") == []


def test_answer_returns_only_the_passages_the_answer_cites():
    db = FakeDB(DOCS)

    def ask(prompt):
        assert "[1] (from docs/keys.md)" in prompt
        return "Open the settings page and rotate it [1]."

    text, cited = answer(db, "rotate key", ask=ask, top_k=3)

    assert text.startswith("Open the settings page")
    assert [p.source for p in cited] == ["docs/keys.md"]
    assert db.asked_top_k == 3


def test_answer_without_matches_skips_the_model():
    def ask(prompt):
        raise AssertionError("the model must not be called without passages")

    text, cited = answer(FakeDB([]), "anything", ask=ask)
    assert text == NO_ANSWER
    assert cited == []


def test_citation_numbers_that_do_not_exist_are_ignored():
    def ask(prompt):
        return "Something [9] and [1]."

    _, cited = answer(FakeDB(DOCS), "rotate key", ask=ask, top_k=2)
    assert [p.number for p in cited] == [1]
