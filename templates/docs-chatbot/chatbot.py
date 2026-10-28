"""Folder in, Q&A out: retrieval with numbered citations for a documentation chatbot.

tpt-raglite finds the passages. This module builds a prompt that asks the model to answer
only from those passages and to cite them as [1], [2], and so on. The model call is left to
you: pass any function that takes a prompt string and returns the answer text.

    from tpt_raglite import RAGLite
    from chatbot import answer

    db = RAGLite("./docs_db")
    db.add_directory("./docs")
    text, cited = answer(db, "How do I rotate the API key?", ask=my_llm_call)
"""

import re
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence


class Searchable(Protocol):
    """The part of RAGLite this module uses. Tests can pass a fake."""

    def query(self, text: str, top_k: int = 5) -> Sequence[object]: ...


@dataclass(frozen=True)
class Passage:
    number: int
    source: str
    text: str
    score: float


NO_ANSWER = "I could not find anything about that in the documents."

INSTRUCTIONS = (
    "Answer the question using only the numbered passages below. "
    "Cite the passages you use in square brackets, such as [1] or [2][3]. "
    "If the passages do not contain the answer, say that you do not know."
)


def build_prompt(question: str, passages: Sequence[Passage]) -> str:
    blocks = [f"[{p.number}] (from {p.source})\n{p.text}" for p in passages]
    return "\n\n".join(
        [INSTRUCTIONS, "Passages:", *blocks, f"Question: {question}", "Answer:"]
    )


def find_passages(db: Searchable, question: str, top_k: int = 4) -> list[Passage]:
    results = db.query(question, top_k=top_k)
    return [
        Passage(number=i, source=r.source, text=r.text, score=r.score)
        for i, r in enumerate(results, start=1)
    ]


def cited_numbers(answer_text: str) -> list[int]:
    """Passage numbers the answer cites, in order of first appearance."""
    seen: list[int] = []
    for match in re.finditer(r"\[(\d+)\]", answer_text):
        number = int(match.group(1))
        if number not in seen:
            seen.append(number)
    return seen


def answer(
    db: Searchable,
    question: str,
    ask: Callable[[str], str],
    top_k: int = 4,
) -> tuple[str, list[Passage]]:
    """Returns the model's answer and the passages it cited. Passages the answer does not
    cite are dropped from the second value, so a UI shows only the sources that were used."""
    passages = find_passages(db, question, top_k)
    if not passages:
        return NO_ANSWER, []

    text = ask(build_prompt(question, passages))
    by_number = {p.number: p for p in passages}
    cited = [by_number[n] for n in cited_numbers(text) if n in by_number]
    return text, cited
