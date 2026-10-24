"""Test doubles. The TextGenerator fake lives here, not in src/: the shipped code never fabricates LLM output."""

from __future__ import annotations

from src.data import Record


def make_record(id, text, code5="12262", code3=None):
    """A labelled record; the sector defaults to the first three digits of the purpose code."""
    return Record(id=str(id), text=text, code3=code3 or code5[:3], code5=code5, year=2024, donor="D")


class FakeTextGenerator:
    """Returns canned answers. `answers` is a list (cycled) or a function from prompt to answer."""

    def __init__(self, answers):
        self._answers = answers
        self.prompts: list[str] = []

    def generate(self, prompts: list[str]) -> list[str]:
        self.prompts.extend(prompts)
        if callable(self._answers):
            return [self._answers(p) for p in prompts]
        return [self._answers[(len(self.prompts) - len(prompts) + i) % len(self._answers)] for i in range(len(prompts))]


class FakeEndpoint:
    """Stands in for a Vertex AI Endpoint: records raw_predict calls and replies with TGI's /generate shape."""

    def __init__(self, reply="12262"):
        self.reply = reply
        self.calls: list[tuple[bytes, dict]] = []

    def raw_predict(self, body, headers):
        self.calls.append((body, headers))
        reply = self.reply

        class Response:
            def json(self):
                return {"generated_text": reply}

        return Response()
