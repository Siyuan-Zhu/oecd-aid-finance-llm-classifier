"""Stand-ins for the model and the embedder, plus synthetic projects. All texts here are invented for the tests."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from purpose_coder.data import Example, Project


class FakeGenerator:
    """A TextGenerator that returns scripted answers and records every prompt it was sent."""

    def __init__(self, answers: Sequence[str] | str):
        self._answers = [answers] if isinstance(answers, str) else list(answers)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._answers[min(len(self.prompts), len(self._answers)) - 1]


def fake_embed(texts: Sequence[str]) -> np.ndarray:
    """Bag of words hashed into 64 dimensions: texts that share words are close, texts that share none are not."""
    out = np.zeros((len(texts), 64))
    for row, text in enumerate(texts):
        for word in text.lower().replace(";", " ").replace(",", " ").replace(".", " ").split():
            out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
    return out


def example(id: str, title: str, description: str, code5: str) -> Example:
    return Example(id, title, description, code5)


# Synthetic basic-health examples (sector 122), three topics with two examples each.
POOL = [
    example("v1", "Vaccination drive", "Teams vaccinate children against measles in rural villages.", "12250"),
    example("v2", "Immunisation outreach", "Mobile clinics vaccinate children against polio and measles.", "12250"),
    example("c1", "Clinic building", "Builds a district clinic with wards and a pharmacy building.", "12230"),
    example("c2", "New dispensary", "Constructs a dispensary building and a maternity ward.", "12230"),
    example("n1", "Nurse training", "Trains nurses and midwives in basic care for health workers.", "12281"),
    example("n2", "Midwife course", "A course that trains health workers, nurses and midwives.", "12281"),
]

PROJECT = Project("Measles campaign", "Vaccinate children against measles across several districts.", "122")
