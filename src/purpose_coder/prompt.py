"""The prompt for one project, laid out as in the 2023 FLAN-UL2 work.

The layout, from David's 2023 prompt notes:

    Task: <instruction>

    Criteria and clarifications:

    <code> <name>; clarification: <OECD clarification>      (one per 5-digit code under the project's 3-digit sector)

    input:(title and description): <title> ; <description>   (one block per in-context example)
    Answer: <5-digit code>

    New:
    input(title and description): <title> ; <description>
    answer:

Differences from the 2023 text: its spelling slips ("Criterias", "clearification") are corrected, and the
guideline wording comes from the official OECD list in config/purpose_codes.json instead of hand-edited
clarifications. With no examples the prompt is the zero-shot prompt.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from purpose_coder.data import Example, Project
from purpose_coder.taxonomy import Taxonomy

INSTRUCTION = (
    "Task: Given the project title and description, which of the following criteria does the project "
    "relate or contribute to the most? Return the relevant code numbers only."
)


def format_guidelines(code3: str, taxonomy: Taxonomy) -> str:
    """One line per 5-digit code under the sector, in the OECD list's order. Raises KeyError for unknown sectors."""
    lines = []
    for child in taxonomy.children(code3):
        line = f"{child.code} {child.name}"
        if child.description:
            line += f"; clarification: {child.description}"
        lines.append(line)
    return "\n\n".join(lines)


def build_prompt(project: Project, examples: Sequence[Example], taxonomy: Taxonomy) -> str:
    """The prompt for `project`. Examples are shown in the order given."""
    parts = [
        INSTRUCTION,
        "Criteria and clarifications:\n\n" + format_guidelines(project.code3, taxonomy),
    ]
    parts += [f"input:(title and description): {e.text}\nAnswer: {e.code5}" for e in examples]
    parts.append(f"New:\ninput(title and description): {project.text}\nanswer:")
    return "\n\n".join(parts)


_FIVE_DIGITS = re.compile(r"(?<!\d)\d{5}(?!\d)")


def parse_answer(output: str, code3: str, taxonomy: Taxonomy) -> str | None:
    """The 5-digit code in a model answer, or None.

    Accepted: exactly one distinct 5-digit number, and it is a child of `code3` in the OECD list. Anything else
    (no code, two different codes, a code from another sector, a made-up code) is None and gets marked unparsed.
    """
    found = set(_FIVE_DIGITS.findall(output))
    if len(found) != 1:
        return None
    (code,) = found
    try:
        valid = {c.code for c in taxonomy.children(code3)}
    except KeyError:
        return None
    return code if code in valid else None
