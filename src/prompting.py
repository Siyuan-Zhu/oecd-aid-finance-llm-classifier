"""The few-shot prompt sent to the LLM.

This is a reconstruction. The prompt used in the 2023 FLAN-UL2 work was not kept; this follows what the 2023
write-up describes: the task framed as reading comprehension plus text categorisation, the OECD criteria for the
sector's 5-digit codes in the instruction, and k labelled examples as input/output pairs chosen by similarity to
the project being coded. The most similar example goes last, next to the project to classify.

FLAN-UL2 takes 2048 input tokens. `build_prompt` drops the least similar examples until the prompt fits. Token
counts are approximate unless the caller passes `count_tokens` backed by the real T5 tokenizer (for example
`lambda s: len(tokenizer(s).input_ids)` with a `transformers` tokenizer); the default estimate is deliberately high.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from src.data import Record
from src.settings import MAX_INPUT_TOKENS, RESERVED_TOKENS
from src.taxonomy import PurposeCode, Taxonomy

CHARS_PER_TOKEN = 3.5    # English prose is about 4 characters per SentencePiece token; 3.5 errs on the safe side


def approx_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def _instruction(sector_code: str, sector_name: str, candidates: list[PurposeCode]) -> str:
    criteria = "\n".join(
        f"{c.code} - {c.name}" + (f": {c.description}" if c.description else "") for c in candidates
    )
    return (
        "This is a reading comprehension and text categorisation task. Read the title and description of an "
        "aid project and decide which OECD Creditor Reporting System purpose code describes it.\n"
        f"The project belongs to sector {sector_code} ({sector_name}). "
        "Choose exactly one of these 5-digit purpose codes:\n\n"
        f"{criteria}\n\n"
        "Answer with the 5-digit code only.\n"
    )


def _example(record: Record) -> str:
    return f"Project: {record.text}\nPurpose code: {record.code5}\n"


def _query(text: str) -> str:
    return f"Project: {text}\nPurpose code:"


def build_prompt(
    record: Record,
    examples: list[Record],
    taxonomy: Taxonomy,
    candidates: list[PurposeCode] | None = None,
    max_tokens: int = MAX_INPUT_TOKENS,
    reserve_tokens: int = RESERVED_TOKENS,
    count_tokens: Callable[[str], int] = approx_tokens,
) -> str:
    """Prompt for `record`. `examples` come most similar first (retriever order) and are shown in reverse.

    `candidates` defaults to the sector's core (non-voluntary) codes. If the prompt is over budget, the least
    similar examples are dropped; if it is still over with no examples, the project text is cut at a word boundary.
    """
    if candidates is None:
        candidates = [c for c in taxonomy.children(record.code3) if not c.voluntary]
    instruction = _instruction(record.code3, taxonomy.name(record.code3), candidates)
    budget = max_tokens - reserve_tokens

    def render(kept: list[Record], query_text: str) -> str:
        shots = "\n".join(_example(e) for e in reversed(kept))
        return instruction + "\n" + (shots + "\n" if shots else "") + _query(query_text)

    kept = list(examples)
    while kept and count_tokens(render(kept, record.text)) > budget:
        kept.pop()
    if count_tokens(render(kept, record.text)) <= budget:
        return render(kept, record.text)

    words = record.text.split()
    low, high = 0, len(words)
    while low < high:
        mid = (low + high + 1) // 2
        if count_tokens(render([], " ".join(words[:mid]))) <= budget:
            low = mid
        else:
            high = mid - 1
    return render([], " ".join(words[:low]))
