"""Classify projects into 5-digit purpose codes.

    purpose-classify --unlabelled projects.csv --pool labelled.csv --endpoint my-flan-ul2 --output predictions.csv

Each project gets a prompt keyed on its 3-digit sector code: that sector's 5-digit guidelines plus k examples
from the pool rows of the same sector. An answer that is not a valid 5-digit child of the project's sector is
written with status "unparsed" and an empty code, never guessed.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from purpose_coder.backend import FlanUL2SageMaker, TextGenerator
from purpose_coder.data import Example, Project, read_pool, read_unlabelled
from purpose_coder.embeddings import DEFAULT_SBERT_MODEL, Embed, sbert_embedder
from purpose_coder.prompt import build_prompt, parse_answer
from purpose_coder.selection import HandPicked, Random, ReverseOrder, Selector, Similarity
from purpose_coder.taxonomy import Taxonomy, load_taxonomy

# The 2023 runs used k = 5, 7 (hand-picked), 5, 8 and 10 (similarity). Which k is the default is
# not specified in the 2023 drafts; 5 is the smallest value that was run with similarity selection.
DEFAULT_K = 5

OK = "ok"
UNPARSED = "unparsed"
UNKNOWN_SECTOR = "unknown_sector"


@dataclass(frozen=True)
class Prediction:
    code5: str | None
    status: str
    raw_output: str
    example_ids: tuple[str, ...]


def classify_projects(
    projects: Sequence[Project],
    pool: Sequence[Example],
    generator: TextGenerator,
    selector: Selector,
    taxonomy: Taxonomy,
    k: int = DEFAULT_K,
) -> list[Prediction]:
    """One Prediction per project, in order. k = 0 is zero-shot."""
    by_sector: dict[str, list[Example]] = {}
    for example in pool:
        by_sector.setdefault(example.code3, []).append(example)

    predictions = []
    for project in projects:
        try:
            taxonomy.children(project.code3)
        except KeyError:
            predictions.append(Prediction(None, UNKNOWN_SECTOR, "", ()))
            continue
        examples = selector.select(project, by_sector.get(project.code3, []), k) if k > 0 else []
        raw = generator.generate(build_prompt(project, examples, taxonomy))
        code5 = parse_answer(raw, project.code3, taxonomy)
        predictions.append(Prediction(code5, OK if code5 else UNPARSED, raw, tuple(e.id for e in examples)))
    return predictions


def make_selector(
    name: str,
    pool: Sequence[Example],
    embed: Embed | None = None,
    ids: Sequence[str] = (),
    seed: int = 0,
    reverse: bool = False,
) -> Selector:
    if name == "similarity":
        if embed is None:
            raise ValueError("similarity selection needs an embedder")
        selector: Selector = Similarity(embed)
    elif name == "random":
        selector = Random(seed)
    elif name == "hand-picked":
        unknown = sorted(set(ids) - {e.id for e in pool})
        if not ids or unknown:
            raise ValueError(f"hand-picked needs --ids that exist in the pool (missing: {unknown or 'none given'})")
        selector = HandPicked(ids)
    else:
        raise ValueError(f"unknown selection {name!r}")
    return ReverseOrder(selector) if reverse else selector


def write_predictions(path: Path, rows: Sequence[dict[str, str]], predictions: Sequence[Prediction]) -> None:
    extra = ["predicted_code5", "status", "raw_output", "example_ids"]
    fieldnames = [c for c in (rows[0] if rows else {}) if c not in extra] + extra
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, p in zip(rows, predictions):
            writer.writerow(
                {**{c: row[c] for c in fieldnames if c in row}, "predicted_code5": p.code5 or "", "status": p.status,
                 "raw_output": p.raw_output, "example_ids": ";".join(p.example_ids)}
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify projects into OECD 5-digit purpose codes.")
    parser.add_argument("--unlabelled", type=Path, required=True, help="CSV with columns title, description, code3")
    parser.add_argument("--pool", type=Path, required=True, help="labelled examples: title, description, code5[, id]")
    parser.add_argument("--endpoint", required=True, help="name of the SageMaker endpoint serving FLAN-UL2")
    parser.add_argument("--output", type=Path, required=True, help="predictions CSV to write")
    parser.add_argument("--region", help="AWS region of the endpoint (default: your AWS configuration)")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="examples per prompt; 0 is zero-shot")
    parser.add_argument("--selection", choices=["similarity", "random", "hand-picked"], default="similarity")
    parser.add_argument("--ids", nargs="+", default=[], help="example ids for --selection hand-picked")
    parser.add_argument("--reverse", action="store_true", help="reverse the example order")
    parser.add_argument("--seed", type=int, default=0, help="seed for --selection random")
    parser.add_argument("--sbert-model", default=DEFAULT_SBERT_MODEL)
    parser.add_argument("--taxonomy", type=Path, help="purpose-code JSON (default: config/purpose_codes.json)")
    return parser


def main(argv: Sequence[str] | None = None, generator: TextGenerator | None = None, embed: Embed | None = None) -> int:
    args = build_parser().parse_args(argv)
    taxonomy = load_taxonomy(args.taxonomy)
    rows, projects = read_unlabelled(args.unlabelled)
    pool = read_pool(args.pool)
    if args.k == 0:
        selector: Selector = Random(args.seed)  # zero-shot: nothing is selected, so load no embedding model
    else:
        if args.selection == "similarity":
            embed = embed or sbert_embedder(args.sbert_model)
        selector = make_selector(args.selection, pool, embed, args.ids, args.seed, args.reverse)
    generator = generator or FlanUL2SageMaker(args.endpoint, region_name=args.region)
    predictions = classify_projects(projects, pool, generator, selector, taxonomy, args.k)
    write_predictions(args.output, rows, predictions)
    bad = sum(p.status != OK for p in predictions)
    print(f"wrote {len(predictions)} predictions to {args.output} ({bad} unparsed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
