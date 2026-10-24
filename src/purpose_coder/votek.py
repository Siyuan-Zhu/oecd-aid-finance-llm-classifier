"""Vote-K selective annotation (Su et al., 2022): pick which unlabelled projects to hand-label.

The 2023 plan was to pick fewer than 100 diverse, representative projects per sector to annotate, then use them
as the example pool for similarity-based selection. About 100 projects were annotated instead of about 1,000.

The method is the graph-voting step of Vote-K. Build a directed graph over the projects where each project
points to its k nearest neighbours by embedding cosine similarity. Then choose projects one at a time. Each
project votes for its neighbours; a project's vote shrinks by a factor of rho for every neighbour of it that is
already chosen, so later picks cover parts of the data the earlier picks did not.

Not implemented: the second stage in Su et al., which ranks the rest by a language model's confidence. The
2023 drafts describe only the diversity step.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from purpose_coder.data import Project, read_unlabelled
from purpose_coder.embeddings import DEFAULT_SBERT_MODEL, Embed, normalise, sbert_embedder

# Annotation budget from the 2023 work: about 100 hand-labelled projects.
DEFAULT_BUDGET = 100
# Graph size k and discount rho are not specified in the 2023 drafts. 10 and 10 are chosen defaults.
DEFAULT_GRAPH_K = 10
DEFAULT_RHO = 10.0


def vote_k(embeddings: np.ndarray, budget: int, k: int = DEFAULT_GRAPH_K, rho: float = DEFAULT_RHO) -> list[int]:
    """Row indices of `embeddings` to annotate, in the order Vote-K picks them. Ties go to the lower index."""
    vectors = normalise(np.asarray(embeddings, dtype=float))
    n = len(vectors)
    budget = max(0, min(budget, n))
    if budget == 0:
        return []
    neighbours_per_node = min(k, n - 1)
    similarity = vectors @ vectors.T
    np.fill_diagonal(similarity, -np.inf)
    neighbours = np.argsort(-similarity, axis=1, kind="stable")[:, :neighbours_per_node]  # row v: who v points to

    chosen: list[int] = []
    chosen_count = np.zeros(n)  # for each node, how many of its neighbours are already chosen
    for _ in range(budget):
        vote = rho ** (-chosen_count)
        score = np.zeros(n)
        np.add.at(score, neighbours.ravel(), np.repeat(vote, neighbours_per_node))
        score[chosen] = -np.inf
        best = int(np.argmax(score))
        chosen.append(best)
        chosen_count += (neighbours == best).any(axis=1)
    return chosen


def select_for_annotation(
    projects: Sequence[Project],
    budget: int,
    embed: Embed,
    k: int = DEFAULT_GRAPH_K,
    rho: float = DEFAULT_RHO,
) -> list[int]:
    """Indices into `projects` of the ones to hand-label."""
    if not projects:
        return []
    return vote_k(np.asarray(embed([p.text for p in projects])), budget, k=k, rho=rho)


def main(argv: Sequence[str] | None = None, embed: Embed | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pick which unlabelled projects to hand-label (Vote-K).")
    parser.add_argument("--unlabelled", type=Path, required=True, help="CSV with title, description, code3")
    parser.add_argument("--output", type=Path, required=True, help="CSV of the chosen rows, in pick order")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--code3", help="only consider projects in this 3-digit sector")
    parser.add_argument("--sbert-model", default=DEFAULT_SBERT_MODEL)
    args = parser.parse_args(argv)

    rows, projects = read_unlabelled(args.unlabelled)
    keep = [i for i, p in enumerate(projects) if args.code3 in (None, p.code3)]
    embed = embed or sbert_embedder(args.sbert_model)
    picked = select_for_annotation([projects[i] for i in keep], args.budget, embed)
    chosen_rows = [rows[keep[i]] for i in picked]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["title", "description", "code3"])
        writer.writeheader()
        writer.writerows(chosen_rows)
    print(f"wrote {len(chosen_rows)} projects to annotate to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
