"""End-to-end run on the committed sample: baselines, abstention threshold, LLM row, prompts, figure.

    python src/run_pipeline.py

Writes results/metrics.json, results/sample_prompts.md and assets/figure1_per_code_accuracy.png. The LLM row runs
only when VERTEX_PROJECT, VERTEX_REGION and VERTEX_ENDPOINT_ID point at a deployed endpoint; otherwise it is reported
as not run. LLM output is never cached or written to disk.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.classifiers import FewShotLLM, FewShotPrompter, TfidfKnn, TfidfLogReg  # noqa: E402
from src.data import SAMPLE_SHA256, Record, load_splits, split_statistics  # noqa: E402
from src.evaluate import (  # noqa: E402
    apply_threshold, choose_threshold, evaluate, select_by_validation, selective,
)
from src.figure import make_figure  # noqa: E402
from src.llm import LLMNotConfigured, TextGenerator, VertexEndpointGenerator  # noqa: E402
from src.prompting import approx_tokens  # noqa: E402
from src.retrieval import TfidfRetriever  # noqa: E402
from src.settings import MAX_INPUT_TOKENS, MODEL_ID  # noqa: E402
from src.taxonomy import load_taxonomy  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
FIGURE_PATH = REPO_ROOT / "assets" / "figure1_per_code_accuracy.png"
TARGET_PRECISION = 0.90
SELECTION_METRIC = "macro_f1"
LLM_K = 5
N_SAMPLE_PROMPTS = 3
# key: (label, tuned parameter, grid searched on val, constructor taking one grid value)
BASELINES = {
    "tfidf_logreg": ("TF-IDF logistic regression", "C", [1, 10, 100], lambda value: TfidfLogReg(C=value)),
    "tfidf_knn": ("TF-IDF nearest neighbours", "k", [3, 5, 10], lambda value: TfidfKnn(k=value)),
}


def _round(value: float) -> float:
    return round(float(value), 4)


def _selective_dict(result) -> dict:
    return {"coverage": _round(result.coverage), "precision": _round(result.precision), "answered": result.answered}


def _pick_prompt_records(test: list[Record], prompter: FewShotPrompter) -> list[Record]:
    """One readable test record per health sector, in sector order.

    A record qualifies when its text is 250 to 700 characters and its prompt keeps all k examples inside the token
    budget. The first qualifying record by id is used.
    """
    picked = []
    for sector in sorted({r.code3 for r in test}):
        for record in sorted((r for r in test if r.code3 == sector and 250 <= len(r.text) <= 700), key=lambda r: r.id):
            if len(prompter.prompt_for(record).examples) == LLM_K:
                picked.append(record)
                break
    return picked[:N_SAMPLE_PROMPTS]


def _prompts_markdown(prompter: FewShotPrompter, records: list[Record], taxonomy) -> str:
    parts = [
        "# Sample prompts\n",
        f"These are {len(records)} prompts built from the committed sample by `src/prompting.py`, with the "
        f"k={LLM_K} most similar training projects (TF-IDF retrieval, same 3-digit sector) as input/output "
        "examples. They show what FLAN-UL2 would receive. Nothing was sent to a model to produce this file.\n",
        "The prompt is a reconstruction from the 2023 write-up, not the original prompt text, which was not kept. "
        f"Token counts are character-based estimates, not FLAN-UL2 tokenizer counts; the budget is {MAX_INPUT_TOKENS} "
        "input tokens. Examples run from least to most similar, so the closest one sits next to the project being "
        "coded.\n",
    ]
    for record in records:
        prompt = prompter.prompt_for(record)
        parts.append(
            f"## Test record {record.id}: sector {record.code3} {taxonomy.name(record.code3)}\n\n"
            f"Correct answer (not in the prompt): `{record.code5}` {taxonomy.name(record.code5)}. "
            f"Estimated size: {approx_tokens(prompt.text)} tokens.\n"
        )
        parts.append("```text\n" + prompt.text + "\n```\n")
    return "\n".join(parts)


def _run_baseline(label, parameter, grid, make, splits) -> tuple[dict, dict]:
    selection = select_by_validation(make, grid, splits.train, splits.val, metric=SELECTION_METRIC)
    model = make(selection.best).fit(splits.train)          # refit on train only; val and test are never fitted on
    val_pred, test_pred = model.predict(splits.val), model.predict(splits.test)
    val_metrics, test_metrics = evaluate(val_pred, splits.val), evaluate(test_pred, splits.test)
    threshold = choose_threshold(val_pred, splits.val, TARGET_PRECISION)
    entry = {
        "label": label,
        "selection": {
            "parameter": parameter,
            "metric": selection.metric,
            "chosen": selection.best,
            "candidates": [
                {"value": c.value, "val_accuracy": _round(c.val_accuracy), "val_macro_f1": _round(c.val_macro_f1)}
                for c in selection.candidates
            ],
        },
        "val": {"accuracy": _round(val_metrics.accuracy), "macro_f1": _round(val_metrics.macro_f1)},
        "test": test_metrics.to_dict(),
        "abstention": {
            "chosen_on": "val",
            "target_precision": TARGET_PRECISION,
            "threshold": _round(threshold),
            "val": _selective_dict(selective(apply_threshold(val_pred, threshold), splits.val)),
            "test": _selective_dict(selective(apply_threshold(test_pred, threshold), splits.test)),
        },
    }
    return entry, test_metrics.per_code


def run_full_pipeline(
    results_dir: Path = RESULTS_DIR,
    figure_path: Path = FIGURE_PATH,
    generator: TextGenerator | None = None,
    write: bool = True,
) -> dict:
    taxonomy = load_taxonomy()
    splits = load_splits()
    result: dict = {
        "task": "predict the 5-digit OECD purpose code given the 3-digit sector (health sectors 121, 122, 123)",
        "data": {
            "sample_sha256": SAMPLE_SHA256,
            "train": len(splits.train),
            "val": len(splits.val),
            "test": len(splits.test),
            "test_per_code": dict(sorted(Counter(r.code5 for r in splits.test).items())),
            "split": split_statistics(splits),
        },
        "baselines": {},
    }

    accuracy_by_model = {}
    for key, (label, parameter, grid, make) in BASELINES.items():
        result["baselines"][key], accuracy_by_model[label] = _run_baseline(label, parameter, grid, make, splits)

    not_run_reason = None
    if generator is None:
        try:
            generator = VertexEndpointGenerator.from_env()
        except LLMNotConfigured as error:
            not_run_reason = f"{error}; deploy FLAN-UL2 with deploy/vertex/deploy_flan_ul2.py first"
    if generator is None:
        result["llm"] = {"status": "not run", "reason": not_run_reason, "intended_model": MODEL_ID, "k": LLM_K}
    else:
        few_shot = FewShotLLM(generator, TfidfRetriever, k=LLM_K, taxonomy=taxonomy).fit(splits.train)
        metrics = evaluate(few_shot.predict(splits.test), splits.test)
        result["llm"] = {
            "status": "run", "model_id": getattr(generator, "model_id", "unknown"), "k": LLM_K,
            "test": metrics.to_dict(), "parse_failures": few_shot.parse_failures,
        }

    if write:
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        prompter = FewShotPrompter(TfidfRetriever, k=LLM_K, taxonomy=taxonomy).fit(splits.train)
        prompts = _prompts_markdown(prompter, _pick_prompt_records(splits.test, prompter), taxonomy)
        (results_dir / "sample_prompts.md").write_text(prompts, encoding="utf-8")
        codes = result["data"]["test_per_code"]
        sector_names = {code[:3]: taxonomy.name(code[:3]) for code in codes}
        make_figure(accuracy_by_model, codes, {c: taxonomy.name(c) for c in codes}, sector_names, figure_path)
    return result


def main() -> None:
    result = run_full_pipeline()
    data = result["data"]
    print(f"train {data['train']}, val {data['val']}, test {data['test']}")
    print(f"near-duplicate groups: {data['split']}")
    for entry in result["baselines"].values():
        test, cut, chosen = entry["test"], entry["abstention"], entry["selection"]
        print(
            f"{entry['label']} ({chosen['parameter']}={chosen['chosen']}, chosen on val): "
            f"test accuracy {test['accuracy']:.4f}, macro-F1 {test['macro_f1']:.4f}; "
            f"threshold {cut['threshold']} (chosen on val) -> coverage {cut['test']['coverage']:.4f}, "
            f"precision {cut['test']['precision']:.4f}"
        )
    llm = result["llm"]
    print("LLM: " + (llm["reason"] if llm["status"] == "not run" else f"accuracy {llm['test']['accuracy']:.4f}"))


if __name__ == "__main__":
    main()
