"""End-to-end run on the committed sample: baselines, abstention threshold, LLM row, prompts, figure.

    python src/run_pipeline.py

Writes results/metrics.json, results/sample_prompts.md and assets/figure1_per_code_accuracy.png. The LLM row runs
only when VERTEX_PROJECT, VERTEX_REGION and VERTEX_ENDPOINT_ID point at a deployed endpoint; otherwise it is reported
as not run. LLM output is never cached or written to disk.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.classifiers import FewShotLLM, TfidfKnn, TfidfLogReg  # noqa: E402
from src.data import SAMPLE_SHA256, Record, load_splits  # noqa: E402
from src.evaluate import apply_threshold, choose_threshold, evaluate, selective  # noqa: E402
from src.figure import make_figure  # noqa: E402
from src.llm import LLMNotConfigured, TextGenerator, UnconfiguredGenerator, VertexEndpointGenerator  # noqa: E402
from src.retrieval import TfidfRetriever  # noqa: E402
from src.taxonomy import load_taxonomy  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
FIGURE_PATH = REPO_ROOT / "assets" / "figure1_per_code_accuracy.png"
TARGET_PRECISION = 0.90
LLM_K = 5
N_SAMPLE_PROMPTS = 3
BASELINES = {"tfidf_logreg": ("TF-IDF logistic regression", TfidfLogReg), "tfidf_knn": ("TF-IDF nearest neighbours", TfidfKnn)}


def _round(value: float) -> float:
    return round(float(value), 4)


def _selective_dict(result) -> dict:
    return {"coverage": _round(result.coverage), "precision": _round(result.precision), "answered": result.answered}


def _pick_prompt_records(test: list[Record], llm: FewShotLLM) -> list[Record]:
    """One readable test record per health sector, in sector order.

    A record qualifies when its text is 250 to 700 characters and its prompt keeps all k examples inside the token
    budget. The first qualifying record by id is used.
    """
    picked = []
    for sector in sorted({r.code3 for r in test}):
        for record in sorted((r for r in test if r.code3 == sector and 250 <= len(r.text) <= 700), key=lambda r: r.id):
            shots = len(re.findall(r"^Purpose code: \d{5}$", llm.prompt_for(record), flags=re.M))
            if shots == LLM_K:
                picked.append(record)
                break
    return picked[:N_SAMPLE_PROMPTS]


def _prompts_markdown(llm: FewShotLLM, records: list[Record], taxonomy) -> str:
    from src.prompting import approx_tokens

    parts = [
        "# Sample prompts\n",
        f"These are {len(records)} prompts built from the committed sample by `src/prompting.py`, with the k={LLM_K} most "
        "similar training projects (TF-IDF retrieval, same 3-digit sector) as input/output examples. They show what "
        "FLAN-UL2 would receive. Nothing was sent to a model to produce this file.\n",
        "The prompt is a reconstruction from the 2023 write-up, not the original prompt text, which was not kept. "
        "Token counts are character-based estimates, not FLAN-UL2 tokenizer counts; the budget is 2048 input tokens. "
        "Examples run from least to most similar, so the closest one sits next to the project being coded.\n",
    ]
    for record in records:
        prompt = llm.prompt_for(record)
        parts.append(
            f"## Test record {record.id}: sector {record.code3} {taxonomy.name(record.code3)}\n\n"
            f"Correct answer (not in the prompt): `{record.code5}` {taxonomy.name(record.code5)}. "
            f"Estimated size: {approx_tokens(prompt)} tokens.\n"
        )
        parts.append("```text\n" + prompt + "\n```\n")
    return "\n".join(parts)


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
        },
        "baselines": {},
    }

    accuracy_by_model = {}
    for key, (label, factory) in BASELINES.items():
        model = factory().fit(splits.train)
        val_pred, test_pred = model.predict(splits.val), model.predict(splits.test)
        val_metrics, test_metrics = evaluate(val_pred, splits.val), evaluate(test_pred, splits.test)
        threshold = choose_threshold(val_pred, splits.val, TARGET_PRECISION)
        result["baselines"][key] = {
            "label": label,
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
        accuracy_by_model[label] = test_metrics.per_code

    reason = (
        "no endpoint configured: set VERTEX_PROJECT, VERTEX_REGION and VERTEX_ENDPOINT_ID after deploying "
        "FLAN-UL2 with deploy/vertex/deploy_flan_ul2.py"
    )
    if generator is None:
        try:
            generator = VertexEndpointGenerator.from_env()
        except LLMNotConfigured:
            generator = UnconfiguredGenerator(reason)
    few_shot = FewShotLLM(generator, TfidfRetriever, k=LLM_K, taxonomy=taxonomy).fit(splits.train)
    if isinstance(generator, UnconfiguredGenerator):
        result["llm"] = {"status": "not run", "reason": generator.reason, "intended_model": "google/flan-ul2", "k": LLM_K}
    else:
        metrics = evaluate(few_shot.predict(splits.test), splits.test)
        result["llm"] = {
            "status": "run", "intended_model": "google/flan-ul2", "k": LLM_K,
            "test": metrics.to_dict(), "parse_failures": few_shot.parse_failures,
        }

    if write:
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        prompts = _prompts_markdown(few_shot, _pick_prompt_records(splits.test, few_shot), taxonomy)
        (results_dir / "sample_prompts.md").write_text(prompts, encoding="utf-8")
        codes = result["data"]["test_per_code"]
        sector_names = {code[:3]: taxonomy.name(code[:3]) for code in codes}
        make_figure(accuracy_by_model, codes, {c: taxonomy.name(c) for c in codes}, sector_names, figure_path)
    return result


def main() -> None:
    result = run_full_pipeline()
    print(f"train {result['data']['train']}, val {result['data']['val']}, test {result['data']['test']}")
    for key, entry in result["baselines"].items():
        test, cut = entry["test"], entry["abstention"]
        print(
            f"{entry['label']}: test accuracy {test['accuracy']:.4f}, macro-F1 {test['macro_f1']:.4f}; "
            f"threshold {cut['threshold']} (chosen on val) -> coverage {cut['test']['coverage']:.4f}, "
            f"precision {cut['test']['precision']:.4f}"
        )
    llm = result["llm"]
    print("LLM: " + (llm["reason"] if llm["status"] == "not run" else f"accuracy {llm['test']['accuracy']:.4f}"))


if __name__ == "__main__":
    main()
