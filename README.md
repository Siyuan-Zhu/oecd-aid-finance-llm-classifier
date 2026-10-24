# OECD aid purpose-code classifier

This classifier reads the text of a real OECD aid project and assigns its 5-digit purpose code, which says what the money pays for. On 645 held-out health projects from 2024, a TF-IDF logistic regression (a standard word-based model) gets 74.1% right across 17 codes. Answering only when confident, it handles 59% of projects at 88% precision and leaves the rest for a person.

It shows how I clean a 370,264-row government file into a leakage-safe dataset, choose models on validation data, route unsure cases to human review, design a few-shot LLM prompt, and write cloud deployment code with a cost estimate. The LLM classifier is built and tested but not yet run on a real model.

![Test accuracy by purpose code for the two baselines](assets/figure1_per_code_accuracy.png)

## Why this project

Every aid project reported to the OECD's Creditor Reporting System (CRS) has a purpose code. The first three digits give the sector (122 Basic Health), all five the purpose (12262 Malaria control). Donors pick the codes themselves, and checking them by hand is slow work.

In 2023, at AidData (William & Mary), I did this task with FLAN-UL2, an open 20-billion-parameter model from Google, prompted with the OECD code definitions and a few already-coded examples. On 481 hand-coded projects my best accuracy was 96.84%. That data is private and the prompt was not kept, so those numbers cannot be reproduced here and are not comparable with the ones below.

This project rebuilds the approach on public data and measures the baseline an LLM must beat. The same pattern fits finance work such as coding transactions: a model pre-fills the code and a person checks the uncertain ones.

## What I built

- A clean dataset. `scripts/build_sample.py` filters the OECD's 2024 bulk file to English health projects, drops duplicates and texts filed under conflicting codes, and caps each code at 250 rows: 3,227 projects over 17 codes ([filter counts](DATA.md#filters-and-row-counts)).
- A leakage-safe split. CRS repeats text heavily, and a leaked near-copy inflates every score. I grouped near-duplicate texts (TF-IDF cosine similarity of 0.8 or more) and kept each group in one split, which a test checks.
- Two baselines, TF-IDF logistic regression and a nearest-neighbour vote, each tuned on validation only and scored on the test split once.
- A confidence threshold for human review, set on validation as the lowest confidence that reaches 90% precision.
- A few-shot LLM classifier. `src/prompting.py` rebuilds the 2023 prompt from its write-up: the sector's code definitions, the 5 most similar training projects as examples, then the project, trimmed to FLAN-UL2's 2,048-token input limit. It is tested against a stand-in model, and [results/sample_prompts.md](results/sample_prompts.md) shows three real prompts.
- Cloud deployment code. `deploy/vertex/deploy_flan_ul2.py` deploys FLAN-UL2 to Google Cloud Vertex AI and tears it down; its default dry run prints the config and hourly cost. It has not been run, and [its README](deploy/vertex/README.md) lists what is unverified.

## Results

| Model | Test accuracy | Macro-F1 | Answering only when confident |
| :--- | ---: | ---: | :--- |
| TF-IDF logistic regression | 74.1% | 0.672 | 59% of projects, 88.2% precision |
| TF-IDF nearest neighbours | 70.7% | 0.642 | 39% of projects, 95.7% precision |
| Few-shot FLAN-UL2 | not run yet | | |

Macro-F1 averages over codes, so small codes count as much as large ones.

Codes with distinctive words are easy: malaria, tuberculosis and mental health score 82% or more with both models. Small codes and close neighbours are hard: health personnel development scores under 13% (16 test projects), and infectious disease control is often coded as COVID-19 control. The sample also drops ambiguous texts, so it is easier than the raw data.

The FLAN-UL2 row needs a GPU endpoint, about $2.30 an hour on Vertex AI's default two-GPU machine at list price. With one set up, the pipeline fills in the row on the same test split. An LLM is worth that only if it beats the 74.1% TF-IDF reaches in under a minute on a laptop. Full tables, method and limits are in [docs/METHOD.md](docs/METHOD.md).

## Tech stack

Python, pandas, scikit-learn, matplotlib, pytest, GitHub Actions, FLAN-UL2, Hugging Face text-generation-inference, Google Cloud Vertex AI.

## What I'd do next

- Deploy FLAN-UL2, measure the LLM row, then tear the endpoint down.
- Try SBERT example retrieval, as in 2023 (written, not yet in the default run).
- Predict the 3-digit sector too; in practice it is often unknown.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest
python src/run_pipeline.py   # no download; rewrites results/ and the figure
pytest -q
```

To rebuild the data or run the LLM row, see [docs/METHOD.md](docs/METHOD.md#other-commands). MIT licensed; the data is the OECD's, cited in [DATA.md](DATA.md).
