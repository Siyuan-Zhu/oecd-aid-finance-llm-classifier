# Deploying FLAN-UL2 on Vertex AI

`deploy_flan_ul2.py` uploads `google/flan-ul2` to Vertex AI behind Hugging Face's text-generation-inference (TGI)
container and deploys it to a GPU endpoint. The few-shot classifier in `src/classifiers.py` calls that endpoint.

**Status: written, never run.** No GPU deployment of FLAN-UL2 was done while building this repo, so every setting below
comes from documentation and source code, not from a working endpoint. `python src/run_pipeline.py` reports the LLM
row as "not run" until you deploy this and set the environment variables below.

## What it costs

A deployed replica bills every hour it is up, whether or not it receives requests. The script sets
`min_replica_count=1`, and no scale-to-zero option was found for this setup.

| Profile | Machine | GPUs | List price per node hour | Per day | Per 30 days |
| :--- | :--- | :--- | ---: | ---: | ---: |
| `l4x2` (default) | `g2-standard-24` | 2 x NVIDIA L4 (48 GB total) | $2.30138 | $55.23 | $1,656.99 |
| `a100-80` | `a2-ultragpu-1g` | 1 x NVIDIA A100 80 GB | $5.7818474 | $138.76 | $4,162.93 |

Prices are Vertex AI online-prediction list prices from <https://cloud.google.com/vertex-ai/pricing> (the page now
redirects to <https://cloud.google.com/products/gemini-enterprise-agent-platform/pricing>), read on 2026-10-23. The
page shows no date, and the region of the figures was not confirmed; the Iowa (us-central1) column was listed first.
The per-day and per-30-day columns are the hourly price multiplied out. Billing runs in 30-second increments. Network
and storage are extra. Run `--undeploy` as soon as an evaluation finishes. A full test split of 645 prompts is a
short job; the 40 GB weight download and load at start-up is the slow part.

## Run it

```bash
pip install google-cloud-aiplatform
python deploy/vertex/deploy_flan_ul2.py                     # dry run (default): prints the config and the cost
python deploy/vertex/deploy_flan_ul2.py --deploy --project YOUR_PROJECT --region us-central1
```

`--deploy` and `--undeploy` do nothing unless you pass them explicitly together with `--project` and `--region`. The
script prints the new model and endpoint ids. Then point the pipeline at the endpoint:

```bash
export VERTEX_PROJECT=YOUR_PROJECT VERTEX_REGION=us-central1 VERTEX_ENDPOINT_ID=<endpoint id>
python src/run_pipeline.py        # now measures the LLM row on the test split; output is not cached
```

Tear down when finished:

```bash
python deploy/vertex/deploy_flan_ul2.py --undeploy --project YOUR_PROJECT --region us-central1 \
  --endpoint-id <endpoint id> --model-id <model id>
```

This calls `endpoint.undeploy_all()` (stops the hourly charge), `endpoint.delete()` and `model.delete()`. The same
with gcloud: `gcloud ai endpoints undeploy-model ENDPOINT_ID --region=REGION --deployed-model-id=DEPLOYED_ID`, then
`gcloud ai endpoints delete ENDPOINT_ID --region=REGION`, then `gcloud ai models delete MODEL_ID --region=REGION`.
Check the Vertex AI console afterwards that no endpoint is left deployed.

## Why these settings

- **Container.** The Google-hosted Hugging Face TGI image
  `us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-generation-inference-cu124.2-4.ubuntu2204.py311`
  (TGI 2.4.0, the newest TGI image in that registry when checked). TGI maps the T5 model type to its sequence-to-sequence
  path: <https://github.com/huggingface/text-generation-inference/blob/v2.4.0/server/text_generation_server/models/__init__.py>.
  Its supported-models list names `google/flan-t5-xxl` and does not name FLAN-UL2:
  <https://github.com/huggingface/text-generation-inference/blob/main/docs/source/supported_models.md>.
- **Model.** `google/flan-ul2` is a `T5ForConditionalGeneration`, Apache-2.0, not gated, stored as 8 PyTorch `.bin`
  shards totalling 39.45 GB in bfloat16: <https://huggingface.co/google/flan-ul2>. The container downloads it from the
  Hub at start-up, so no token is needed.
- **`DTYPE=bfloat16`.** TGI's seq2seq code defaults to float16 on CUDA, and T5-family models overflow in float16
  (`seq2seq_lm.py` in v2.4.0, linked from the file above).
- **`NUM_SHARD=2` on two L4s.** The weights alone are 39.45 GB. Split over two 24 GB cards that is about 19.7 GB each,
  leaving roughly 4 GB per card for activations and the cache. That is my arithmetic, not a measurement. A 2048-token
  prompt may not fit. If it runs out of memory, use the `a100-80` profile (1 x 80 GB, no sharding). A single A100 40 GB
  does not hold the bfloat16 weights.
- **`MAX_INPUT_TOKENS=2048`, `MAX_TOTAL_TOKENS=2064`.** The 2023 work was limited to 2048 input tokens, and the
  prompt builder budgets for that. The extra 16 tokens hold the generated code. An open TGI issue reports T5 warm-up
  failing in v3.1.0 when these limits are set (<https://github.com/huggingface/text-generation-inference/issues/3019>).
  This container is 2.4.0 and may not be affected. If the container fails at warm-up, remove these two variables first.
- **`/generate` as the predict route.** The container's default Vertex handler, as read from the v2.4.0 source
  (<https://github.com/huggingface/text-generation-inference/blob/v2.4.0/router/src/vertex.rs>), forwards only
  `max_new_tokens` and `seed` and hard-codes `do_sample: true`. Sampled output is wrong for classification. With
  `/generate`, `src/llm.py` sends `{"inputs": ..., "parameters": {"do_sample": false, "max_new_tokens": 8}}` through
  `endpoint.raw_predict` and reads `generated_text` from the reply.
- **SDK calls.** `Model.upload`, `Model.deploy`, `Endpoint.raw_predict`, `Endpoint.undeploy_all` and `Endpoint.delete`
  follow the signatures in <https://github.com/googleapis/python-aiplatform/blob/main/google/cloud/aiplatform/models.py>
  and the Hugging Face Vertex notebook
  <https://github.com/huggingface/Google-Cloud-Containers/blob/main/examples/vertex-ai/notebooks/deploy-gemma-on-vertex-ai/vertex-notebook.ipynb>.

## What was not verified

- That FLAN-UL2 loads and answers on this container. Nobody ran it.
- That it fits in memory on 2 x L4 at 2048 input tokens.
- That the `/generate` route and response shape work through Vertex's `raw_predict` on this image.
- Model Garden: FLAN-UL2 is not a curated entry. `gcloud ai model-garden models list --model-filter=ul2
  --can-deploy-hugging-face-models` returned no items (gcloud 587.0.0, 2026-10-23). `gcloud ai model-garden models
  list-deployment-config --model=google/flan-ul2` prints an auto-generated vLLM configuration, but vLLM's supported-model
  list shows no native T5 support, so that configuration is probably unusable and was not tried.
- TGI is in maintenance mode (latest release v3.3.7, 2025-12-19): <https://github.com/huggingface/text-generation-inference>.
- GPU quota in your project and region, and whether the machine types are offered there.
- The price page's region and date, noted above.

If you hit any of these, the fallback in the Hugging Face tooling is the PyTorch inference container
(`huggingface-pytorch-inference-cu121.2-3.transformers.4-48.ubuntu2204.py311`) with `HF_TASK=text2text-generation`.
That route is not wired into the script.
