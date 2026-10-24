"""Text generation backends for the few-shot classifier.

`TextGenerator` is the seam. The only adapter here is `VertexEndpointGenerator`, which calls a FLAN-UL2 endpoint
deployed with deploy/vertex/deploy_flan_ul2.py. There is no fake generator in this package: without a configured
endpoint the pipeline reports the LLM row as not run. Test doubles live in tests/fakes.py.

The request shape follows Hugging Face text-generation-inference's native /generate route (greedy decoding). It has
not been run against a live endpoint; see deploy/vertex/README.md.
"""

from __future__ import annotations

import importlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from src.settings import MODEL_ID

ENV_VARS = ("VERTEX_PROJECT", "VERTEX_REGION", "VERTEX_ENDPOINT_ID")
MAX_NEW_TOKENS = 8   # a 5-digit code is a handful of SentencePiece tokens


class TextGenerator(Protocol):
    model_id: str   # which model answers; written into results/metrics.json

    def generate(self, prompts: list[str]) -> list[str]: ...


class LLMNotConfigured(RuntimeError):
    """No Vertex AI endpoint is configured in the environment."""


class VertexEndpointGenerator:
    def __init__(
        self,
        project: str,
        region: str,
        endpoint_id: str,
        endpoint=None,
        max_workers: int = 4,
        model_id: str = MODEL_ID,
    ):
        # The endpoint does not report what it serves. model_id is what deploy_flan_ul2.py deploys; pass another
        # value if the endpoint was set up differently.
        self.model_id = model_id
        self.project = project
        self.region = region
        self.endpoint_id = endpoint_id
        self._endpoint = endpoint
        self._max_workers = max_workers

    @classmethod
    def from_env(cls) -> VertexEndpointGenerator:
        missing = [name for name in ENV_VARS if not os.environ.get(name)]
        if missing:
            raise LLMNotConfigured("no endpoint configured: set " + ", ".join(missing))
        return cls(os.environ["VERTEX_PROJECT"], os.environ["VERTEX_REGION"], os.environ["VERTEX_ENDPOINT_ID"])

    def _get_endpoint(self):
        if self._endpoint is None:
            try:
                aiplatform = importlib.import_module("google.cloud.aiplatform")
            except ImportError as error:
                raise ImportError(
                    "VertexEndpointGenerator needs google-cloud-aiplatform: pip install google-cloud-aiplatform"
                ) from error
            aiplatform.init(project=self.project, location=self.region)
            self._endpoint = aiplatform.Endpoint(self.endpoint_id)
        return self._endpoint

    def _one(self, prompt: str) -> str:
        body = json.dumps({"inputs": prompt, "parameters": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS}})
        response = self._get_endpoint().raw_predict(
            body=body.encode("utf-8"), headers={"Content-Type": "application/json"}
        )
        return str(response.json()["generated_text"])

    def generate(self, prompts: list[str]) -> list[str]:
        self._get_endpoint()   # fail fast, once, before any worker thread starts
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            return list(pool.map(self._one, prompts))
