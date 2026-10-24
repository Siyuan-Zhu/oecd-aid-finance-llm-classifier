"""The model side: a TextGenerator interface and the FLAN-UL2 SageMaker adapter.

FLAN-UL2 is served as in David's SageMaker notebook (deploy/sagemaker/): the Hugging Face inference container
runs code/inference.py, which takes {"inputs": <prompt>, "parameters": {...generate kwargs}} and returns
[{"generated_text": <answer>}]. FLAN-UL2's context window is 2,048 tokens, which caps the prompt (and so k).
"""

from __future__ import annotations

import json
from typing import Any, Protocol


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str:
        """The model's answer to `prompt`."""
        ...


# Decoding settings are not specified in the 2023 drafts. Greedy decoding and a short limit fit an answer
# that is a single 5-digit code.
DEFAULT_PARAMETERS: dict[str, Any] = {"do_sample": False, "max_new_tokens": 16}


class FlanUL2SageMaker:
    """Calls a SageMaker real-time endpoint that serves FLAN-UL2. boto3 is imported on first use."""

    def __init__(
        self,
        endpoint_name: str,
        region_name: str | None = None,
        parameters: dict[str, Any] | None = None,
        client: Any = None,
    ):
        self.endpoint_name = endpoint_name
        self.region_name = region_name
        self.parameters = dict(DEFAULT_PARAMETERS if parameters is None else parameters)
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import boto3
            except ImportError as error:
                raise ImportError(
                    "The SageMaker backend needs boto3: pip install 'oecd-aid-finance-llm-classifier[aws]'"
                ) from error
            self._client = boto3.client("sagemaker-runtime", region_name=self.region_name)
        return self._client

    def request_body(self, prompt: str) -> str:
        return json.dumps({"inputs": prompt, "parameters": self.parameters})

    def generate(self, prompt: str) -> str:
        response = self.client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Body=self.request_body(prompt),
        )
        payload = json.loads(response["Body"].read())
        return str(payload[0]["generated_text"]).strip()
