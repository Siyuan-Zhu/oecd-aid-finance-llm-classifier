"""The FLAN-UL2 adapter sends the 2023 request format to a SageMaker endpoint and reads the generated text."""

import io
import json
import sys

import pytest

from purpose_coder.backend import DEFAULT_PARAMETERS, FlanUL2SageMaker


class FakeRuntime:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def invoke_endpoint(self, **kwargs):
        self.calls.append(kwargs)
        return {"Body": io.BytesIO(json.dumps(self.reply).encode())}


def test_request_is_inputs_plus_parameters_as_json():
    runtime = FakeRuntime([{"generated_text": " 12250\n"}])
    generator = FlanUL2SageMaker("my-endpoint", client=runtime)
    answer = generator.generate("the prompt")
    (call,) = runtime.calls
    assert call["EndpointName"] == "my-endpoint"
    assert call["ContentType"] == "application/json"
    assert json.loads(call["Body"]) == {"inputs": "the prompt", "parameters": DEFAULT_PARAMETERS}
    assert answer == "12250"


def test_generation_parameters_can_be_replaced():
    runtime = FakeRuntime([{"generated_text": "x"}])
    FlanUL2SageMaker("e", parameters={"max_new_tokens": 5}, client=runtime).generate("p")
    assert json.loads(runtime.calls[0]["Body"])["parameters"] == {"max_new_tokens": 5}


def test_boto3_is_only_needed_when_the_first_request_is_made(monkeypatch):
    monkeypatch.setitem(sys.modules, "boto3", None)  # makes `import boto3` raise ImportError
    generator = FlanUL2SageMaker("e")  # building the adapter is fine
    with pytest.raises(ImportError, match="boto3"):
        generator.generate("p")
