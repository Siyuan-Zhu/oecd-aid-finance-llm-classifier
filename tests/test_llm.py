"""The Vertex AI text generator: configuration from the environment, and the request it sends."""

import json
import sys

import pytest

from src.llm import LLMNotConfigured, VertexEndpointGenerator
from src.settings import MODEL_ID
from tests.fakes import FakeEndpoint

ENV = {"VERTEX_PROJECT": "my-project", "VERTEX_REGION": "us-central1", "VERTEX_ENDPOINT_ID": "1234567890"}


@pytest.fixture
def clean_env(monkeypatch):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_from_env_raises_when_nothing_is_configured(clean_env):
    with pytest.raises(LLMNotConfigured) as error:
        VertexEndpointGenerator.from_env()
    for name in ENV:
        assert name in str(error.value)


def test_from_env_names_only_the_missing_variables(clean_env):
    clean_env.setenv("VERTEX_PROJECT", "my-project")
    with pytest.raises(LLMNotConfigured) as error:
        VertexEndpointGenerator.from_env()
    assert "VERTEX_PROJECT" not in str(error.value)
    assert "VERTEX_REGION" in str(error.value)


def test_from_env_builds_a_generator_without_touching_gcp(clean_env):
    for name, value in ENV.items():
        clean_env.setenv(name, value)
    generator = VertexEndpointGenerator.from_env()
    assert (generator.project, generator.region, generator.endpoint_id) == ("my-project", "us-central1", "1234567890")


def test_generate_sends_greedy_decoding_requests_and_returns_answers_in_order():
    endpoint = FakeEndpoint(reply="12263")
    generator = VertexEndpointGenerator("p", "r", "e", endpoint=endpoint)
    answers = generator.generate(["first prompt", "second prompt"])
    assert answers == ["12263", "12263"]
    sent = [json.loads(body) for body, _ in endpoint.calls]
    assert [s["inputs"] for s in sent] == ["first prompt", "second prompt"]
    assert all(s["parameters"] == {"do_sample": False, "max_new_tokens": 8} for s in sent)
    assert all(headers == {"Content-Type": "application/json"} for _, headers in endpoint.calls)


def test_missing_vertex_sdk_is_reported_clearly(monkeypatch):
    monkeypatch.setitem(sys.modules, "google.cloud.aiplatform", None)
    generator = VertexEndpointGenerator("p", "r", "e")
    with pytest.raises(ImportError, match="google-cloud-aiplatform"):
        generator.generate(["prompt"])


def test_the_generator_names_the_model_it_expects_the_endpoint_to_serve(clean_env):
    for name, value in ENV.items():
        clean_env.setenv(name, value)
    assert VertexEndpointGenerator.from_env().model_id == MODEL_ID
    assert VertexEndpointGenerator("p", "r", "e", model_id="other/model").model_id == "other/model"
