"""The Vertex AI deployment script: config builder, CLI guards, and call order against a stand-in SDK.

Nothing here talks to Google Cloud. The SDK is replaced by a recording stand-in.
"""

import json
import sys
import types

import pytest

from deploy.vertex.deploy_flan_ul2 import build_config, main


def test_default_config_serves_flan_ul2_with_the_tgi_container_on_two_l4s():
    config = build_config("l4x2")
    upload = config["model_upload"]
    assert upload["serving_container_image_uri"].endswith(
        "huggingface-text-generation-inference-cu124.2-4.ubuntu2204.py311"
    )
    env = upload["serving_container_environment_variables"]
    assert env["MODEL_ID"] == "google/flan-ul2"
    assert env["DTYPE"] == "bfloat16"                 # T5 overflows in fp16
    assert env["NUM_SHARD"] == "2"
    assert (env["MAX_INPUT_TOKENS"], env["MAX_TOTAL_TOKENS"]) == ("2048", "2064")
    assert upload["serving_container_predict_route"] == "/generate"   # the default Vertex route forces sampling
    assert config["deploy"]["machine_type"] == "g2-standard-24"
    assert (config["deploy"]["accelerator_type"], config["deploy"]["accelerator_count"]) == ("NVIDIA_L4", 2)


def test_a100_profile_uses_one_gpu_and_no_sharding():
    config = build_config("a100-80")
    assert config["deploy"]["machine_type"] == "a2-ultragpu-1g"
    assert config["deploy"]["accelerator_type"] == "NVIDIA_A100_80GB"
    assert config["model_upload"]["serving_container_environment_variables"]["NUM_SHARD"] == "1"


def test_cost_estimate_follows_from_the_hourly_price():
    cost = build_config("l4x2")["estimated_cost_usd"]
    assert cost["per_hour"] == pytest.approx(2.30138)
    assert cost["per_day"] == pytest.approx(2.30138 * 24)
    assert cost["per_30_days"] == pytest.approx(2.30138 * 24 * 30)


def test_one_replica_only_so_nothing_scales_out_by_surprise():
    deploy = build_config("l4x2")["deploy"]
    assert (deploy["min_replica_count"], deploy["max_replica_count"]) == (1, 1)


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError):
        build_config("tpu-pod")


def test_default_run_is_a_dry_run_that_prints_the_config(capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "google.cloud.aiplatform", None)   # would fail if the SDK were touched
    assert main([]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["model_upload"]["serving_container_environment_variables"]["MODEL_ID"] == "google/flan-ul2"


@pytest.mark.parametrize("argv", [
    ["--deploy"], ["--deploy", "--project", "p"], ["--deploy", "--region", "r"],
    ["--undeploy", "--project", "p", "--region", "r"],            # no endpoint id
    ["--deploy", "--undeploy", "--project", "p", "--region", "r", "--endpoint-id", "1"],
])
def test_deploy_and_undeploy_need_explicit_arguments(argv, monkeypatch):
    monkeypatch.setitem(sys.modules, "google.cloud.aiplatform", None)
    with pytest.raises(SystemExit) as stop:
        main(argv)
    assert stop.value.code == 2


class _Recorder:
    def __init__(self):
        self.calls = []


def _fake_sdk(recorder):
    """A stand-in for google.cloud.aiplatform that records calls."""
    sdk = types.ModuleType("google.cloud.aiplatform")

    class Model:
        def __init__(self, name="model-1"):
            self.name = name

        @classmethod
        def upload(cls, **kwargs):
            recorder.calls.append(("Model.upload", kwargs))
            return cls("model-1")

        def wait(self):
            recorder.calls.append(("model.wait", {}))

        def deploy(self, **kwargs):
            recorder.calls.append(("model.deploy", kwargs))

        def delete(self):
            recorder.calls.append(("model.delete", {"name": self.name}))

    class Endpoint:
        def __init__(self, endpoint_name="endpoint-1"):
            self.name = endpoint_name

        @classmethod
        def create(cls, **kwargs):
            recorder.calls.append(("Endpoint.create", kwargs))
            return cls("endpoint-1")

        def undeploy_all(self):
            recorder.calls.append(("endpoint.undeploy_all", {"name": self.name}))

        def delete(self):
            recorder.calls.append(("endpoint.delete", {"name": self.name}))

    sdk.init = lambda **kwargs: recorder.calls.append(("init", kwargs))
    sdk.Model, sdk.Endpoint = Model, Endpoint
    return sdk


def test_deploy_calls_the_sdk_in_order_with_the_built_config(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setitem(sys.modules, "google.cloud.aiplatform", _fake_sdk(recorder))
    assert main(["--deploy", "--project", "proj", "--region", "us-central1"]) == 0
    names = [name for name, _ in recorder.calls]
    assert names == ["init", "Model.upload", "model.wait", "Endpoint.create", "model.deploy"]
    assert recorder.calls[0][1] == {"project": "proj", "location": "us-central1"}
    assert recorder.calls[-1][1]["machine_type"] == "g2-standard-24"


def test_undeploy_stops_billing_then_deletes_the_endpoint_and_model(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setitem(sys.modules, "google.cloud.aiplatform", _fake_sdk(recorder))
    argv = ["--undeploy", "--project", "proj", "--region", "us-central1", "--endpoint-id", "123", "--model-id", "456"]
    assert main(argv) == 0
    names = [name for name, _ in recorder.calls]
    assert names == ["init", "endpoint.undeploy_all", "endpoint.delete", "model.delete"]
