"""The SageMaker deploy script's dry run, config and teardown, with no AWS calls."""

import sys

import pytest

from deploy.sagemaker import deploy_flan_ul2 as deploy


def test_config_matches_the_2023_setup():
    config = deploy.DeployConfig()
    assert config.hf_model_id == "google/flan-ul2"
    assert config.instance_type == "ml.g5.12xlarge"
    assert config.instance_count == 1
    assert (config.transformers_version, config.pytorch_version, config.py_version) == ("4.26", "1.13", "py39")


def test_dry_run_is_the_default_and_prints_config_price_and_source(capsys):
    assert deploy.main([]) == 0
    out = capsys.readouterr().out
    assert "Dry run" in out
    assert '"instance_type": "ml.g5.12xlarge"' in out
    assert "$5.67 per hour" in out
    assert "Source: 2023" in out


def test_dry_run_does_not_import_aws_libraries(capsys):
    deploy.main([])
    assert "sagemaker" not in sys.modules


def test_price_scales_with_instance_count():
    lines = deploy.price_summary(deploy.DeployConfig(instance_count=2))
    assert "$5.67 per hour for ml.g5.12xlarge x 2" in lines[0]


def test_deploy_and_delete_cannot_be_combined():
    with pytest.raises(SystemExit):
        deploy.main(["--deploy", "--delete"])


def test_options_flow_into_the_config():
    _, config = deploy.parse_args(["--endpoint-name", "mine", "--region", "us-west-2"])
    assert (config.endpoint_name, config.region) == ("mine", "us-west-2")


class FakeSageMaker:
    class exceptions:
        class ClientError(Exception):
            pass

    def __init__(self, missing=()):
        self.missing = set(missing)
        self.deleted = []

    def _delete(self, kind, name):
        if kind in self.missing:
            raise self.exceptions.ClientError("not found")
        self.deleted.append((kind, name))

    def delete_endpoint(self, EndpointName):
        self._delete("endpoint", EndpointName)

    def delete_endpoint_config(self, EndpointConfigName):
        self._delete("endpoint config", EndpointConfigName)

    def delete_model(self, ModelName):
        self._delete("model", ModelName)


def test_delete_removes_endpoint_config_and_model_by_the_endpoint_name():
    client = FakeSageMaker()
    removed = deploy.delete(deploy.DeployConfig(endpoint_name="mine"), client=client)
    assert removed == ["endpoint", "endpoint config", "model"]
    assert client.deleted == [("endpoint", "mine"), ("endpoint config", "mine"), ("model", "mine")]


def test_delete_carries_on_when_one_piece_is_already_gone():
    client = FakeSageMaker(missing={"endpoint config"})
    removed = deploy.delete(deploy.DeployConfig(), client=client)
    assert removed == ["endpoint", "model"]
