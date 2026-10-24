"""Deploy FLAN-UL2 to a SageMaker real-time endpoint, or delete it again.

    python deploy/sagemaker/deploy_flan_ul2.py                 # dry run (default): print config and hourly price
    python deploy/sagemaker/deploy_flan_ul2.py --deploy        # build model.tar.gz, upload it, create the endpoint
    python deploy/sagemaker/deploy_flan_ul2.py --delete        # delete the endpoint, its config and the model

Follows the "Deploy FLAN-UL2 20B on Amazon SageMaker" notebook used for the 2023 deployment: download
google/flan-ul2 from the Hugging Face Hub, add code/inference.py, pack model.tar.gz, upload it to S3 and deploy it
with the Hugging Face inference container on an ml.g5.12xlarge instance.

An endpoint bills every hour it exists. Run --delete when you are done. --delete leaves the model archive in S3.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

# Hourly price of the instance. Source: the 2023 write-up of the FLAN-UL2 work (Model Deployment Details and
# Cost Analysis), which gives $5.67 per hour for g5.12xlarge. That is the on-demand rate quoted in 2023;
# SageMaker's own ml.g5.12xlarge rate can differ, so check the AWS SageMaker pricing page before you deploy.
HOURLY_PRICE_USD = 5.67
PRICE_SOURCE = "2023 AidData FLAN-UL2 write-up, 'Model Deployment Details and Cost Analysis' (g5.12xlarge, hourly rate)"
# The same write-up: experiments, including trial and error, took about 12 to 15 hours of instance runtime.
EXAMPLE_HOURS = (12, 15)


@dataclass(frozen=True)
class DeployConfig:
    endpoint_name: str = "flan-ul2-purpose-coder"
    hf_model_id: str = "google/flan-ul2"              # notebook and write-up
    instance_type: str = "ml.g5.12xlarge"             # notebook, write-up and progress report
    instance_count: int = 1                           # notebook
    transformers_version: str = "4.26"                # notebook
    pytorch_version: str = "1.13"                     # notebook
    py_version: str = "py39"                          # notebook
    model_server_workers: int = 1                     # notebook
    s3_prefix: str = "flan-ul2"                       # notebook
    # Not specified in the 2023 drafts: the notebook left these two timeouts commented out as optional. They
    # are enabled here because a 20B model is slow to download and load.
    container_startup_health_check_timeout: int = 600
    model_data_download_timeout: int = 600
    region: str | None = None

    def describe(self) -> dict[str, Any]:
        return asdict(self)


def price_summary(config: DeployConfig) -> list[str]:
    low, high = (HOURLY_PRICE_USD * h * config.instance_count for h in EXAMPLE_HOURS)
    return [
        f"Hourly instance price: ${HOURLY_PRICE_USD:.2f} per hour for {config.instance_type} x {config.instance_count}",
        f"Source: {PRICE_SOURCE}",
        f"For scale, the 2023 experiments ran {EXAMPLE_HOURS[0]} to {EXAMPLE_HOURS[1]} hours: ${low:.0f} to ${high:.0f}",
        "Check the current AWS SageMaker pricing page; this figure is the 2023 one.",
    ]


def dry_run(config: DeployConfig) -> None:
    print("Dry run. Nothing is created. Pass --deploy to create the endpoint.\n")
    print(json.dumps(config.describe(), indent=2))
    print()
    print("\n".join(price_summary(config)))


def build_model_archive(config: DeployConfig, workdir: Path) -> Path:
    """Download the model, add code/, and pack model.tar.gz with the files at the archive root."""
    from huggingface_hub import snapshot_download

    model_dir = workdir / config.hf_model_id.split("/")[-1]
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(config.hf_model_id, local_dir=str(model_dir), local_dir_use_symlinks=False)
    shutil.copytree(HERE / "code", model_dir / "code", dirs_exist_ok=True)
    archive = model_dir / "model.tar.gz"
    compressor = "pigz" if shutil.which("pigz") else "gzip"
    subprocess.run(
        ["tar", "-cf", archive.name, f"--use-compress-program={compressor}"]
        + sorted(p.name for p in model_dir.iterdir() if p.name != archive.name),
        cwd=model_dir,
        check=True,
    )
    return archive


def deploy(config: DeployConfig, workdir: Path) -> None:
    import sagemaker
    from sagemaker.huggingface.model import HuggingFaceModel
    from sagemaker.s3 import S3Uploader

    import boto3

    session = sagemaker.Session()
    try:
        role = sagemaker.get_execution_role()
    except ValueError:  # outside SageMaker Studio: fall back to a role named as in the notebook
        role = boto3.client("iam").get_role(RoleName="sagemaker_execution_role")["Role"]["Arn"]
    archive = build_model_archive(config, workdir)
    s3_uri = S3Uploader.upload(
        local_path=str(archive), desired_s3_uri=f"s3://{session.default_bucket()}/{config.s3_prefix}"
    )
    print(f"model uploaded to {s3_uri}")
    model = HuggingFaceModel(
        name=config.endpoint_name,
        model_data=s3_uri,
        role=role,
        transformers_version=config.transformers_version,
        pytorch_version=config.pytorch_version,
        py_version=config.py_version,
        model_server_workers=config.model_server_workers,
    )
    model.deploy(
        initial_instance_count=config.instance_count,
        instance_type=config.instance_type,
        endpoint_name=config.endpoint_name,
        container_startup_health_check_timeout=config.container_startup_health_check_timeout,
        model_data_download_timeout=config.model_data_download_timeout,
    )
    print(f"endpoint {config.endpoint_name} is in service")
    print("\n".join(price_summary(config)))
    print("It bills until you run this script with --delete.")


def delete(config: DeployConfig, client: Any = None) -> list[str]:
    """Delete the endpoint, its endpoint config and the model, all named after the endpoint. Returns what was removed."""
    if client is None:
        import boto3

        client = boto3.client("sagemaker", region_name=config.region)
    name = config.endpoint_name
    removed = []
    for label, call in (
        ("endpoint", lambda: client.delete_endpoint(EndpointName=name)),
        ("endpoint config", lambda: client.delete_endpoint_config(EndpointConfigName=name)),
        ("model", lambda: client.delete_model(ModelName=name)),
    ):
        try:
            call()
            removed.append(label)
        except client.exceptions.ClientError as error:
            print(f"could not delete {label} {name}: {error}", file=sys.stderr)
    return removed


def parse_args(argv: list[str] | None) -> tuple[argparse.Namespace, DeployConfig]:
    defaults = DeployConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--deploy", action="store_true", help="really create the endpoint (default is a dry run)")
    mode.add_argument("--delete", action="store_true", help="delete the endpoint, endpoint config and model")
    parser.add_argument("--endpoint-name", default=defaults.endpoint_name)
    parser.add_argument("--instance-type", default=defaults.instance_type)
    parser.add_argument("--region", default=defaults.region)
    parser.add_argument("--workdir", type=Path, default=Path("build"), help="scratch space for the ~40 GB download")
    args = parser.parse_args(argv)
    config = DeployConfig(endpoint_name=args.endpoint_name, instance_type=args.instance_type, region=args.region)
    return args, config


def main(argv: list[str] | None = None) -> int:
    args, config = parse_args(argv)
    if args.deploy:
        deploy(config, args.workdir)
    elif args.delete:
        removed = delete(config)
        print(f"deleted: {', '.join(removed) or 'nothing'}")
    else:
        dry_run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
