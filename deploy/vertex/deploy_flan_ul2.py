"""Deploy google/flan-ul2 to a Vertex AI endpoint, or tear it down. Written from documentation, never run.

    python deploy/vertex/deploy_flan_ul2.py                        # dry run: print the config and cost, call nothing
    python deploy/vertex/deploy_flan_ul2.py --deploy   --project P --region us-central1
    python deploy/vertex/deploy_flan_ul2.py --undeploy --project P --region us-central1 --endpoint-id E [--model-id M]

--deploy starts billing as soon as the node is up and keeps billing until --undeploy. Read deploy/vertex/README.md
first: it lists the cost, what was and was not verified, and the manual teardown commands.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.settings import MAX_INPUT_TOKENS, MAX_TOTAL_TOKENS, MODEL_ID  # noqa: E402

TGI_IMAGE = (
    "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/"
    "huggingface-text-generation-inference-cu124.2-4.ubuntu2204.py311"
)

# Vertex AI online-prediction list prices per node hour, read on 2026-10-23 from
# https://cloud.google.com/vertex-ai/pricing. The page shows no date, and the region of the figures was not confirmed.
PROFILES = {
    "l4x2": {
        "machine_type": "g2-standard-24",
        "accelerator_type": "NVIDIA_L4",
        "accelerator_count": 2,
        "num_shard": 2,
        "usd_per_hour": 2.30138,
    },
    "a100-80": {
        "machine_type": "a2-ultragpu-1g",
        "accelerator_type": "NVIDIA_A100_80GB",
        "accelerator_count": 1,
        "num_shard": 1,
        "usd_per_hour": 5.7818474,
    },
}


def build_config(profile: str = "l4x2") -> dict:
    """Everything --deploy would send, as plain data. Pure: no SDK, no network."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    p = PROFILES[profile]
    return {
        "profile": profile,
        "untested": "Written from documentation. No GPU deployment of FLAN-UL2 was run.",
        "model_upload": {
            "display_name": "flan-ul2-tgi",
            "serving_container_image_uri": TGI_IMAGE,
            "serving_container_ports": [8080],
            # TGI's native route. The default Vertex route (/vertex) sets do_sample=true, which is wrong for
            # classification; /generate takes {"inputs": ..., "parameters": {"do_sample": false}}.
            "serving_container_predict_route": "/generate",
            "serving_container_health_route": "/health",
            "serving_container_environment_variables": {
                "MODEL_ID": MODEL_ID,
                "NUM_SHARD": str(p["num_shard"]),
                "DTYPE": "bfloat16",   # TGI defaults seq2seq models to float16, and T5 overflows in float16
                "MAX_INPUT_TOKENS": str(MAX_INPUT_TOKENS),
                "MAX_TOTAL_TOKENS": str(MAX_TOTAL_TOKENS),
            },
        },
        "endpoint": {"display_name": "flan-ul2-endpoint"},
        "deploy": {
            "machine_type": p["machine_type"],
            "accelerator_type": p["accelerator_type"],
            "accelerator_count": p["accelerator_count"],
            "min_replica_count": 1,
            "max_replica_count": 1,
            "deploy_request_timeout": 3600,   # about 40 GB of weights to download and load
        },
        "estimated_cost_usd": {
            "per_hour": p["usd_per_hour"],
            "per_day": p["usd_per_hour"] * 24,
            "per_30_days": p["usd_per_hour"] * 24 * 30,
            "note": "List price while a replica is deployed, idle or not. Excludes network and storage.",
        },
    }


def _sdk(project: str, region: str):
    try:
        aiplatform = importlib.import_module("google.cloud.aiplatform")
    except ImportError as error:
        raise SystemExit(
            "google-cloud-aiplatform is needed for --deploy/--undeploy: pip install google-cloud-aiplatform"
        ) from error
    aiplatform.init(project=project, location=region)
    return aiplatform


def deploy(config: dict, project: str, region: str) -> dict:
    aiplatform = _sdk(project, region)
    model = aiplatform.Model.upload(**config["model_upload"])
    model.wait()
    endpoint = aiplatform.Endpoint.create(**config["endpoint"])
    model.deploy(endpoint=endpoint, **config["deploy"])
    return {"model": model.name, "endpoint": endpoint.name}


def undeploy(project: str, region: str, endpoint_id: str, model_id: str | None) -> None:
    aiplatform = _sdk(project, region)
    endpoint = aiplatform.Endpoint(endpoint_id)
    endpoint.undeploy_all()     # stops node-hour billing
    endpoint.delete()
    if model_id:
        aiplatform.Model(model_id).delete()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="l4x2")
    parser.add_argument("--dry-run", action="store_true", help="print the config and call nothing (the default)")
    parser.add_argument(
        "--deploy", action="store_true", help="create the model and endpoint and deploy (starts billing)"
    )
    parser.add_argument("--undeploy", action="store_true", help="undeploy and delete the endpoint (and model if given)")
    parser.add_argument("--project", help="Google Cloud project id")
    parser.add_argument("--region", help="for example us-central1")
    parser.add_argument("--endpoint-id", help="endpoint to tear down (--undeploy)")
    parser.add_argument("--model-id", help="model to delete (--undeploy, optional)")
    args = parser.parse_args(argv)

    if sum([args.deploy, args.undeploy, args.dry_run]) > 1:
        parser.error("--dry-run, --deploy and --undeploy cannot be combined")
    if (args.deploy or args.undeploy) and not (args.project and args.region):
        parser.error("--deploy and --undeploy need --project and --region")
    if args.undeploy and not args.endpoint_id:
        parser.error("--undeploy needs --endpoint-id")

    config = build_config(args.profile)
    if args.deploy:
        machine = config["deploy"]["machine_type"]
        print(f"deploying {MODEL_ID} on {machine}; billing starts when the node is up", file=sys.stderr)
        print(json.dumps(deploy(config, args.project, args.region), indent=2))
    elif args.undeploy:
        undeploy(args.project, args.region, args.endpoint_id, args.model_id)
        print("undeployed and deleted", file=sys.stderr)
    else:
        print(json.dumps(config, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
