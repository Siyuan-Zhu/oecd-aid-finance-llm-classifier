# FLAN-UL2 on SageMaker

Deploys the model the classifier calls: Google's FLAN-UL2 (20 billion parameters) behind a SageMaker real-time
endpoint, as in the 2023 work.

```bash
pip install -e ".[deploy]"
python deploy/sagemaker/deploy_flan_ul2.py            # dry run: prints the config and the hourly price
python deploy/sagemaker/deploy_flan_ul2.py --deploy   # downloads about 40 GB, uploads it, creates the endpoint
python deploy/sagemaker/deploy_flan_ul2.py --delete   # tears the endpoint, its config and the model down
```

- **Instance:** one `ml.g5.12xlarge` (4 x NVIDIA A10G). The 2023 write-up gives about 42 GB of GPU memory for the model.
- **Container:** the Hugging Face inference container (transformers 4.26, PyTorch 1.13, Python 3.9) running `code/inference.py`, which loads the model in bfloat16 across the GPUs.
- **Request:** `{"inputs": "<prompt>", "parameters": {"max_new_tokens": 16, "do_sample": false}}`. **Response:** `[{"generated_text": "<answer>"}]`.
- **Price:** the dry run prints $5.67 per hour and cites its source, the 2023 write-up. That is the 2023 figure; check the current SageMaker pricing before you deploy.
- **Billing:** the endpoint bills every hour it exists, whether or not it gets requests. Run `--delete` when you finish. It does not remove the `model.tar.gz` in S3.
- **Credentials and role:** the script reads your normal AWS configuration and needs a SageMaker execution role. It stores nothing.

The deployment follows the "Deploy FLAN-UL2 20B on Amazon SageMaker" notebook used for the 2023 work. Only the dry run, the config and the delete path are covered by tests; `--deploy` has not been run from this repository.
