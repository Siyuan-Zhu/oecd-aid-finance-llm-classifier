"""SageMaker inference handler for FLAN-UL2, bundled into model.tar.gz under code/.

Follows the handler in the public "Deploy FLAN-UL2 20B on Amazon SageMaker" notebook (Philipp Schmid, MIT
licence), the notebook used for the 2023 deployment. Request: {"inputs": "<prompt>", "parameters": {<generate kwargs>}}.
Response: [{"generated_text": "<answer>"}].
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def model_fn(model_dir):
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir, device_map="auto", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer


def predict_fn(data, model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    inputs = data.pop("inputs", data)
    parameters = data.pop("parameters", None)
    input_ids = tokenizer(inputs, return_tensors="pt").input_ids.to(model.device)
    outputs = model.generate(input_ids, **parameters) if parameters is not None else model.generate(input_ids)
    return [{"generated_text": tokenizer.decode(outputs[0], skip_special_tokens=True)}]
