import pathlib
import os
import transformers
from training_utils import make_supervised_data_module
from collections import namedtuple
from typing import Dict, List
import torch
from tqdm import tqdm
from eval_utils import load_data, extract_decision, load_model, get_model_generations




def get_eval_accuracy(decisions_yhat: List[str], decisions_y: List[str]) -> float:
    """
    Get the evaluation accuracy for a given model and tokenized data.
    """
    assert len(decisions_yhat) == len(
        decisions_y
    ), f"Lengths are not equal: {len(decisions_yhat)} != {len(decisions_y)}"
    return sum(1 for yhat, y in zip(decisions_yhat, decisions_y) if yhat == y) / len(
        decisions_yhat
    )


def get_eval_loss(
    model, tokenized_data: Dict[str, torch.Tensor], batch_size: int = 8
) -> float:
    """
    Get the evaluation loss for a given model and tokenized data.

    Args:
        model: The model to get the loss from.
        tokenized_data: The tokenized data to get the loss from.

    Returns:
        float: The average loss.
        List[float]: The loss for each batch.
    """
    loss_by_batch = []
    device = model.device
    num_samples = len(tokenized_data["input_ids"])

    for i in tqdm(range(0, num_samples, batch_size), desc="Calculating loss.."):
        batch = {k: v[i : i + batch_size].to(device) for k, v in tokenized_data.items()}

        with torch.no_grad():
            outputs = model(**batch)
            loss_by_batch.append(outputs.loss.mean().item())

    return sum(loss_by_batch) / len(loss_by_batch), loss_by_batch

# Load pubmed test set
import json
import uuid
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Evaluate model on PubMedQA dataset")
parser.add_argument(
    "--subset",
    type=str,
    default="pqa_labeled",
    choices=["pqa_labeled", "pqa_artificial"],
    help="Subset of PubMedQA to use",
)
parser.add_argument(
    "--ckpt_path", type=str, required=True, help="Path to model checkpoint"
)
parser.add_argument(
    "--run_name",
    type=str,
    required=True,
    help="Run name, under pub-med-eval/ e.g pqa_labeled_results will be saved under pub-med-eval/pqa_labeled_results.json",
)
parser.add_argument(
    "--data_length",
    type=int,
    default=1000,
    help="Number of samples to use from dataset",
)
parser.add_argument(
    "--eval_base_model",
    type=bool,
    default=False,
    help="Whether to evaluate the base model only",
)

args = parser.parse_args()
print(f"Args:\n{args}")

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model from {args.ckpt_path}...")
model = load_model(args.ckpt_path)
model.to(device)

data_path = "qiaojin/PubMedQA"
subset = args.subset
ckpt_path = args.ckpt_path
save_path = pathlib.Path(f"pub-med-eval-v2/{args.run_name}.json")
print(f"Results will be saved to {save_path}")
if not save_path.exists():
    save_path.parent.mkdir(parents=True, exist_ok=True)

# Load model
print(f"Loading model from {ckpt_path}...")
model = load_model(ckpt_path)
model.to(device)
if args.eval_base_model:
    print("Disabling adapters...")
    model.disable_adapters()

# Load the data using the new function
data_module, tokenizer = load_data(
    data_path=data_path,
    base_model="meta-llama/Llama-3.2-1B-Instruct",
    subset=subset,
    data_length=args.data_length,
)

# List[Dict[str, str]]
# Array with num_samples elements. Each element has two fields, input_ids and labels.
data = data_module["train_dataset"]
print(f"Loaded {len(data)} samples")

prompts = [data[i]["input_ids"] for i in range(len(data))]

# Get model generations
generations = get_model_generations(model, tokenizer, prompts)
generations_str = tokenizer.batch_decode(generations, skip_special_tokens=True)

# Save all results
results = {
    "generations_str": generations_str,
    "data_path": data_path,
    "subset": subset,
    "ckpt_path": ckpt_path,
}

# Save results to json
with open(save_path, "w") as f:
    json.dump(results, f)
