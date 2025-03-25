import pathlib
import os
import transformers
from training_utils import make_supervised_data_module
from collections import namedtuple
from typing import Dict, List
import torch
from tqdm import tqdm
from eval_utils import load_data, extract_decision, load_model, get_model_generations, get_eval_accuracy, get_eval_loss

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

data_path = "qiaojin/PubMedQA"
subset = args.subset
ckpt_path = args.ckpt_path
save_path = pathlib.Path(f"pub-med-eval/{args.run_name}.json")
print(f"Results will be saved to {save_path}")
if not save_path.exists():
    save_path.parent.mkdir(parents=True, exist_ok=True)

# Load model
print(f"Loading model from {ckpt_path}...")
model = load_model(ckpt_path)
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
