# %%
import transformers
from training_utils import make_supervised_data_module
from collections import namedtuple
from typing import Dict, List
from peft import AutoPeftModelForCausalLM
import torch
from tqdm import tqdm


def setup_data_args(
    data_path, base_model, data_length=None, val_split=None, subset=None
):
    DataArgs = namedtuple(
        "DataArgs", ["data_path", "data_length", "val_split", "subset", "is_chat"]
    )
    is_chat = "Llama-3" in base_model and "Instruct" in base_model
    return DataArgs(data_path, data_length, val_split, subset, is_chat)


def load_data(
    data_path: str,
    base_model: str,
    data_length: int = None,
    val_split: float = 0.0,
    subset: str = None,
    model_max_length: int = 1024,  # Set to something large enough to handle the data for eval usecase. Doesnt have to match train.
) -> tuple:
    """
    Load and prepare dataset for training and evaluation.

    Args:
        data_path: Path to the dataset ('meta-math/MetaMathQA' or 'qiaojin/PubMedQA')
        base_model: Name of the base model to use
        data_length: Number of samples to use from dataset
        val_split: Fraction of data to use for validation
        model_max_length: Maximum sequence length for tokenizer. If None, uses defaults (512 for meta-math, 768 for PubMedQA)

    Returns:
        tuple: (train_dataset, eval_dataset, tokenizer)
    """
    # Setup data arguments based on dataset
    if "meta-math" in data_path or data_path in ("openai/gsm8k", "qiaojin/PubMedQA"):
        data_args = setup_data_args(
            data_path, base_model, data_length=data_length, val_split=val_split
        )
    else:
        raise ValueError(f"Unsupported dataset: {data_path}")

    # Initialize tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token_id = 2  # unk token, different from eos token

    # Create data module and return datasets
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    return data_module, tokenizer


def load_model(ckpt_path: str) -> AutoPeftModelForCausalLM:
    """
    Load a PEFT model from a checkpoint path.

    Args:
        ckpt_path: Path to the model checkpoint

    Returns:
        AutoPeftModelForCausalLM: The loaded model
    """
    model = AutoPeftModelForCausalLM.from_pretrained(
        ckpt_path,
        # device_map="auto",
    )
    return model


# Load the model
# ckpt_path = '/root/MoRA/pub-med-qa/save_test_lora_rank128_lr1e-4/checkpoint-1200'
# model = load_model(ckpt_path)


# %%
def get_model_generations(
    model, prompts: List[str], max_new_tokens: int = 512
) -> List[torch.Tensor]:
    """
    Get predictions from the model for a list of prompts.

    Args:
        model: The model to get predictions from.
        prompts: List of prompt strings to generate from.

    Returns:
        List[torch.Tensor]: The generated outputs from the model.
    """
    predictions = []
    device = model.device

    for prompt in tqdm(prompts, desc="Generating.."):
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            prediction = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=2,
            )

        predictions.append(prediction.cpu().squeeze(0))

    return predictions


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


def extract_decision(label: str) -> str:
    decision = label.split("Final Decision: ")[1].split("<|eot_id|>")[0].strip()
    if decision not in ["yes", "no", "maybe"]:
        print(f"Invalid decision: {decision}")
        return None
    return decision



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

args = parser.parse_args()
print(f"Args:\n{args}")

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_path = "/root/MoRA/pub-med-qa/lora_rank128_lr1e-4_witheval/checkpoint-400"
print("Loading model...")
model = load_model(ckpt_path)
model.to(device)

data_path = "qiaojin/PubMedQA"
subset = args.subset
ckpt_path = args.ckpt_path
save_path = f"pub-med-eval/{args.run_name}.json"
print(f"Results will be saved to {save_path}")

# Load model
print(f"Loading model from {ckpt_path}...")
model = load_model(ckpt_path)
model.to(device)

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

# One of ['yes', 'no', 'maybe']
decisions_y = [extract_decision(d["labels"]) for d in data]
decisions_y

# Setup input prompts
prompts = [data[i]["input_ids"] for i in range(len(data))]
# print(prompts[0])

# Get model generations
generations = get_model_generations(model, prompts)
generations_str = tokenizer.batch_decode(generations, skip_special_tokens=True)


decisions_yhat = []
for g in generations_str:
    try:
        decision = extract_decision(g)
        decisions_yhat.append(decision)
    except Exception as e:
        print(f"Error extracting decision from {g}")
        decisions_yhat.append("EXTRACT_FAILURE")
eval_accuracy = get_eval_accuracy(decisions_yhat, decisions_y)

# Save all results
results = {
    "decisions_yhat": decisions_yhat,
    "decisions_y": decisions_y,
    "generations_str": generations_str,
    "eval_accuracy": eval_accuracy,
    "data_path": data_path,
    "subset": subset,
    "ckpt_path": ckpt_path,
}

# Save results to json
with open(save_path, "w") as f:
    json.dump(results, f)
