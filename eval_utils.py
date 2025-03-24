from typing import Dict, List
from training_utils import make_supervised_data_module
from collections import namedtuple
import transformers
import torch
from tqdm import tqdm
from peft import AutoPeftModelForCausalLM

def setup_data_args(
    data_path, is_chat, data_length=None, val_split=None, subset=None
):
    DataArgs = namedtuple(
        "DataArgs", ["data_path", "data_length", "val_split", "subset", "is_chat"]
    )
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
    data_args = setup_data_args(
        data_path,
        base_model,
        data_length=data_length,
        val_split=val_split,
        subset=subset,
    )

    # Initialize tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token_id = 2  # unk token, different from eos token
    
    data_args.is_chat = "Llama-3" in base_model and "Instruct" in base_model
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    return data_module, tokenizer

def extract_decision(label: str) -> str:
    try:
        decision = label.split('Final Decision: ')[1].split('<|eot_id|>')[0].strip()
        if decision not in ['yes', 'no', 'maybe']:
            print(f"Invalid decision: {decision}")
            return None
        return decision
    except:
        print(f"Error extracting decision from label: {label}")
        return None
    
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
        device_map="auto",
    )
    return model


# Load the model
# ckpt_path = '/root/MoRA/pub-med-qa/save_test_lora_rank128_lr1e-4/checkpoint-1200'
# model = load_model(ckpt_path)

def get_model_generations(
    model, tokenizer, prompts: List[str] | List[torch.Tensor], max_new_tokens: int = 512
) -> List[torch.Tensor]:
    """
    Get predictions from the model for a list of prompts one by one.

    Args:
        model: The model to get predictions from.
        prompts: List of prompt strings to generate from or list of input_ids tensors
    Returns:
        List[torch.Tensor]: The generated outputs from the model.
    """
    predictions = []
    device = model.device

    for prompt in tqdm(prompts, desc="Generating.."):
        if isinstance(prompt, str):
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
        elif isinstance(prompt, torch.Tensor):
            input_ids = prompt.to(device)
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        with torch.no_grad():
            prediction = model.generate(
                input_ids=input_ids,
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
