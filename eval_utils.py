from typing import List
from training_utils import make_supervised_data_module
from collections import namedtuple
import transformers
import torch
from tqdm import tqdm
from peft import AutoPeftModelForCausalLM

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

    # Create data module and return datasets
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
        # device_map="auto",
    )
    return model


# Load the model
# ckpt_path = '/root/MoRA/pub-med-qa/save_test_lora_rank128_lr1e-4/checkpoint-1200'
# model = load_model(ckpt_path)

def get_model_generations(
    model, tokenizer, prompts: List[str], max_new_tokens: int = 512
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