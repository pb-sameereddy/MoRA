from training_utils import make_supervised_data_module
from dataclasses import dataclass
from collections import namedtuple
import transformers
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

# data_args = setup_data_args(
#     data_path="openai/gsm8k",
#     base_model="meta-llama/Meta-Llama-3-8B-Instruct",
#     data_length=10,
#     val_split=0.0,
#     subset=None,
# )
data_module, tokenizer = load_data(
    data_path="openai/gsm8k",
    base_model="meta-llama/Llama-3.2-1B-Instruct",
    data_length=10,
    val_split=0.0,
    subset=None,
)
import pdb; pdb.set_trace()