from training_utils import make_supervised_data_module
from dataclasses import dataclass
from collections import namedtuple
import transformers
from eval_utils import load_data

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