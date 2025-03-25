import json
import os
from training_utils import make_supervised_data_module
from dataclasses import dataclass
from collections import namedtuple
from eval_utils import load_data, get_model_generations, load_model

CKPT_PATH = "/root/MoRA/meta-math/save_mora_rank128_lr2.5e-5/checkpoint-1562"
save_root = "meta-math-eval/save_mora_rank128_lr2.5e-5_checkpoint-1562"
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
    data_length=None,
    val_split=0.0,
    subset=None,
)

# ckpt model generations
model = load_model(CKPT_PATH)
prompts = [data_module["train_dataset"][i]["input_ids"] for i in range(len(data_module["train_dataset"]))]
generations = get_model_generations(model, tokenizer, prompts, max_new_tokens=100)

with open(os.path.join(save_root, "gsm8k_generations.json"), "w") as f:
    json.dump(generations, f)

# base model generations
print("Generating base model outputs")
model.disable_adapters()
base_model_generations = get_model_generations(model, tokenizer, prompts)
with open('meta-math-eval/gsm8k_base_model_generations.json', 'w') as f:
    json.dump(base_model_generations, f)
