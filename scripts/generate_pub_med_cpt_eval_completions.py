# %%
from types import SimpleNamespace
import pathlib
from typing import Dict, List
import torch
from tqdm import tqdm
from eval_utils import load_model, get_model_generations, get_eval_accuracy, get_eval_loss
import json
import argparse
import datasets
from transformers import AutoTokenizer
from training_utils import DataCollatorForCausalLM

# %%
# Parse command line arguments
parser = argparse.ArgumentParser(description="Evaluate model on PubMedQA dataset")
parser.add_argument(
    "--subset",
    type=str,
    default="labeled",
    choices=["labeled", "artificial"],
    help="Subset of PubMedQA to use",
)
parser.add_argument(
    "--ckpt_path", type=str, required=True, help="Path to model checkpoint"
)
parser.add_argument(
    "--run_name",
    type=str,
    required=True,
    help="Run name, under pub-med-eval/ e.g pqa_labeled_results will be saved under pub-med-eval/pqa_labeled_results.json. Can have path prefix.",
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


# %%
# Setup
run_name = args.run_name 
subset = args.subset
ckpt_path = args.ckpt_path
save_path = pathlib.Path(f"pub-med-eval-v2/{args.run_name}_subset={subset}.json")
print(f"Results will be saved to {save_path}")
if not save_path.exists():
    save_path.parent.mkdir(parents=True, exist_ok=True)
subset = f"pqa_{subset}"


# %%
# Setup data (load, prompt,tokenize)
data = datasets.load_dataset('qiaojin/PubMedQA', subset, split="train")
questions = data['question']
reasoning = data['long_answer']
answer = data['final_decision'] # one of 'yes', 'no', 'maybe'

# Must have left padding
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", padding_side="left")
tokenizer.pad_token_id = (
    # NOTE: set this to eos token, set to unk(0) while make output nan
    2  # unk. we want this to be different from the eos token
)

prompts = [[
    {"role": "system", "content": "You are a helpful assistant that can answer questions about a medical paper. Always respond with a short explanation for your decision followed by a final decision which can be one of 'yes', 'no', 'maybe'.\nYour response should always end with the format 'Reasoning: <explanation>\nFinal Decision:<decision>'"},
    {"role": "user", "content": f"{question}"},
] for question in questions]

print('Tokenizing and batching...')
bsz = 64
input_batches = []
for i in tqdm(range(len(prompts) // bsz + 1)):
    batch = prompts[i*bsz:(i+1)*bsz]
    if batch:
        tokenized_batch = tokenizer.apply_chat_template(batch, tokenize=True, return_tensors="pt", padding='longest')
        input_batches.append(tokenized_batch)
        # print(tokenized_batch.shape)

# %%
# Load model
print(f"Loading model from {args.ckpt_path}...")
model = load_model(args.ckpt_path)

# %%
# Generate completions
generated_ids = []
for i, batch in enumerate(tqdm(input_batches, desc="Generating completions...")):
    batch_generated_ids = model.generate(batch.to(model.device), max_new_tokens=512, pad_token_id=tokenizer.pad_token_id)
    generated_ids.extend(batch_generated_ids)
    
    # Save intermediate results every 10 batches
    if (i + 1) % 10 == 0 or i == len(input_batches) - 1:
        generations_str = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Save all results
        results = {
            "generations_str": generations_str,
            "subset": subset,
            "ckpt_path": ckpt_path,
            "total_generations": len(generated_ids),
            "batch_idx": i,
        }
        
        # Save results to json
        with open(save_path, "w") as f:
            json.dump(results, f)
            print(f"Saved results after {i+1}/{len(input_batches)} batches")
