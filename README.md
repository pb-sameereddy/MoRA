# [MoRA: High-Rank Updating for Parameter-Efﬁcient Fine-Tuning](https://arxiv.org/abs/2405.12130)



## Setup

We implement MoRA in peft-mora based on HF peft in the [`apply_mora`](https://github.com/kongds/MoRA/blob/main/peft-mora/src/peft/tuners/lora/layer.py#L229) and [`get_delta_weight`](https://github.com/kongds/MoRA/blob/main/peft-mora/src/peft/tuners/lora/layer.py#L514).
``` sh
pip install -e ./peft-mora; pip install -r requirements.txt
```

After installation, it can be used like

``` python
from peft import LoraConfig, get_peft_model
config = LoraConfig(
    # enable MoRA
    use_mora=True,
    # type 1 (Sharing) for large lora ranks, Eq. 6 in paper
    # type 6 (RoPE based) for small lora ranks, Eq. 9 in paper
    mora_type=6,
    # lora rank here, we will calculate corresponding $\hat{r}$ in MoRA
    r=lora_r,
    # MoRA does not use lora_alpha
    # lora_alpha=lora_alpha,
    target_modules=lora_target_modules,
    lora_dropout=lora_dropout,
    task_type="CAUSAL_LM",
    **kwargs,
)
model = get_peft_model(model, config)

# training here...

# can be merged into model via `merge_and_unload` like LoRA
model = model.merge_and_unload() 
```

## Usage

### Training 

`train.py` is the entrypoint for running training. Run `python train.py --help` for a full list of arguments. 

`data_path` can be one of `meta-math`, `pub-med-cpt`, `pub-med-qa` corresponding to either the MetaMathQA or PubMedQA datasets on huggingface.

#### Additional info

MetaMathQA are GSM8K and MATH like word problems. 

In PubMedQA the model is given the user question and a relevant abstract and the goal is to provide a reasoning + answer (yes, no, or maybe). In `pub-med-cpt` the model is pretrained only on the abstracts. In `meta-math` and `pub-med-qa` the model is instruction tuned on the completions. 

Currently weights and biases is disabled manually in the training code. Metrics are saved to tensorboard, in a `logs` folder in the checkpoint dir which is derived from the `wandb_run_name` and saved to a folder name corresponding to the `data_path`. 

For the instruction tuning cases the data is manually limited to 100k samples and a 2% validation split. The different data paths also have different sequence lengths for training.

### Scripts

The `scripts/` folder contains scripts for dataset prep, generation from trained models, and evaluation.

When using these scripts use the syntax `python -m scripts.<script_name>` so that the scripts are run with the root folder as its path.

The shell scripts in the main folder are examples of multiple generations or evals.

## Examples

### Training on Pub med

These examples were for training 16 bit 1B llama models on 1xA5000 GPUs. This does instruction tuning on model completions. F 

LoRA with rank 256
``` sh
python train.py --base_model=meta-llama/Llama-3.2-1B-Instruct --batch_size=64 --micro_batch_size=4 --data_path=pub-med-qa --eval_steps=400 --save_steps=800 --use_16bit=True --use_bf16=True --lora_r=256 --use_mora=False  --lora_alpha=512 --learning_rate=1e-4 --wandb_run_name=${RUN_NAME}
```

MoRA
``` sh
python train.py --base_model=meta-llama/Llama-3.2-1B-Instruct --batch_size=64 --micro_batch_size=4 --data_path=pub-med-qa --eval_steps=400 --save_steps=800 --use_16bit=True --use_bf16=True --lora_r=256 --use_mora=True --mora_type=1 --learning_rate=1e-5 --wandb_run_name=${RUN_NAME}
```

Make sure to use the correct `mora_type` depending on low vs high lora ranks.

### Continued Pretraining

Setup the data. You should see a new dataset saved to a folder `datasets` in your project root..
```sh
python -m scripts.create_pub_med_cpt_dataset.py
```

Train
``` sh
python train.py --base_model=meta-llama/Llama-3.2-1B-Instruct --wandb_project=mora-pubmed --batch_size=64 --micro_batch_size=4 --use_16bit=True --use_bf16=True --use_mora=True --mora_type=1 --lora_r=256 --learning_rate=1e-5 ---data_path=pub-med-cpt --num_epochs=3 --eval_steps=400 --save_steps=1600 --wandb_run_name=mora_cpt_witheval_lr1e-5  
```

### Evaluation


Generate
```
 python generate_pub_med_cpt_eval_completions.py --subset labeled --ckpt_path ${CKPT_DIR}/checkpoint-${idx} --run_name ${RUN_NAME} --data_length 1000
```

The `labeled` vs `artificial` arguments refer to which subset of the PubMed dataset is sampled from for evaluation. The `run_name` argument will be used in defining the filename which generations are saved to. Provide a unique run name for each different set of weights. 

The `generate_pub_med` script is for generating completions in the fully supervised setting, where the abstract is also provided. In the CPT eval only the question is provided. The cpt script also has batched generation, which should be used in all generation, since it can be slow..

Evaluation, by comparing accuracy of the 'final_decision' to the model's final decision, is in `notebooks/eval_pub_med_cpt_generations.ipynb` for CPT `scripts/eval_pub_med_generations.py` for the instruction tuning evaluation. 