#!/bin/bash
# Enable printing of commands before execution
set -x

# Exit immediately if a command exits with a non-zero status
set -e

# Set base checkpoint directory
CKPT_DIR="/root/MoRA/pub-med-qa/lora_rank128_lr1e-4_witheval"

# List of checkpoint indices to evaluate
# IDX=(400 800 1200 1531)
IDX=(800 1200 1531)

# PQA LABELED #
for idx in "${IDX[@]}"; do
    python eval.py --subset pqa_labeled --ckpt_path ${CKPT_DIR}/checkpoint-${idx} --run_name lora_rank128_lr1e-4_witheval_ckpt${idx}_test --data_length 1000
done

# ------------------------------------------------------------------------------------------------

# PQA ARTIFICIAL #
for idx in "${IDX[@]}"; do
    python eval.py --subset pqa_artificial --ckpt_path ${CKPT_DIR}/checkpoint-${idx} --run_name lora_rank128_lr1e-4_witheval_ckpt${idx}_train --data_length 1000
done
