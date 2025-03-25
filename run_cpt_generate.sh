#!/bin/bash
# Enable printing of commands before execution
set -x

# Exit immediately if a command exits with a non-zero status
set -e

# Set base checkpoint directory
# CKPT_DIR="/root/MoRA/pub-med-qa/test_lora_rank8_lr2e-4"
CKPT_DIR="/root/MoRA/pub-med-cpt/mora_cpt_witheval"
# split ckpt dir and get base name
RUN_NAME_BASE=$(basename ${CKPT_DIR})

# List of checkpoint indices to evaluate
# IDX=(800 1200 1531)
IDX=(3200 6400 9600)

#PQA LABELED
for idx in "${IDX[@]}"; do
    RUN_NAME="cpt/${RUN_NAME_BASE}_ckpt${idx}_rerun2" # e.g. cpt/lora_cpt_witheval_checkpoint-3200
    python generate_pub_med_cpt_eval_completions.py --subset labeled --ckpt_path ${CKPT_DIR}/checkpoint-${idx} --run_name ${RUN_NAME} --data_length 1000
done

