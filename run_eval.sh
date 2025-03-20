#!/bin/bash
# Enable printing of commands before execution
set -x

# Exit immediately if a command exits with a non-zero status
set -e

# Set base checkpoint directory
# CKPT_DIR="/root/MoRA/pub-med-qa/test_lora_rank8_lr2e-4"
CKPT_DIR="/root/MoRA/pub-med-qa/lora_rank128_lr1e-4_witheval"
# split ckpt dir and get base name
RUN_NAME_BASE=$(basename ${CKPT_DIR})

# List of checkpoint indices to evaluate
IDX=(400 800 1200 1531)

# PQA LABELED #
for idx in "${IDX[@]}"; do
    RUN_NAME=${RUN_NAME_BASE}_ckpt${idx}_test
    python eval.py --subset pqa_labeled --ckpt_path ${CKPT_DIR}/checkpoint-${idx} --run_name ${RUN_NAME} --data_length 1000
done

# ------------------------------------------------------------------------------------------------

# PQA ARTIFICIAL #
for idx in "${IDX[@]}"; do
    RUN_NAME=${RUN_NAME_BASE}_ckpt${idx}_train
    python eval.py --subset pqa_artificial --ckpt_path ${CKPT_DIR}/checkpoint-${idx} --run_name ${RUN_NAME} --data_length 1000
done
