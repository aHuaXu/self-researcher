#!/bin/bash
# Planner-only cold-start SFT for Hi-IGPO.
#
# This is separate from scripts/train/igpo_drvenus_sft.sh:
#   - igpo_drvenus_sft.sh: DR-Venus-4B-SFT -> single-agent IGPO/RL.
#   - this script: base/instruct model -> Planner LoRA using multi-turn planner messages.
#
# Expected data schema:
#   data/planner_sft/train.parquet and val.parquet with a `messages` column.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

BASE=${BASE:-/home/zjx/self_llm/self-researcher}
cd "${BASE}"

source /home/zjx/anaconda3/bin/activate deepresearcher

IFS=',' read -ra _CUDA_DEVICES_ARR <<< "${CUDA_VISIBLE_DEVICES}"
NGPU=${#_CUDA_DEVICES_ARR[@]}

# Planner SFT base MUST match the RL Planner base (single thinking base + dual LoRA in
# hi_igpo_phase2b_drvenus.sh). DR-Venus-4B-RL is Qwen3-4B-Thinking-2507 (thinking-only).
# Using Qwen3-4B-Instruct here would break LoRA transfer (different base weights) and
# mismatch the chat template (Instruct does not auto-open think). See design §7.1.1.1.
MODEL_PATH=${MODEL_PATH:-${BASE}/models/DR-Venus-4B-RL}
TRAIN_FILE=${TRAIN_FILE:-${BASE}/data/planner_sft/train.parquet}
VAL_FILE=${VAL_FILE:-${BASE}/data/planner_sft/val.parquet}
SAVE_DIR=${SAVE_DIR:-${BASE}/ckpts/deepresearcher/planner_sft_deepresearch}

torchrun --standalone --nnodes=1 --nproc_per_node="${NGPU}" \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size=${TRAIN_BATCH_SIZE:-16} \
  data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU:-1} \
  data.max_length=${MAX_LENGTH:-8192} \
  data.truncation=${TRUNCATION:-error} \
  +data.multiturn.enable=true \
  +data.multiturn.messages_key=messages \
  model.partial_pretrain="${MODEL_PATH}" \
  model.enable_gradient_checkpointing=true \
  model.lora_rank=${LORA_RANK:-64} \
  model.lora_alpha=${LORA_ALPHA:-128} \
  model.target_modules=all-linear \
  model.fsdp_config.cpu_offload=${CPU_OFFLOAD:-true} \
  model.fsdp_config.offload_params=${OFFLOAD_PARAMS:-true} \
  optim.lr=${LR:-1e-5} \
  trainer.default_local_dir="${SAVE_DIR}" \
  trainer.default_hdfs_dir=null \
  trainer.project_name=deepresearcher \
  trainer.experiment_name=planner_sft_deepresearch \
  trainer.logger=${LOGGER:-"['console']"} \
  trainer.total_epochs=${TOTAL_EPOCHS:-2} \
  trainer.total_training_steps=${TOTAL_STEPS:-null} \
  use_remove_padding=${USE_REMOVE_PADDING:-false} \
  ulysses_sequence_parallel_size=${ULYSSES_SEQUENCE_PARALLEL_SIZE:-1}
