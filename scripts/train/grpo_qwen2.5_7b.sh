#!/bin/bash
# Stage 1: Single-agent GRPO training — Qwen2.5-7B-Instruct, 6 GPU (V100 32GB)
# Goal: teach the model web_search tool-calling ability.
#
# Prerequisites:
#   - Qwen2.5-7B-Instruct under ./models/Qwen2.5-7B-Instruct
#   - Repo-root .env: SERPER_API_KEY; for browse_webpage also LLM_API_KEY or JUDGE_API_KEY
#     (same MiniMax judge endpoint as scripts/train/multi_agent_0.5b.sh — see LLM_* exports below)
#
# After training, export checkpoint for Stage 2:
#   python scripts/export_fsdp_to_hf.py \
#     --ckpt_dir  ./ckpts/deepresearcher/qwen2.5_7b_grpo/global_step_<N>/actor \
#     --base_model ./models/Qwen2.5-7B-Instruct \
#     --output_dir ./ckpts/deepresearcher/qwen2.5_7b_grpo/exported_hf
#   # Multi-GPU export (must match training world_size=6):
#   torchrun --nproc_per_node=6 scripts/export_fsdp_to_hf.py \
#     --ckpt_dir  ./ckpts/deepresearcher/qwen2.5_7b_grpo/global_step_<N>/actor \
#     --base_model ./models/Qwen2.5-7B-Instruct \
#     --output_dir ./ckpts/deepresearcher/qwen2.5_7b_grpo/exported_hf
set -euo pipefail

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PET_NODE_RANK=${PET_NODE_RANK:-0}
# 使用空闲 GPU 1-6（避开 0 号卡）
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# NCCL: disable InfiniBand/RDMA; must be set before ray start so workers inherit
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export project_name="deepresearcher"
export experiment_name="qwen2.5_7b_grpo"

BASE=/home/zjx/ahua_llm/self-researcher
cd ${BASE}

# Load repo-root .env (SERPER_API_KEY, JUDGE_API_KEY / LLM_API_KEY, etc.) if present
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

# Ray CLI must be on PATH (same as interactive training on the server)
source /home/zjx/anaconda3/bin/activate deepresearcher

# Tool-side LLM (browse_webpage / ReadingAgent via research_agent.config): same defaults as
# multi_agent reward judge in scripts/train/multi_agent_0.5b.sh (MiniMax OpenAI-compatible API).
# Set LLM_API_KEY or JUDGE_API_KEY in repo-root .env or export before running.
export LLM_BASE_URL="${LLM_BASE_URL:-https://api.minimaxi.com/v1}"
export LLM_MODEL="${LLM_MODEL:-MiniMax-M2.7}"
export LLM_API_KEY="${LLM_API_KEY:-${JUDGE_API_KEY:-}}"

# Restart Ray so workers inherit the correct CUDA_VISIBLE_DEVICES and NCCL settings
ray stop --force 2>/dev/null || true
sleep 2
export RAY_memory_monitor_refresh_ms=0
ray start --head
sleep 2

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=${BASE}/data/train.parquet \
    data.val_files=${BASE}/data/dev.parquet \
    data.train_batch_size=18 \
    data.max_prompt_length=10240 \
    data.max_response_length=1000 \
    +data.max_model_len=12240 \
    actor_rollout_ref.model.path=${BASE}/models/Qwen2.5-7B-Instruct \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=18 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.20 \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.enable_chunked_prefill=false \
    actor_rollout_ref.rollout.max_model_len=12240 \
    actor_rollout_ref.rollout.max_num_batched_tokens=12240 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    ++actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=bf16 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8000 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    trainer.logger=['console'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=6 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.remove_previous_ckpt_in_save=true \
    agent_grpo.n=8 \
    max_turns=3 \
    search_engine=online_search \
    trainer.total_epochs=1 2>&1 | tee ${BASE}/${project_name}_${experiment_name}.log
