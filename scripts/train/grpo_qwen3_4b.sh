#!/bin/bash
# Stage 1: Single-agent GRPO training — Qwen3-4B-Instruct, 4 GPU (V100 32GB)
# Goal: teach the model web_search tool-calling ability.
#
# Prerequisites:
#   - Qwen3-4B-Instruct under ./models/Qwen3-4B-Instruct
#   - Repo-root .env: SERPER_API_KEY; for browse_webpage also LLM_API_KEY or JUDGE_API_KEY
set -euo pipefail

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PET_NODE_RANK=${PET_NODE_RANK:-0}
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export project_name="deepresearcher"
export experiment_name_base="qwen3_4b_grpo"
IFS=',' read -ra _CUDA_DEVICES_ARR <<< "${CUDA_VISIBLE_DEVICES}"
export experiment_name="${experiment_name_base}_ws${#_CUDA_DEVICES_ARR[@]}"

BASE=/home/zjx/self_llm/self-researcher
cd ${BASE}

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

source /home/zjx/anaconda3/bin/activate deepresearcher

export LLM_BASE_URL="${LLM_BASE_URL:-https://api.minimaxi.com/v1}"
export LLM_MODEL="${LLM_MODEL:-MiniMax-M2.7}"
export LLM_API_KEY="${LLM_API_KEY:-${JUDGE_API_KEY:-}}"

ray stop --force 2>/dev/null || true
sleep 2
export RAY_memory_monitor_refresh_ms=0
ray start --head
sleep 2

# Prompt clamp in data pipeline; consumed by rollout.prompt_length via config mapping.
# Code: verl/trainer/config/ppo_trainer.yaml (rollout.prompt_length: ${data.max_prompt_length})
#
# Single-turn generation cap in rollout sampling (max_tokens).
# Code: verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py (SamplingParams max_tokens=config.response_length)
#
# Final training-side clamp after rollout text is assembled (pre-PPO update safety valve).
# Code: scrl/llm_agent/generation.py (self.config.max_seq_len_for_training)
#
# vLLM per-request context limit (KV cache capacity gate).
# Code: verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py (LLM max_model_len=...)
#
# vLLM scheduler token budget per batch (throughput vs memory trade-off).
# Code: verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py (LLM max_num_batched_tokens=...)
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=${BASE}/data/train.parquet \
    data.val_files=${BASE}/data/dev.parquet \
    data.train_batch_size=60 \
    data.max_prompt_length=3096 \
    data.max_response_length=1000 \
    max_seq_len_for_training=5000 \
    actor_rollout_ref.model.path=${BASE}/models/Qwen3-4B-Instruct \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=480 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.enable_chunked_prefill=false \
    actor_rollout_ref.rollout.max_model_len=6144 \
    actor_rollout_ref.rollout.max_num_batched_tokens=12288 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    ++actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=fp16 \
    ++actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=fp32 \
    ++actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype=fp32 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=5120 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    trainer.logger=['console'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.resume_mode=auto \
    trainer.test_freq=5 \
    trainer.remove_previous_ckpt_in_save=true \
    agent_grpo.n=8 \
    max_turns=5 \
    search_engine=online_search \
    trainer.total_epochs=3 2>&1 | tee ${BASE}/${project_name}_${experiment_name}.log
