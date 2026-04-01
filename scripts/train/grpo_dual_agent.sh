#!/bin/bash
# Dual-agent (Planner + Executor) GRPO training — Qwen3-4B-Instruct, 4 GPU (V100 32GB)
# Goal: train planner decomposition + executor search via DAG-based multi-turn RL.
#
# Phase 1: L1+L2 data (medium difficulty)
# Phase 2: L2+L3 data (hard difficulty) — change train/val files below
#
# Prerequisites:
#   - Qwen3-4B-Instruct under ./models/Qwen3-4B-Instruct
#   - Data prepared via: python scripts/prepare_deepresearch_data.py
#   - Repo-root .env: SERPER_API_KEY; for browse_webpage also LLM_API_KEY
set -euo pipefail

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PET_NODE_RANK=${PET_NODE_RANK:-0}
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export project_name="deepresearcher"
export experiment_name_base="qwen3_4b_dual_agent"
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

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=${BASE}/data/deepresearch_phase1.parquet \
    data.val_files=${BASE}/data/deepresearch_phase1_val.parquet \
    data.train_batch_size=4 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    max_seq_len_for_training=4096 \
    actor_rollout_ref.model.path=${BASE}/models/Qwen3-4B-Instruct \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
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
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
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
    multi_agent.enable=true \
    multi_agent.base_model=${BASE}/models/Qwen3-4B-Instruct \
    multi_agent.lora_save_dir=${BASE}/tmp_lora_adapters \
    multi_agent.lora.rank=64 \
    multi_agent.lora.alpha=128 \
    multi_agent.lora.dropout=0.05 \
    multi_agent.agents.planner.max_tokens=1024 \
    multi_agent.agents.executor.max_turns=5 \
    multi_agent.reward.alpha=0.2 \
    multi_agent.reward.beta=0.3 \
    multi_agent.reward.max_turns=5 \
    agent_grpo.n=2 \
    max_turns=5 \
    search_engine=online_search \
    trainer.total_epochs=1 \
    trainer.total_training_steps=2 2>&1 | tee ${BASE}/${project_name}_${experiment_name}.log
