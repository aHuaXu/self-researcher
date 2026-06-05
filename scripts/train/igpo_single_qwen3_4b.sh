#!/bin/bash
# Hi-IGPO Phase 1: single-agent IGPO (turn-level info-gain) — Qwen3-4B-Instruct, V100 32GB.
# 2-step SMOKE config. Validates the full IGPO path: igpo_generation rollout + belief
# (vectorized GT logprob) + per-turn IG/F1 scatter (info_gain.py) + turn-level advantage.
#
# Prereqs: models/Qwen3-4B-Instruct, .env (SERPER_API_KEY [+ LLM/JUDGE key for browse]).
# Pick idle GPUs (check nvidia-smi first!). Defaults to 2,3,5,7.
set -euo pipefail

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PET_NODE_RANK=${PET_NODE_RANK:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3,5,7}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export project_name="deepresearcher"
IFS=',' read -ra _CUDA_DEVICES_ARR <<< "${CUDA_VISIBLE_DEVICES}"
NGPU=${#_CUDA_DEVICES_ARR[@]}
export experiment_name="qwen3_4b_igpo_smoke"

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

# IGPO knobs:
#   algorithm.adv_estimator=igpo                         -> turn-level IG advantage (compute_igpo_turn_advantage)
#   +algorithm.use_vectorized_gt_logprob=true            -> enable belief (GT logprob) → per-turn IG
#   +algorithm.info_gain_type=log_prob_diff              -> IG = Δ mean log-prob of golden answer
#   +algorithm.info_gain_norm_mode=separate              -> IG / F1 normalized separately
#   algorithm.gamma=1.0                                  -> turn discount
# (ig_group_mode defaults to 'global' = faithful IGPO; turn_group is a later ablation.)
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=${BASE}/data/train.parquet \
    data.val_files=${BASE}/data/dev.parquet \
    data.train_batch_size=8 \
    data.max_prompt_length=3096 \
    data.max_response_length=1000 \
    max_seq_len_for_training=5000 \
    algorithm.adv_estimator=igpo \
    +algorithm.use_vectorized_gt_logprob=true \
    +algorithm.info_gain_type=log_prob_diff \
    +algorithm.info_gain_norm_mode=separate \
    algorithm.gamma=1.0 \
    actor_rollout_ref.model.path=${BASE}/models/Qwen3-4B-Instruct \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
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
    trainer.n_gpus_per_node=${NGPU} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.resume_mode=disable \
    agent_grpo.n=4 \
    max_turns=5 \
    search_engine=online_search \
    trainer.total_training_steps=2 \
    trainer.total_epochs=1 2>&1 | tee ${BASE}/${project_name}_${experiment_name}.log
