#!/bin/bash
# Hi-IGPO Phase 2b: interleaved Planner(LoRA, trained) <-> frozen Executor — Qwen3-4B, 4x V100.
# 2-step SMOKE. multi_agent.enable=true now means INTERLEAVED (one-shot DAG removed).
# Planner gets per-turn info-gain + final F1 (NOT shared F1); Executor LoRA is frozen.
#
# NOTE (WIP): requires the ray_trainer multi_agent branch rewired to InterleavedRolloutManager +
# scatter_planner_token_rewards + adv_estimator=igpo, and run_loop._assemble_planner_tensors
# finalized on server. This script is the intended config; validate end-to-end on GPU.
# Prereqs: SearXNG up (search returns results), Phase-1 model as Executor base, 4 idle GPUs.
set -euo pipefail

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PET_NODE_RANK=${PET_NODE_RANK:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3,4,6,7}   # pick 4 IDLE gpus (check nvidia-smi!)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export project_name="deepresearcher"
export experiment_name="qwen3_4b_hi_igpo_phase2b_smoke"
IFS=',' read -ra _D <<< "${CUDA_VISIBLE_DEVICES}"; NGPU=${#_D[@]}

BASE=/home/zjx/self_llm/self-researcher
cd ${BASE}
if [ -f .env ]; then set -a; . ./.env; set +a; fi
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
    algorithm.adv_estimator=igpo \
    +algorithm.use_vectorized_gt_logprob=true \
    +algorithm.info_gain_type=log_prob_diff \
    +algorithm.info_gain_norm_mode=separate \
    +algorithm.ig_group_mode=turn_group \
    algorithm.gamma=1.0 \
    actor_rollout_ref.model.path=${BASE}/models/Qwen3-4B-Instruct \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.enable_chunked_prefill=false \
    actor_rollout_ref.rollout.max_model_len=4200 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    ++actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=fp16 \
    ++actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=fp32 \
    ++actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype=fp32 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048 \
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
    multi_agent.enable=true \
    multi_agent.max_planner_turns=5 \
    multi_agent.freeze_executor=true \
    multi_agent.base_model=${BASE}/models/Qwen3-4B-Instruct \
    multi_agent.lora_save_dir=${BASE}/tmp_lora_adapters \
    multi_agent.lora.rank=64 \
    multi_agent.lora.alpha=128 \
    multi_agent.lora.dropout=0.05 \
    multi_agent.agents.executor.max_turns=5 \
    agent_grpo.n=2 \
    max_turns=5 \
    search_engine=online_search \
    trainer.total_epochs=1 \
    trainer.total_training_steps=2 2>&1 | tee ${BASE}/${project_name}_${experiment_name}.log
