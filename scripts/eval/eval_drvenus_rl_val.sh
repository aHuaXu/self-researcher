#!/bin/bash
# Evaluate DR-Venus-4B-RL on the DeepResearch-9K val set (single-agent, no training).
# Runs only _validate() once (val_before_train=true, total_training_steps=0): DR-Venus native
# rollout (search_engine=drvenus + enable_think) then F1/EM scoring, bucketed per difficulty
# (val/test_score/deepresearch_L{1,2,3}_{f1,em}). See docs/design/hi_igpo_design.md §13.
#
# VAL_FILE selects the eval set (default: 96-sample all_val_small = 32 per difficulty).
# Set VAL_FILE=data/deepresearch_all_val.parquet for the full 1298-sample run.
# Prereqs: models/DR-Venus-4B-RL, SearXNG up, idle GPUs.
set -euo pipefail

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PET_NODE_RANK=${PET_NODE_RANK:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2,3,4}   # pick IDLE gpus (check nvidia-smi!)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export project_name="deepresearcher"
export experiment_name="drvenus_rl_eval"
IFS=',' read -ra _D <<< "${CUDA_VISIBLE_DEVICES}"; NGPU=${#_D[@]}

BASE=/home/zjx/self_llm/self-researcher
cd ${BASE}
if [ -f .env ]; then set -a; . ./.env; set +a; fi
source /home/zjx/anaconda3/bin/activate deepresearcher
export LLM_BASE_URL="${LLM_BASE_URL:-https://api.minimaxi.com/v1}"
export LLM_MODEL="${LLM_MODEL:-MiniMax-M2.7}"
export LLM_API_KEY="${LLM_API_KEY:-${JUDGE_API_KEY:-}}"

VAL_FILE=${VAL_FILE:-data/deepresearch_all_val_small.parquet}
DRV=${BASE}/models/DR-Venus-4B-RL

ray stop --force 2>/dev/null || true
sleep 2
export RAY_memory_monitor_refresh_ms=0
ray start --head
sleep 2

# total_training_steps=0 -> only val_before_train runs, then exit (pure evaluation).
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=${BASE}/data/deepresearch_phase1.parquet \
    data.val_files=${BASE}/${VAL_FILE} \
    data.train_batch_size=4 \
    data.max_prompt_length=3096 \
    data.max_response_length=2048 \
    max_seq_len_for_training=7168 \
    algorithm.adv_estimator=igpo \
    +algorithm.use_vectorized_gt_logprob=true \
    +algorithm.info_gain_type=log_prob_diff \
    +algorithm.info_gain_norm_mode=separate \
    +algorithm.enable_think=true \
    algorithm.gamma=1.0 \
    actor_rollout_ref.model.path=${DRV} \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=7168 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.enable_chunked_prefill=false \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    ++actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=fp16 \
    ++actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=fp32 \
    ++actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype=fp32 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=7168 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    trainer.logger=['console'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=${NGPU} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.resume_mode=disable \
    agent_grpo.n=8 \
    max_turns=6 \
    search_engine=drvenus \
    trainer.total_training_steps=0 \
    trainer.total_epochs=1 2>&1 | tee ${BASE}/${project_name}_${experiment_name}.log
