#!/bin/bash
set -euo pipefail

export PET_NODE_RANK=${PET_NODE_RANK:-0}
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export CUDA_VISIBLE_DEVICES=7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export project_name="multi_agent_research"
export experiment_name="multi_agent_lora_0.5b_test"

cd /home/zjx/ahua_llm/self-researcher

# Clean up any stale Ray cluster to avoid placement group naming conflicts
ray stop --force 2>/dev/null || true
sleep 3
ray start --head 2>&1 | tail -2

BASE=/home/zjx/ahua_llm/self-researcher

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=${BASE}/data/multi-research.parquet \
    data.val_files=${BASE}/data/multi-research_dev.parquet \
    data.train_batch_size=4 \
    data.max_prompt_length=1024 \
    data.max_response_length=768 \
    +data.max_model_len=2560 \
    data.data_writing_file=${BASE}/signal/data.json \
    data.signal_writing_file=${BASE}/signal/signal.json \
    actor_rollout_ref.model.path=/home/zjx/ahua_llm/self-researcher/models/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.1 \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.enable_chunked_prefill=false \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['console'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.remove_previous_ckpt_in_save=true \
    agent_grpo.n=4 \
    max_turns=10 \
    search_engine=rag \
    multi_agent.enable=true \
    multi_agent.lora.rank=64 \
    multi_agent.lora.alpha=128 \
    multi_agent.reward.judge_model="MiniMax-M2.7" \
    multi_agent.reward.judge_base_url="https://api.minimaxi.com/v1" \
    multi_agent.reward.judge_api_key="sk-cp-ls65iNzF3RxCUXDv0HpObet6FyczQAQYIEyJA-W7mPuqqMQ9qLeD-x6CL-9UcqOC6AhZ8u-m1W7qKEqxEjUHbgZT6imI2pQW-vHQGDGS5DaKyllhVfIUGpM" \
    multi_agent.reward.judge_max_concurrent=10 \
    multi_agent.reward.alpha=0.2 \
    multi_agent.reward.beta=0.3 \
    trainer.total_epochs=1 2>&1 | tee ./${project_name}_${experiment_name}.log