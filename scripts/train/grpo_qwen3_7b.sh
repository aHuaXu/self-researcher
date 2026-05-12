#!/bin/bash
# Stage 1: Single-agent GRPO training — Qwen3-7B-Instruct, 4 GPU (V100 32GB)
# Goal: teach the model web_search tool-calling ability.
#
# Prerequisites:
#   - Qwen3-7B-Instruct downloaded to ./models/Qwen3-7B-Instruct
#   - online_search handler running (see CLAUDE.md)
#
# After training, export checkpoint for Stage 2:
#   python scripts/export_fsdp_to_hf.py \
#     --ckpt_dir  ./ckpts/deepresearcher/qwen3_7b_grpo/global_step_<N>/actor \
#     --base_model ./models/Qwen3-7B-Instruct \
#     --output_dir ./ckpts/deepresearcher/qwen3_7b_grpo/exported_hf
#   # Multi-GPU export (must match training world_size=4):
#   torchrun --nproc_per_node=4 scripts/export_fsdp_to_hf.py \
#     --ckpt_dir  ./ckpts/deepresearcher/qwen3_7b_grpo/global_step_<N>/actor \
#     --base_model ./models/Qwen3-7B-Instruct \
#     --output_dir ./ckpts/deepresearcher/qwen3_7b_grpo/exported_hf
set -euo pipefail

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PET_NODE_RANK=${PET_NODE_RANK:-0}
export CUDA_VISIBLE_DEVICES=0,1,2,5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export project_name="deepresearcher"
export experiment_name="qwen3_7b_grpo"

BASE=/home/zjx/ahua_llm/self-researcher
cd ${BASE}

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=${BASE}/data/train.parquet \
    data.val_files=${BASE}/data/dev.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=10240 \
    data.max_response_length=2000 \
    +data.max_model_len=12240 \
    data.data_writing_file=${BASE}/signal/data.json \
    data.signal_writing_file=${BASE}/signal/signal.json \
    actor_rollout_ref.model.path=${BASE}/models/Qwen3-7B-Instruct \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.enable_chunked_prefill=false \
    actor_rollout_ref.rollout.max_num_batched_tokens=40000 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['console'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.remove_previous_ckpt_in_save=true \
    agent_grpo.n=16 \
    max_turns=5 \
    search_engine=online_search \
    trainer.total_epochs=1 2>&1 | tee ${BASE}/${project_name}_${experiment_name}.log
