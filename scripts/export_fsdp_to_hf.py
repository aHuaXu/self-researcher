#!/usr/bin/env python3
"""
Export a FSDP checkpoint (from train_grpo.sh Stage 1) to HuggingFace format.
This enables using Stage 1 output as the base model for Stage 2 multi-agent training.

Each rank loads its own shard file, mirroring FSDPCheckpointManager.load_checkpoint.
FULL_STATE_DICT is then collected on rank 0 to write model.safetensors.

Usage:
    # Single GPU (world_size=1)
    python scripts/export_fsdp_to_hf.py \
        --ckpt_dir  ./ckpts/deepresearcher/grpo_exp/global_step_200/actor \
        --base_model ./models/Qwen2.5-0.5B-Instruct \
        --output_dir ./ckpts/deepresearcher/grpo_exp/exported_hf

    # Multi GPU  (world_size=N, must match the world_size used during training)
    torchrun --nproc_per_node=N scripts/export_fsdp_to_hf.py \
        --ckpt_dir  ./ckpts/deepresearcher/grpo_exp/global_step_200/actor \
        --base_model ./models/Qwen2.5-0.5B-Instruct \
        --output_dir ./ckpts/deepresearcher/grpo_exp/exported_hf

Then in train_multi_agent.sh set:
    actor_rollout_ref.model.path=<output_dir>
"""

import argparse
import os
import warnings

import torch
import torch.distributed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
from torch.distributed.fsdp import FullStateDictConfig, ShardedStateDictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='Export FSDP checkpoint to HuggingFace format')
    parser.add_argument('--ckpt_dir', required=True,
                        help='Actor checkpoint directory containing model_world_size_N_rank_K.pt')
    parser.add_argument('--base_model', required=True,
                        help='Base HF model path for structure initialization')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for the exported HF model')
    parser.add_argument('--dtype', default='auto', choices=['auto', 'float16', 'bfloat16', 'float32'],
                        help='Model dtype (default: auto → bfloat16 on CUDA, float32 on CPU)')
    return parser.parse_args()


def init_distributed():
    """Initialize torch.distributed.

    - If torchrun env vars are present (RANK / WORLD_SIZE), use them directly.
    - Otherwise fall back to single-process init (world_size=1, rank=0).
    """
    if torch.distributed.is_initialized():
        return

    if 'RANK' in os.environ:
        # Launched via torchrun: env vars already set by the launcher
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        torch.distributed.init_process_group(backend=backend)
    else:
        # Plain python invocation → single process
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29501')
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        torch.distributed.init_process_group(backend=backend, world_size=1, rank=0)


def main():
    args = parse_args()
    init_distributed()

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', rank))

    if rank == 0:
        print('=' * 60)
        print('FSDP → HuggingFace Checkpoint Export')
        print('=' * 60)
        print(f'  ckpt_dir   : {args.ckpt_dir}')
        print(f'  base_model : {args.base_model}')
        print(f'  output_dir : {args.output_dir}')
        print(f'  world_size : {world_size}')
        print()

    # ── Step 1: validate this rank's shard file ──────────────────────
    # Mirrors FSDPCheckpointManager.load_checkpoint file naming convention
    ckpt_path = os.path.join(args.ckpt_dir, f'model_world_size_{world_size}_rank_{rank}.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f'[rank {rank}] Checkpoint shard not found: {ckpt_path}\n'
            f'Make sure --nproc_per_node matches the world_size used during training.'
        )
    if rank == 0:
        print(f'[1/5] Checkpoint shard found for each rank  '
              f'(e.g. {os.path.basename(ckpt_path)})')

    # ── Step 2: device setup ─────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device_id = local_rank
    else:
        device_id = None
    if rank == 0:
        print(f'[2/5] Device: {"cuda:" + str(local_rank) if device_id is not None else "cpu"}')

    # ── Step 3: load model structure (rank 0 loads real weights) ─────
    if rank == 0:
        print(f'[3/5] Loading model structure from {args.base_model}...')

    if args.dtype == 'auto':
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    else:
        torch_dtype = getattr(torch, args.dtype)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    # FSDP wrap — sync_module_states=True broadcasts rank 0 weights to all ranks
    model_fsdp = FSDP(
        model,
        device_id=device_id,
        use_orig_params=True,
        sync_module_states=True,
    )

    # ── Step 4: load FSDP sharded checkpoint (each rank loads its shard) ──
    if rank == 0:
        print(f'[4/5] Loading FSDP sharded checkpoint (each rank loads its own shard)...')

    model_state_dict = torch.load(ckpt_path, map_location='cpu')
    state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with FSDP.state_dict_type(model_fsdp, StateDictType.SHARDED_STATE_DICT, state_dict_cfg):
            model_fsdp.load_state_dict(model_state_dict)
    del model_state_dict

    # ── Step 5: collect full state dict on rank 0 and save ───────────
    if rank == 0:
        print(f'[5/5] Collecting full weights on rank 0 and saving to {args.output_dir}...')

    full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with FSDP.state_dict_type(model_fsdp, StateDictType.FULL_STATE_DICT, full_cfg):
            full_state_dict = model_fsdp.state_dict()

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        # Build a clean (non-FSDP) model from config to avoid flat_param shape issues
        # that arise when loading into _fsdp_wrapped_module on multi-GPU (world_size > 1).
        clean_model = AutoModelForCausalLM.from_config(model_fsdp._fsdp_wrapped_module.config)
        clean_model.load_state_dict(full_state_dict)
        clean_model.save_pretrained(args.output_dir, safe_serialization=True)
        del full_state_dict, clean_model

        # Copy tokenizer — prefer checkpoint's huggingface/ subdir, fallback to base_model
        hf_subdir = os.path.join(args.ckpt_dir, 'huggingface')
        tokenizer_src = hf_subdir if os.path.isdir(hf_subdir) else args.base_model
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
        tokenizer.save_pretrained(args.output_dir)

        print()
        print('=' * 60)
        print(f'Export complete: {args.output_dir}')
        print()
        print('Next step — set in train_multi_agent.sh:')
        print(f'  actor_rollout_ref.model.path={os.path.abspath(args.output_dir)}')
        print('=' * 60)

    torch.distributed.barrier()


if __name__ == '__main__':
    main()
