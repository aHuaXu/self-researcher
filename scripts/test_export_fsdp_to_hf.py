#!/usr/bin/env python3
"""
Smoke tests for export_fsdp_to_hf.py.

Single GPU (default):
    python scripts/test_export_fsdp_to_hf.py

Multi GPU (requires 2 free GPUs, must match world_size used for saving):
    torchrun --nproc_per_node=2 scripts/test_export_fsdp_to_hf.py --mode multi_gpu

Both modes:
    1. Create a tiny synthetic model and simulate a training step (add noise).
    2. Save a FSDP SHARDED_STATE_DICT checkpoint (each rank saves its own shard).
    3. Reload via export logic (each rank loads its shard → FULL_STATE_DICT on rank 0).
    4. Verify: exported weights == trained weights (not original weights).
"""

import argparse
import os
import shutil
import tempfile
import warnings

import torch
import torch.distributed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
from torch.distributed.fsdp import FullStateDictConfig, ShardedStateDictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config


# ── Shared helpers ────────────────────────────────────────────────────

def make_tiny_model():
    """2-layer GPT2 model, ~648 KB — fast to init and save."""
    cfg = GPT2Config(vocab_size=512, n_embd=64, n_layer=2, n_head=4, n_inner=128)
    return AutoModelForCausalLM.from_config(cfg)


def find_local_tokenizer():
    candidates = [
        '/home/zjx/self_llm/self-researcher/models/Qwen2.5-0.5B-Instruct',
        '/home/zjx/self_llm/self-researcher/models/Qwen2.5-3B-Instruct',
        os.path.expanduser('~/models/Qwen2.5-0.5B-Instruct'),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    raise RuntimeError('No local tokenizer found. Checked:\n' + '\n'.join(candidates))


def run_export_inline(model_fsdp, ckpt_dir, output_dir, tokenizer_path, rank, world_size):
    """Core export logic, mirroring export_fsdp_to_hf.py main()."""
    state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)

    # Each rank loads its own shard (mirrors FSDPCheckpointManager.load_checkpoint)
    ckpt_path = os.path.join(ckpt_dir, f'model_world_size_{world_size}_rank_{rank}.pt')
    model_state_dict = torch.load(ckpt_path, map_location='cpu')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with FSDP.state_dict_type(model_fsdp, StateDictType.SHARDED_STATE_DICT, state_dict_cfg):
            model_fsdp.load_state_dict(model_state_dict)
    del model_state_dict

    # Collect full state dict on rank 0
    full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with FSDP.state_dict_type(model_fsdp, StateDictType.FULL_STATE_DICT, full_cfg):
            full_sd = model_fsdp.state_dict()

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        # Build a clean (non-FSDP) model from config to avoid flat_param shape issues
        # when loading into _fsdp_wrapped_module on multi-GPU (world_size > 1).
        clean_model = AutoModelForCausalLM.from_config(model_fsdp._fsdp_wrapped_module.config)
        clean_model.load_state_dict(full_sd)
        clean_model.save_pretrained(output_dir, safe_serialization=True)
        tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        tok.save_pretrained(output_dir)

    torch.distributed.barrier()


def verify_output(output_dir, trained_sd):
    """Run assertions on the exported directory. Called on rank 0 only."""
    safetensors_path = os.path.join(output_dir, 'model.safetensors')
    assert os.path.exists(safetensors_path), f'FAIL: model.safetensors not found'
    print(f'  [PASS] model.safetensors exists ({os.path.getsize(safetensors_path) // 1024} KB)')

    assert os.path.exists(os.path.join(output_dir, 'config.json')), 'FAIL: config.json not found'
    print('  [PASS] config.json exists')

    loaded = AutoModelForCausalLM.from_pretrained(output_dir)
    exported_sd = loaded.state_dict()

    max_diff_trained = max(
        (exported_sd[k].cpu() - trained_sd[k]).abs().max().item() for k in trained_sd
    )
    assert max_diff_trained < 1e-5, f'FAIL: max diff vs trained = {max_diff_trained}'
    print(f'  [PASS] Max weight diff vs trained : {max_diff_trained:.2e}  (should be ~0)')
    print(f'  [PASS] from_pretrained reload successful')


# ── Single GPU test ───────────────────────────────────────────────────

def run_single_gpu_test(tokenizer_path):
    if not torch.distributed.is_initialized():
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29502')
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        torch.distributed.init_process_group(backend=backend, world_size=1, rank=0)

    rank, world_size = 0, 1
    device_id = 0 if torch.cuda.is_available() else None

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = os.path.join(tmpdir, 'actor')
        output_dir = os.path.join(tmpdir, 'exported_hf')
        base_model_dir = os.path.join(tmpdir, 'base_model')

        # Create and save base model
        torch.manual_seed(42)
        model = make_tiny_model()
        os.makedirs(base_model_dir, exist_ok=True)
        model.save_pretrained(base_model_dir)
        tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        tok.save_pretrained(base_model_dir)

        # Simulate training
        torch.manual_seed(123)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        trained_sd = {k: v.clone().cpu() for k, v in model.state_dict().items()}

        # FSDP wrap + save SHARDED checkpoint
        model_fsdp = FSDP(model, device_id=device_id, use_orig_params=True, sync_module_states=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with FSDP.state_dict_type(model_fsdp, StateDictType.SHARDED_STATE_DICT, state_dict_cfg):
                sd = model_fsdp.state_dict()
        torch.save(sd, os.path.join(ckpt_dir, f'model_world_size_{world_size}_rank_{rank}.pt'))

        hf_dir = os.path.join(ckpt_dir, 'huggingface')
        os.makedirs(hf_dir, exist_ok=True)
        model_fsdp._fsdp_wrapped_module.config.save_pretrained(hf_dir)

        # Export: build fresh FSDP model, load shard, dump to HF
        torch.manual_seed(99)
        export_model = make_tiny_model()
        export_fsdp = FSDP(export_model, device_id=device_id, use_orig_params=True, sync_module_states=True)
        run_export_inline(export_fsdp, ckpt_dir, output_dir, tokenizer_path, rank, world_size)

        verify_output(output_dir, trained_sd)


# ── Multi GPU test ────────────────────────────────────────────────────

def run_multi_gpu_test(tokenizer_path, dist_backend='gloo'):
    """Must be launched via: torchrun --nproc_per_node=N test_export_fsdp_to_hf.py --mode multi_gpu

    Uses gloo backend by default for portability (NCCL can hang on some servers
    when testing on loopback; correctness does not depend on backend choice).
    """
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=dist_backend)

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', rank))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device_id = local_rank
    else:
        device_id = None

    # rank 0 creates tmpdir and broadcasts the path to all ranks
    if rank == 0:
        tmpdir = tempfile.mkdtemp(prefix='test_export_multi_')
    else:
        tmpdir = None
    tmpdir_list = [tmpdir]
    torch.distributed.broadcast_object_list(tmpdir_list, src=0)
    tmpdir = tmpdir_list[0]

    ckpt_dir = os.path.join(tmpdir, 'actor')
    output_dir = os.path.join(tmpdir, 'exported_hf')

    try:
        # All ranks create the same model (fixed seed)
        torch.manual_seed(42)
        model = make_tiny_model()

        # Record trained weights BEFORE FSDP (all ranks have full model here)
        torch.manual_seed(123)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        trained_sd = {k: v.clone().cpu() for k, v in model.state_dict().items()}

        # FSDP wrap + save each rank's shard
        model_fsdp = FSDP(model, device_id=device_id, use_orig_params=True, sync_module_states=True)

        if rank == 0:
            os.makedirs(ckpt_dir, exist_ok=True)
        torch.distributed.barrier()

        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with FSDP.state_dict_type(model_fsdp, StateDictType.SHARDED_STATE_DICT, state_dict_cfg):
                sd = model_fsdp.state_dict()
        torch.save(sd, os.path.join(ckpt_dir, f'model_world_size_{world_size}_rank_{rank}.pt'))

        if rank == 0:
            hf_dir = os.path.join(ckpt_dir, 'huggingface')
            os.makedirs(hf_dir, exist_ok=True)
            model_fsdp._fsdp_wrapped_module.config.save_pretrained(hf_dir)

        torch.distributed.barrier()

        # Export: fresh FSDP model, each rank loads its shard
        torch.manual_seed(0)  # different seed, weights will be overwritten by load
        export_model = make_tiny_model()
        export_fsdp = FSDP(export_model, device_id=device_id, use_orig_params=True, sync_module_states=True)
        run_export_inline(export_fsdp, ckpt_dir, output_dir, tokenizer_path, rank, world_size)

        if rank == 0:
            verify_output(output_dir, trained_sd)

    finally:
        torch.distributed.barrier()
        if rank == 0 and os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)

    if rank == 0:
        torch.distributed.destroy_process_group()


# ── Entry point ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['single_gpu', 'multi_gpu'], default='single_gpu')
    parser.add_argument('--dist_backend', default='gloo', choices=['gloo', 'nccl'],
                        help='Distributed backend for multi_gpu mode (default: gloo)')
    args = parser.parse_args()

    tokenizer_path = find_local_tokenizer()

    if args.mode == 'single_gpu':
        print('=' * 55)
        print('Smoke test: single GPU')
        print('=' * 55)
        print(f'Using local tokenizer: {tokenizer_path}')
        run_single_gpu_test(tokenizer_path)
        print()
        print('All single-GPU checks passed.')

    else:
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        if rank == 0:
            print('=' * 55)
            print(f'Smoke test: multi GPU  (world_size={world_size}, backend={args.dist_backend})')
            print('=' * 55)
            print(f'Using local tokenizer: {tokenizer_path}')
        run_multi_gpu_test(tokenizer_path, dist_backend=args.dist_backend)
        if int(os.environ.get('RANK', 0)) == 0:
            print()
            print('All multi-GPU checks passed.')


if __name__ == '__main__':
    main()
