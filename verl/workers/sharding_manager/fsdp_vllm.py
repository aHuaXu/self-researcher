# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import re
import logging
import torch
import numpy as np
from collections import OrderedDict
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy, ShardedStateDictConfig, StateDictType, FullStateDictConfig
from torch.distributed.device_mesh import DeviceMesh

from verl.third_party.vllm import LLM
from verl.third_party.vllm import parallel_state as vllm_ps
from verl import DataProto
from verl.utils.torch_functional import (broadcast_dict_tensor, allgather_dict_tensors)
from verl.utils.debug import log_gpu_memory_usage
from verl.third_party.vllm import vllm_version

from .base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


class FSDPVLLMShardingManager(BaseShardingManager):

    def __init__(self,
                 module: FSDP,
                 inference_engine: LLM,
                 model_config,
                 full_params: bool = False,
                 device_mesh: DeviceMesh = None,
                 lora_config: dict = None):
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.device_mesh = device_mesh
        self._lora_config = lora_config
        self._cpu_weight_backup = None
        try:
            self._use_sleep_mode = inference_engine.llm_engine.vllm_config.model_config.enable_sleep_mode
        except AttributeError:
            self._use_sleep_mode = False

        # Full params
        self.full_params = full_params
        # For loading vLLM with dtensor, we need full state dict (not sharded)
        # Force full_params=True for lora case since we need LoRA keys
        _force_full_params = full_params or (lora_config is not None)
        if _force_full_params:
            FSDP.set_state_dict_type(self.module,
                                     state_dict_type=StateDictType.FULL_STATE_DICT,
                                     state_dict_config=FullStateDictConfig(
                                         offload_to_cpu=True, rank0_only=True))
        else:
            FSDP.set_state_dict_type(self.module,
                                     state_dict_type=StateDictType.SHARDED_STATE_DICT,
                                     state_dict_config=ShardedStateDictConfig())

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh['dp'].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    def _log_physical_mem(self, label: str):
        """Log actual GPU physical memory (nvidia-smi level), not just PyTorch tracked."""
        free, total = torch.cuda.mem_get_info()
        used = total - free
        pt_alloc = torch.cuda.memory_allocated()
        pt_reserved = torch.cuda.memory_reserved()
        logger.warning(
            f"[GPU MEM] {label}: physical_used={used/1e9:.2f}GiB, "
            f"physical_free={free/1e9:.2f}GiB, "
            f"pt_alloc={pt_alloc/1e9:.2f}GiB, pt_reserved={pt_reserved/1e9:.2f}GiB")

    def __enter__(self):
        self._log_physical_mem('Before __enter__ (before wake_up)')
        log_gpu_memory_usage('Before state_dict() in sharding manager memory', logger=logger)
        params = self.module.state_dict()
        log_gpu_memory_usage('After state_dict() in sharding manager memory', logger=logger)

        if self._lora_config is not None:
            from verl.utils.fsdp_utils import offload_fsdp_model_to_cpu
            offload_fsdp_model_to_cpu(self.module)
            log_gpu_memory_usage('After FSDP offload before vLLM sync', logger=logger)
            self._save_lora_adapters(params)
            params = self._filter_lora_keys(params)
            log_gpu_memory_usage('After LoRA filter/save in sharding manager', logger=logger)

        load_format = 'hf' if self.full_params else 'dtensor'
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            self.inference_engine.init_cache_engine()
            self.inference_engine.sync_model_weights(params, load_format=load_format)
        else:
            if self._use_sleep_mode:
                self.inference_engine.wake_up()
                self._log_physical_mem('After wake_up()')
            else:
                self._restore_kv_cache()
                self._log_physical_mem('After _restore_kv_cache()')
            if load_format == 'dtensor':
                from verl.third_party.vllm import load_dtensor_weights
                vllm_model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
                for param in vllm_model.parameters():
                    if param.device.type == 'cpu':
                        param.data = param.data.cuda()
                load_dtensor_weights(params, vllm_model)
            else:
                raise NotImplementedError(f'load_format {load_format} not implemented')
        self._log_physical_mem('After sync model weights')
        log_gpu_memory_usage('After sync model weights in sharding manager', logger=logger)

        del params
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After del state_dict and empty_cache in sharding manager', logger=logger)

        # TODO: offload FSDP model weights
        # self.module.cpu()
        # torch.cuda.empty_cache()
        # if torch.distributed.get_rank() == 0:
        # print(f'after model to cpu in sharding manager memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB')

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    def _get_vllm_model(self):
        return self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model


    def _get_vllm_worker(self):
        return self.inference_engine.llm_engine.model_executor.driver_worker.worker

    def _offload_vllm_weights_to_cpu(self):
        """Manually offload vLLM model weights to CPU (like old vLLM 0.6.3)."""
        model = self._get_vllm_model()
        if self._cpu_weight_backup is None:
            self._cpu_weight_backup = {}
            for name, param in model.named_parameters():
                self._cpu_weight_backup[name] = param.data.detach().cpu().clone()
                param.data = self._cpu_weight_backup[name]
        else:
            for name, param in model.named_parameters():
                param.data = self._cpu_weight_backup[name]

    def _free_kv_cache(self):
        """Free the KV cache GPU memory when sleep mode is disabled.
        
        Must also clear kv_cache references stored in static_forward_context
        by bind_kv_cache(), otherwise tensors remain referenced and GC can't free them.
        """
        worker = self._get_vllm_worker()
        # Clear kv_cache references stored in each attention layer's forward context
        model_runner = worker.model_runner
        if hasattr(model_runner, 'vllm_config') and model_runner.vllm_config is not None:
            ctx = model_runner.vllm_config.compilation_config.static_forward_context
            for layer_name in list(ctx.keys()):
                forward_ctx = ctx[layer_name]
                if hasattr(forward_ctx, 'kv_cache'):
                    for i in range(len(forward_ctx.kv_cache)):
                        forward_ctx.kv_cache[i] = None
        if hasattr(worker, 'gpu_cache') and worker.gpu_cache is not None:
            del worker.gpu_cache
            worker.gpu_cache = None
        if hasattr(worker, 'cache_engine') and worker.cache_engine is not None:
            del worker.cache_engine
            worker.cache_engine = None

    def _restore_kv_cache(self):
        """Re-initialize KV cache after it was freed, and re-bind to attention layers."""
        worker = self._get_vllm_worker()
        if worker.gpu_cache is None:
            worker._init_cache_engine()
            # Re-bind kv_cache to attention layers (reverses what _free_kv_cache cleared)
            from vllm.utils import bind_kv_cache
            model_runner = worker.model_runner
            bind_kv_cache(
                model_runner.vllm_config.compilation_config.static_forward_context,
                worker.gpu_cache)

    def __exit__(self, exc_type, exc_value, traceback):
        self._log_physical_mem('Before __exit__ (before sleep/offload)')
        log_gpu_memory_usage('Before vllm offload in sharding manager', logger=logger)
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            self.inference_engine.offload_model_weights()
            self.inference_engine.free_cache_engine()
        elif self._use_sleep_mode:
            self.inference_engine.sleep(level=1)
        else:
            self._offload_vllm_weights_to_cpu()
            self._free_kv_cache()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        self._log_physical_mem('After __exit__ (after sleep/offload + empty_cache)')
        log_gpu_memory_usage('After vllm offload in sharding manager', logger=logger)

        self.module.train()
        torch.cuda.empty_cache()

        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def _remap_qwen2_keys_for_vllm(self, params: dict) -> dict:
        """Remap Qwen2 separate Q/K/V projection keys to vLLM's fused qkv_proj format.

        FSDP state_dict has:    model.layers.0.self_attn.q_proj.weight
                                model.layers.0.self_attn.k_proj.weight
                                model.layers.0.self_attn.v_proj.weight
        vLLM's qwen2 expects:  model.layers.0.self_attn.qkv_proj.weight  (fused, loaded via shard_id)

        Uses exact substring matching to prevent qkv_proj.weight from matching k_proj.
        """
        remapped = OrderedDict()
        for key, value in params.items():
            # Skip if already contains qkv_proj (avoid double replacement)
            if 'qkv_proj' in key:
                remapped[key] = value
                continue
            # Use exact substring match to avoid qkv_proj matching k_proj pattern
            # qkv_proj.weight contains k_proj.weight as suffix but NOT .self_attn.k_proj.
            if '.self_attn.q_proj.' in key:
                new_key = key.replace('.self_attn.q_proj.', '.self_attn.qkv_proj.')
            elif '.self_attn.k_proj.' in key:
                new_key = key.replace('.self_attn.k_proj.', '.self_attn.qkv_proj.')
            elif '.self_attn.v_proj.' in key:
                new_key = key.replace('.self_attn.v_proj.', '.self_attn.qkv_proj.')
            else:
                new_key = key
            remapped[new_key] = value
        return remapped

    def _filter_lora_keys(self, params: dict) -> dict:
        """Remove LoRA keys and remap PEFT base_layer names to plain weight names.

        PEFT wraps target modules so state_dict keys look like:
          base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight
          base_model.model.model.layers.0.self_attn.q_proj.lora_A.planner.weight
        We need to:
          1. Drop all keys containing lora_A / lora_B
          2. Rename *.base_layer.weight -> *.weight
          3. Strip the "base_model.model." prefix added by PEFT
        """
        filtered = OrderedDict()
        for key, value in params.items():
            if 'lora_A' in key or 'lora_B' in key:
                continue
            new_key = key.replace('.base_layer.weight', '.weight')
            new_key = new_key.replace('.base_layer.bias', '.bias')
            if new_key.startswith('base_model.model.'):
                new_key = new_key[len('base_model.model.'):]
            filtered[new_key] = value
        return filtered

    def _save_lora_adapters(self, params: dict):
        """Extract per-adapter LoRA weights from the FSDP state_dict and save
        them as PEFT-compatible adapter directories that vLLM can load.

        Only rank 0 writes to disk; all ranks hit a barrier afterwards.
        """
        cfg = self._lora_config
        save_dir = cfg['save_dir']
        adapter_names = cfg['adapter_names']
        rank = cfg['rank']
        alpha = cfg['alpha']
        target_modules = cfg.get('target_modules', ['q_proj', 'k_proj', 'v_proj', 'o_proj'])

        is_rank0 = (torch.distributed.get_rank() == 0) if torch.distributed.is_initialized() else True

        for adapter_name in adapter_names:
            adapter_dir = os.path.join(save_dir, adapter_name)
            if is_rank0:
                os.makedirs(adapter_dir, exist_ok=True)

                adapter_weights = OrderedDict()
                for key, value in params.items():
                    # PEFT key format: _fsdp_wrapped_module.base_model.model.model.layers.x.self_attn.q_proj.lora_A.planner.weight
                    # FSDP state_dict after use_orig_params=True: base_model.model.model.layers.x...lora_A.planner.weight
                    # vLLM's parse_fine_tuned_lora_name requires lora_A/lora_B to be the second-to-last path component.
                    # The adapter name (planner/executor) is passed via LoRARequest.lora_adapter_name, not in the key.
                    # So we strip the adapter name suffix: .lora_A.planner.weight -> .lora_A.weight
                    if f'.lora_A.{adapter_name}.' in key or f'.lora_B.{adapter_name}.' in key:
                        new_key = key.replace(
                            f'.lora_A.{adapter_name}.weight', '.lora_A.weight'
                        ).replace(
                            f'.lora_B.{adapter_name}.weight', '.lora_B.weight'
                        )
                        # Strip _fsdp_wrapped_module. prefix if present
                        if '_fsdp_wrapped_module.' in new_key:
                            new_key = new_key.replace('_fsdp_wrapped_module.', '')
                        # vLLM expects: base_model.model.layers.x.self_attn.q_proj.lora_A.weight
                        # So we strip one "model." from the prefix
                        if new_key.startswith('base_model.model.model.'):
                            new_key = 'base_model.model.' + new_key[len('base_model.model.model.'):]
                        if hasattr(value, 'full_tensor'):
                            full_value = value.full_tensor()
                        else:
                            full_value = value
                        adapter_weights[new_key] = full_value.contiguous().cpu()

                if adapter_weights:
                    from safetensors.torch import save_file
                    save_file(adapter_weights, os.path.join(adapter_dir, 'adapter_model.safetensors'))

                adapter_config = {
                    'r': rank,
                    'lora_alpha': alpha,
                    'target_modules': list(target_modules),
                    'bias': 'none',
                    'task_type': 'CAUSAL_LM',
                    'peft_type': 'LORA',
                    'base_model_name_or_path': cfg.get('base_model', ''),
                }
                with open(os.path.join(adapter_dir, 'adapter_config.json'), 'w') as f:
                    json.dump(adapter_config, f, indent=2)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def preprocess_data(self, data: DataProto) -> DataProto:
        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3'):
            group = vllm_ps.get_tensor_model_parallel_group()
        else:
            group = vllm_ps.get_tensor_model_parallel_group().device_group

        prev_device = data.batch.device
        data.batch = data.batch.cuda(device=torch.cuda.current_device())
        data.batch = allgather_dict_tensors(data.batch.contiguous(), size=tp_size, group=group, dim=0)
        data.batch = data.batch.to(prev_device)
        # all gather non_tensor_batch
        all_non_tensor_batch = [None for _ in range(tp_size)]
        torch.distributed.all_gather_object(all_non_tensor_batch, data.non_tensor_batch, group=group)
        data.non_tensor_batch = {k: np.concatenate([d[k] for d in all_non_tensor_batch]) for k in data.non_tensor_batch}
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        local_world_size = vllm_ps.get_tensor_model_parallel_world_size()
        src_rank = (torch.distributed.get_rank() // local_world_size) * local_world_size
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3'):
            broadcast_dict_tensor(data.batch, src=src_rank, group=vllm_ps.get_tensor_model_parallel_group())
        else:
            broadcast_dict_tensor(data.batch,
                                  src=src_rank,
                                  group=vllm_ps.get_tensor_model_parallel_group().device_group)
        dp_rank = torch.distributed.get_rank()
        dp_size = torch.distributed.get_world_size()  # not consider torch micro-dp
        tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        if tp_size > 1:
            # TODO: shall we build a micro_dp group for vllm when integrating with vLLM?
            local_prompts = data.chunk(chunks=tp_size)
            data = local_prompts[dp_rank % tp_size]
        return data
