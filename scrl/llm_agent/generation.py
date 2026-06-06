# =============================================================================
# Based on the Search-R1 example from the Search-R1 project.
#
# Original Authors: Jin Bowen, Zeng Hansi, Yue Zhenrui, Wang Dong, Zamani Hamed, Han Jiawei
#
# License: Apache 2.0
# Project URL: https://github.com/PeterGriffinJin/Search-R1
# =============================================================================

import torch
import re
from collections import defaultdict
import concurrent.futures
import os
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from scrl.llm_agent.tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import json
import numpy as np
import time
from time import strftime, gmtime

@dataclass
class GenerationConfig:
    max_turns: int
    num_gpus: int
    max_seq_len_for_training: int = 7000
    model_name: str = None
    n: int = 1,
    project_name: str = None,
    experiment_name: str = None,
    search_engine: str = "rag",
    nnodes: int = 1


# Force-answer prefill: appended after the generation prompt on the FINAL turn so the model is
# constrained to emit an answer (it continues from "<answer>"). Guarantees every trajectory ends
# with a parseable <answer>…</answer> (live F1 outcome + clean finding), instead of a dangling
# tool_call / empty turn. Standard multi-turn search-RL practice.
FORCE_ANSWER_PREFILL = "<answer>"


TOOLS_FOR_WIKI = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for relevant information. You should use this tool if the historical page content is not enough to answer the question. Or last search result is not relevant to the question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "The query to search, which helps answer the question"
                        },
                        "description": "The queries to search"
                    }
                },
                "required": ["query"],
                "minItems": 1,
                "uniqueItems": True
            }
        }
    }
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for relevant information from google. You should use this tool if the historical page content is not enough to answer the question. Or last search result is not relevant to the question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",  
                        "items": {    
                            "type": "string",
                            "description": "The query to search, which helps answer the question"
                        },
                        "description": "The queries to search"
                    }
                },
                "required": ["query"],
                "minItems": 1,
                "uniqueItems": True
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browse_webpage",
            "description": "Browse the webpage and return the content that not appeared in the conversation history. You should use this tool if the last action is search and the search result maybe relevant to the question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url_list": {"type": "array",
                            "items": {
                                "type": "string",
                                "description": "The chosen url from the search result, do not use url that not appeared in the search result"
                            },
                            "description": "The chosen urls from the search result."
                        },
                },
                "required": ["url_list"]
            }
        }
    }
]


class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id
        ))
        
        if self.config.search_engine == "rag":
            self.tools = TOOLS_FOR_WIKI
            self.system_prompt =  f"""## Background information 
* Today is {strftime("%Y-%m-%d", gmtime())}
* You are Deep AI Research Assistant

The question I give you is a complex question that requires a *deep research* to answer.

I will provide you with one tool to help you answer the question:
* A web search tool to help you perform search.

You don't have to answer the question now, but you should first analyze the question and think about what to search next.

Your output format should be one of the following two formats:

Format 1 - When you have enough information to answer:
[Your analysis and reasoning here]
<answer>
YOUR FINAL ANSWER
</answer>

Format 2 - When you need to search for more information:
[Your analysis and reasoning here]
<tool_call>
YOUR TOOL CALL WITH CORRECT FORMAT
</tool_call>

You should always follow the above two formats strictly.
Only output the final answer (in words, numbers or phrase) inside the <answer></answer> tag, without any explanations or extra information. If this is a yes-or-no question, you should only answer yes or no.
"""
        elif self.config.search_engine == "online_search":
            self.tools = TOOLS
            self.system_prompt = f"""## Background information 
* Today is {strftime("%Y-%m-%d", gmtime())}
* You are Deep AI Research Assistant

The question I give you is a complex question that requires a *deep research* to answer.

I will provide you with two tools to help you answer the question:
* A web search tool to help you perform google search. 
* A webpage browsing tool to help you get new page content.

You don't have to answer the question now, but you should first analyze the question and think about what to search next.

Your output format should be one of the following two formats:

Format 1 - When you have enough information to answer:
[Your analysis and reasoning here]
<answer>
YOUR FINAL ANSWER
</answer>

Format 2 - When you need to search for more information:
[Your analysis and reasoning here]
<tool_call>
YOUR TOOL CALL WITH CORRECT FORMAT
</tool_call>

You should always follow the above two formats strictly.
Only output the final answer (in words, numbers or phrase) inside the <answer></answer> tag, without any explanations or extra information. If this is a yes-or-no question, you should only answer yes or no.
"""
        else:
            assert False


    def _update_right_side(self, original_right_side: Dict, 
                           cur_responses: torch.Tensor,
                           next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side of rollings."""
        if next_obs_ids is not None:
            responses = self.tensor_fn.concatenate_with_padding(
                [original_right_side['responses'], cur_responses, next_obs_ids],
                pad_to_left=False
            )
        else:
            responses = self.tensor_fn.concatenate_with_padding(
                [original_right_side['responses'], cur_responses],
                pad_to_left=False
            )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        
        return {'responses': responses[:, :effective_len]}

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, next_obs_ids: torch.Tensor) -> DataProto:
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        return DataProto.from_dict({
                'input_ids': new_input_ids[:, -effective_len:],
                'position_ids': new_position_ids[:, -effective_len:],
                'attention_mask': new_attention_mask[:, -effective_len:]
            })

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']
        return next_obs_ids
        
    def execute_predictions(self, 
        tool_call_list: List[Tuple[int, str, str, str]], total_number: int = 4096
    ) :
        """直接调用工具（无需文件 IPC）"""
        from research_agent.tools import web_search, browse_webpage
        from research_agent.tools._state import get_tool_state
        from research_agent.tools.context import tool_rollout_message_idx, tool_rollout_user_query

        tool_state = get_tool_state()
        tool_state.ensure_initialized()

        def run_at_index(i: int) -> Dict[str, Any]:
            idx, question, think, tool = tool_call_list[i]
            var_idx = tool_rollout_message_idx.set(idx)
            var_uq = tool_rollout_user_query.set(question)
            try:
                ts = get_tool_state()
                ts.ensure_initialized()
                tool_name = tool.get("name", "")
                args = tool.get("arguments", {})
                if tool_name == "web_search":
                    content = web_search.invoke(args)
                elif tool_name == "browse_webpage":
                    content = browse_webpage.invoke(args)
                else:
                    content = f'{{"error": "Unknown tool: {tool_name}"}}'
                return {
                    "idx": idx,
                    "question": question,
                    "think": think,
                    "tool_call": tool,
                    "content": content,
                }
            finally:
                tool_rollout_user_query.reset(var_uq)
                tool_rollout_message_idx.reset(var_idx)

        n = len(tool_call_list)
        results: List[Optional[Dict[str, Any]]] = [None] * n

        web_positions = [
            i for i in range(n) if tool_call_list[i][3].get("name") == "web_search"
        ]
        browse_positions = [
            i for i in range(n) if tool_call_list[i][3].get("name") == "browse_webpage"
        ]
        other_positions = [
            i for i in range(n) if i not in web_positions and i not in browse_positions
        ]

        search_workers = max(1, int(os.getenv("TOOL_WEB_SEARCH_MAX_WORKERS", "5")))
        browse_workers = max(1, int(os.getenv("TOOL_BROWSE_MAX_WORKERS", "4")))
        timeout = int(os.getenv("TOOL_CALL_TIMEOUT", "120"))

        def _run_parallel(positions: List[int], max_w: int):
            if not positions:
                return
            if len(positions) == 1:
                results[positions[0]] = run_at_index(positions[0])
                return
            w = min(max_w, len(positions))
            with concurrent.futures.ThreadPoolExecutor(max_workers=w) as pool:
                futures = {pool.submit(run_at_index, i): i for i in positions}
                try:
                    for fut in concurrent.futures.as_completed(futures, timeout=timeout):
                        i = futures[fut]
                        try:
                            results[i] = fut.result()
                        except Exception as e:
                            print(f"[execute_predictions] tool error at index {i}: {e}")
                            idx, question, think, tool = tool_call_list[i]
                            results[i] = {
                                "idx": idx, "question": question, "think": think,
                                "tool_call": tool, "content": json.dumps({"error": str(e)}),
                            }
                except concurrent.futures.TimeoutError:
                    print(f"[execute_predictions] timeout after {timeout}s, filling remaining with errors")
                    for fut, i in futures.items():
                        if results[i] is None:
                            idx, question, think, tool = tool_call_list[i]
                            results[i] = {
                                "idx": idx, "question": question, "think": think,
                                "tool_call": tool, "content": json.dumps({"error": "timeout"}),
                            }
                            fut.cancel()

        _run_parallel(web_positions, search_workers)
        _run_parallel(browse_positions + other_positions, browse_workers)

        return results

    def _generate_with_gpu_padding(self, active_batch: DataProto, lora_adapter_name: str = None) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        _ADAPTER_BASE_ID = {"planner": 1, "executor": 2}

        if lora_adapter_name is not None:
            lora_save_dir = getattr(self, 'lora_save_dir', './tmp_lora_adapters')
            lora_step = getattr(self, '_lora_step', 0)
            active_batch.meta_info['lora_adapter_path'] = os.path.join(lora_save_dir, lora_adapter_name)
            active_batch.meta_info['lora_adapter_name'] = lora_adapter_name
            active_batch.meta_info['lora_adapter_id'] = lora_step * 10 + _ADAPTER_BASE_ID.get(lora_adapter_name, 1)

        num_gpus = self.config.num_gpus * self.config.nnodes
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)

        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus

        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}

        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch, meta_info=active_batch.meta_info)

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}

        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta

        padded_output.batch = trimmed_batch
        return padded_output

    def parse_question(self, input_ids: torch.Tensor) -> str:
        """Parse question to get the query content."""
        query_contents = self.tokenizer.batch_decode(input_ids)
        query_contents = [re.sub(r'^(<\|endoftext\|>)+', '', content) for content in query_contents]
        query_contents = [content.split("<|im_start|>user\n")[1].split("<|im_end|>")[0] for content in query_contents]
        return query_contents

    def parse_response(self, input_ids: torch.Tensor, think: bool = False) -> List[Tuple[bool, str, str]]:
        """Parse response to get the reasoning and answer or tool call.
            return: [(is_stop, reasoning, answer/tool_call), ...]
        """
        response_contents = self.tokenizer.batch_decode(input_ids)
        results = []
        for content in response_contents:
            if "<answer>" in content and "</answer>" in content:
                reasoning = content.split("<answer>")[0].strip()
                answer = content.split("<answer>")[1].split("</answer>")[0].strip()
                results.append((True, reasoning, answer))
            elif "<tool_call>" in content and "</tool_call>" in content:
                reasoning = content.split("<tool_call>")[0].strip()
                tool_call_str = content.split("<tool_call>")[1].split("</tool_call>")[0].strip()
                try:
                    tool_call = json.loads(tool_call_str)
                    assert "name" in tool_call, "no valid function name in tool_call"
                    assert "arguments" in tool_call, "no valid arguments in tool_call"
                    assert tool_call["name"] in ["web_search", "browse_webpage"], "invalid tool name"
                    if tool_call["name"] == "web_search":
                        assert "query" in tool_call["arguments"], "no valid query in tool_call"
                        assert isinstance(tool_call["arguments"]["query"], list), "query should be a list"
                    elif tool_call["name"] == "browse_webpage":
                        assert "url_list" in tool_call["arguments"], "no valid url_list in tool_call"
                        assert isinstance(tool_call["arguments"]["url_list"], list), "url_list should be a list"
                        assert len(tool_call["arguments"]["url_list"]) >= 1, "url_list number must be greater than 0"
                    results.append((False, reasoning, tool_call))
                except Exception as e:
                    print(f"model tool call format error: {e}")
                    results.append((True, "", ""))
            else:
                results.append((True, "", ""))
        return results

    def run_llm_loop(self, gen_batch: DataProto, global_steps: int, lora_adapter_name: str = None) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        node_rank = int(os.environ.get("PET_NODE_RANK", 0))
        print(f"node {node_rank} gains {len(gen_batch.batch['input_ids'])} datas!",flush=True)
        query_contents = self.parse_question(gen_batch.batch['input_ids'])
        messages_list = []
        agent_grpo_idx = []
        for idx, query_content in enumerate(query_contents):
            for _ in range(self.config.n):
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": query_content}
                ]
                messages_list.append(messages)
                agent_grpo_idx.append(idx)
        activate_list = [i for i in range(len(messages_list))]
        message_string_list = ["" for _ in range(len(messages_list))]
        
        # 确保保存目录存在
        output_dir = f"./outputs/{self.config.project_name}/{self.config.experiment_name}/rollout"
        if not os.path.exists(output_dir):
            print(f"Directory not exist, create at {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
        for step in range(self.config.max_turns):
            print(f"node {node_rank} step {step} start!")
            activate_messages_list = [messages_list[i] for i in activate_list]
            if activate_list == []:
                break

            rollings_active = self.tokenizer.apply_chat_template(activate_messages_list, add_generation_prompt=True, tools=self.tools, tokenize=False)
            _is_last_turn = (step == self.config.max_turns - 1)
            if _is_last_turn:
                # FORCE ANSWER on the final turn: prefill "<answer>" so still-active samples emit a
                # parseable answer instead of another tool_call (the prefill rides in the prompt, so
                # the reconstructed string below contains the full <answer>…</answer>).
                rollings_active = [r + FORCE_ANSWER_PREFILL for r in rollings_active]
            rollings_active = self.tokenizer(rollings_active, return_tensors="pt",padding=True)

            pad_mask = rollings_active['input_ids'] != self.tokenizer.pad_token_id
            sorted_indices = pad_mask.to(torch.int64).argsort(dim=1, stable=True)
            rollings_active['input_ids'] = rollings_active['input_ids'].gather(1, sorted_indices)
            rollings_active['attention_mask'] = rollings_active['attention_mask'].gather(1, sorted_indices)
            
            attention_mask = rollings_active['attention_mask']
            rollings_active['position_ids'] = self.tensor_fn.create_position_ids(attention_mask)
            
            with open(f"./outputs/{self.config.project_name}/{self.config.experiment_name}/rollout/rollout_step_{global_steps}_round_{step}.json", "w", encoding='utf-8') as f:
                step_write_list = []
                for i, input_ids in enumerate(rollings_active['input_ids']):
                    step_write_list.append({
                        "idx": activate_list[i],
                        "question": query_contents[agent_grpo_idx[activate_list[i]]],
                        "input_ids_no_pad": self.tokenizer.decode(input_ids, skip_special_tokens=False).replace("<|endoftext|>", ""),
                    })
                json.dump(step_write_list, f, indent=4, ensure_ascii=False)
            print(f"rollout_step_{global_steps}_round_{step}.json 写入完成")
            
            print(f"node {node_rank}, turn {step} rollings_active is {len(rollings_active['input_ids'])} datas")
            rollings_active = DataProto.from_dict({
                'input_ids': rollings_active['input_ids'],
                'attention_mask': rollings_active['attention_mask'],
                'position_ids': rollings_active['position_ids'],
            })
            
            gen_output = self._generate_with_gpu_padding(rollings_active, lora_adapter_name=lora_adapter_name)
            meta_info = gen_output.meta_info
            print(f"node {node_rank}, turn {step} gen_output {len(gen_output.batch['responses'])} datas")

            if _is_last_turn:
                # Prompt was prefilled with "<answer>": the generation IS the forced answer. Take it
                # for ALL still-active samples (prompt already holds the open tag) and finish —
                # skip parse_response/tool execution this turn.
                for i in range(len(activate_list)):
                    message_string_list[activate_list[i]] = (
                        self.tokenizer.decode(rollings_active.batch['input_ids'][i], skip_special_tokens=False).replace("<|endoftext|>", "")
                        + self.tokenizer.decode(gen_output.batch['responses'][i], skip_special_tokens=False).replace("<|endoftext|>", "")
                    )
                print(f"第{step}轮(最后一轮): 强制 {len(activate_list)} 条样本输出 <answer>", flush=True)
                activate_list = []
                break

            results = self.parse_response(gen_output.batch['responses'])
            assert len(results) == len(activate_list) # 每一轮更新后，结果数量和当前活跃的query数量一致
            activate_list_copy = []
            tool_call_list = []
            for i in range(len(results)):
                if results[i][0]:
                    message_string_list[activate_list[i]] = self.tokenizer.decode(rollings_active.batch['input_ids'][i], skip_special_tokens=False).replace("<|endoftext|>", "") + self.tokenizer.decode(gen_output.batch['responses'][i], skip_special_tokens=False).replace("<|endoftext|>", "")
                else:
                    activate_list_copy.append(activate_list[i])
                    tool_call_list.append((activate_list[i], messages_list[activate_list[i]][1]["content"], results[i][1], results[i][2]))
                    
            if step == self.config.max_turns - 1:
                print(f"node {node_rank}, turn {step} tool_call_list {len(tool_call_list)} datas")
                print(f"第{step}轮(最后一轮)结束，跳过工具执行，{len(tool_call_list)}条样本未返回answer")
                activate_list = activate_list_copy
                break

            tool_call_list = self.execute_predictions(tool_call_list,len(messages_list))
            print(f"node {node_rank}, turn {step} tool_call_list {len(tool_call_list)} datas")
            tool_content_max_chars = int(os.getenv("TOOL_CONTENT_MAX_CHARS", "3000"))
            for i in range(len(tool_call_list)):
                messages_list[tool_call_list[i]['idx']].append(
                    {
                        "role": "assistant", 
                        "content": tool_call_list[i]['think'],
                        "tool_calls": [
                                        {
                                            "type": "function", 
                                            "function": tool_call_list[i]['tool_call']
                                        }
                                    ]
                    }
                )
                content = tool_call_list[i]['content']
                if len(content) > tool_content_max_chars:
                    content = content[:tool_content_max_chars] + "\n...[truncated]"
                messages_list[tool_call_list[i]['idx']].append(
                    {
                        "role": "tool", 
                        "name": tool_call_list[i]['tool_call']["name"],
                        "content": content
                    }
                )
            print(f"第{step}轮结束， node {node_rank} 原本有{len(activate_list)}个query，现在有{len(activate_list_copy)}个query")
            activate_list = activate_list_copy
        if activate_list != []:
            for i in activate_list:
                message_string_list[i] = self.tokenizer.apply_chat_template(messages_list[i], add_generation_prompt=True, tools=self.tools, tokenize=False)
        
        response_str_list = []
        initial_prompt_list = []
        for i, messages in enumerate(messages_list):
            initial_prompt = self.tokenizer.apply_chat_template(messages[0:2], add_generation_prompt=True, tools=self.tools, tokenize=False)
            initial_prompt_list.append(initial_prompt)
            response_str_list.append(message_string_list[i][len(initial_prompt):])
        
        prompts_tokenizered = self.tokenizer(initial_prompt_list, return_tensors="pt",padding=True)

        prompts_repeated = prompts_tokenizered['input_ids']
        pad_mask = prompts_repeated != self.tokenizer.pad_token_id
        sorted_indices = pad_mask.to(torch.int64).argsort(dim=1, stable=True)

        prompts_repeated = prompts_repeated.gather(1, sorted_indices)
        prompts_attention_mask = prompts_tokenizered['attention_mask'].gather(1, sorted_indices)

        max_seq_len = int(self.config.max_seq_len_for_training)
        max_response_tokens = max(512, max_seq_len - prompts_repeated.shape[1])

        responses_tokenized = self.tokenizer(response_str_list, return_tensors="pt", padding=True)
        responses = responses_tokenized['input_ids'][:, :max_response_tokens]
        responses_attention_mask = responses_tokenized['attention_mask'][:, :max_response_tokens]
        if responses_tokenized['input_ids'].shape[1] > max_response_tokens:
            print(f"[WARN] response truncated from {responses_tokenized['input_ids'].shape[1]} to {max_response_tokens} tokens (max_seq_len={max_seq_len})")

        attention_mask = torch.cat((prompts_attention_mask, responses_attention_mask), dim=-1)
        position_ids = self.tensor_fn.create_position_ids(attention_mask)
        
        message_tensor = DataProto.from_dict({
            'prompts': prompts_repeated,
            'responses': responses,
            'input_ids': torch.cat((prompts_repeated, responses), dim=-1),
            'attention_mask': attention_mask,
            'position_ids': position_ids,
        })
        message_tensor.meta_info.update(meta_info)
        message_tensor.non_tensor_batch['agent_grpo_idx'] = np.array(agent_grpo_idx, dtype=object)
        print("generation结束")
        
        with open(f"./outputs/{self.config.project_name}/{self.config.experiment_name}/rollout/rollout_step_{global_steps}.json", "w", encoding='utf-8') as f:
            write_list = []
            for i, message_str in enumerate(message_string_list):
                write_list.append({
                    "idx": i,
                    "question": query_contents[agent_grpo_idx[i]],
                    "message_str": message_str
                })
            json.dump(write_list, f, indent=4, ensure_ascii=False)
            print(f"rollout_step_{global_steps}.json 写入完成")
        print(f"node {node_rank} message_string_list {len(message_string_list)}")
              
        return message_string_list, message_tensor
    
