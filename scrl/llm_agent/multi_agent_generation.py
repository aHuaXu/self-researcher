"""
Multi-Agent Generation Manager for dual-agent LoRA rollout.

Implements Planner -> Executor pipeline where each agent uses a separate
LoRA adapter on the same base model.

Design rationale:
  - Stage 1 (single-agent GRPO) teaches web-search tool-calling on easy/medium QA.
  - Stage 2 (this dual-agent GRPO) teaches long search chains for hard QA:
      Planner decomposes questions into sub-tasks with dependency annotations;
      Executor researches each sub-task via multi-turn tool use, receiving
      findings from declared dependencies as context.
  - The last Executor's answer is the final output for F1 reward matching.
"""

import re
import os
import json
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

from verl import DataProto
from verl.utils.torch_functional import pad_sequence_to_length
from scrl.llm_agent.generation import LLMGenerationManager, GenerationConfig

from research_agent.prompts.planner import get_planner_prompt, parse_plan, SubTask
from research_agent.prompts.executor import (
    get_executor_prompt,
    format_prior_findings,
    EXECUTOR_TOOLS,
)


def _pad_dataproto_for_concat(parts: List[DataProto], pad_token_id: int) -> List[DataProto]:
    """Pad batch tensors to a common seq length so DataProto.concat(dim=0) succeeds."""
    if len(parts) <= 1:
        return parts

    batch_parts = [p for p in parts if p.batch is not None]
    if not batch_parts:
        return parts

    max_lens: Dict[str, int] = {}
    for part in batch_parts:
        for key, tensor in part.batch.items():
            if tensor.ndim < 2:
                continue
            max_lens[key] = max(max_lens.get(key, 0), tensor.shape[1])

    padded_parts: List[DataProto] = []
    for part in parts:
        if part.batch is None:
            padded_parts.append(part)
            continue

        padded_tensors = {}
        for key, tensor in part.batch.items():
            if key not in max_lens or tensor.ndim < 2:
                padded_tensors[key] = tensor
                continue
            target_len = max_lens[key]
            if key == "attention_mask" or key == "position_ids":
                pad_val = 0
            else:
                pad_val = pad_token_id
            padded_tensors[key] = pad_sequence_to_length(
                tensor, target_len, pad_val, left_pad=False
            )

        padded_parts.append(
            DataProto.from_dict(
                padded_tensors,
                non_tensors=part.non_tensor_batch,
                meta_info=part.meta_info,
            )
        )

    return padded_parts


@dataclass
class ExecutorDAGResult:
    """Result of DAG-based executor execution."""

    exec_outputs: Any  # DataProto containing all executor generation outputs
    final_answers: List[str]  # Final answer per question (from is_final task)
    all_findings: Dict[int, Dict[int, Dict[str, str]]]  # {q_idx: {task_idx: {"question": ..., "answer": ...}}}
    todo_mapping: List[int]  # Maps executor batch index -> question index
    exec_msg_strings: List[str]  # Raw message strings from all waves


@dataclass
class MultiAgentResult:
    """Result of the full dual-agent (Planner + Executor) rollout."""

    planner_outputs: Any  # DataProto from planner generation
    executor_outputs: Any  # DataProto from executor generation (all waves combined)
    queries: List[str]  # Original research questions
    plan_texts: List[str]  # Raw planner output texts
    parsed_plans: List[List[SubTask]]  # Structured plans per question
    final_answers: List[str]  # Final answer per question
    all_findings: Dict[int, Dict[int, Dict[str, str]]]  # Per-question findings
    todo_mapping: List[int]  # Executor batch index -> question index mapping


def schedule_waves(parsed_plan: List[SubTask]) -> List[List[SubTask]]:
    """Topological sort sub-questions into execution waves.

    Each wave contains tasks whose dependencies are ALL resolved
    by prior waves. Tasks within a wave can run in parallel (batched).

    Args:
        parsed_plan: Output of parse_plan(), list of SubTask instances.

    Returns:
        List of waves, each wave is a list of SubTask instances.
        Empty list if parsed_plan is empty.
    """
    if not parsed_plan:
        return []

    resolved = set()
    waves = []
    remaining = list(parsed_plan)

    while remaining:
        wave = [t for t in remaining if all(d in resolved for d in t.deps)]
        if not wave:
            # Circular dependency fallback: force all remaining into one wave
            wave = remaining[:]
        waves.append(wave)
        resolved.update(t.index for t in wave)
        remaining = [t for t in remaining if t.index not in resolved]

    return waves


class MultiAgentGenerationManager(LLMGenerationManager):
    """Dual-agent rollout manager: Planner -> Executor (DAG-based).

    Planner (single-turn): decomposes a research question into 3-5 ordered
        sub-questions with dependency annotations.
    Executor (multi-turn, DAG waves): researches sub-questions in topological
        order, injecting findings from declared dependencies as context.
    The last executor's answer serves as the final output for reward scoring.
    """

    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        lora_save_dir: str = "./tmp_lora_adapters",
        is_validation: bool = False,
    ):
        super().__init__(tokenizer, actor_rollout_wg, config, is_validation)
        self.lora_save_dir = lora_save_dir
        self._lora_step = 0

    # -------------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------------

    def run_multi_agent_loop(
        self,
        gen_batch: DataProto,
        global_steps: int,
        agent_grpo_idx: Optional[List[int]] = None,
    ) -> MultiAgentResult:
        """Run the dual-agent (Planner -> Executor DAG) rollout.

        Args:
            gen_batch: Input DataProto containing tokenized questions.
            global_steps: Current training step (for logging).
            agent_grpo_idx: GRPO group index for each sample in gen_batch.
        """
        node_rank = int(os.environ.get("PET_NODE_RANK", 0))
        questions = self._extract_questions(gen_batch)
        num_questions = len(questions)
        self._lora_step += 1
        print(
            f"[MultiAgent] node {node_rank}, {num_questions} questions, "
            f"step {global_steps}",
            flush=True,
        )

        # ------------------------------------------------------------------
        # Stage 1: Planner (single-turn generation)
        # ------------------------------------------------------------------
        print("[MultiAgent] Stage 1: Planner", flush=True)
        planner_batch = self._build_planner_batch(questions, gen_batch)
        plan_outputs = self._generate_with_gpu_padding(
            planner_batch, lora_adapter_name="planner"
        )
        plan_texts = self._decode_outputs(plan_outputs)
        parsed_plans = [parse_plan(text) for text in plan_texts]

        self._log_stage_outputs(
            "planner", global_steps, questions, plan_texts, parsed_plans
        )

        # ------------------------------------------------------------------
        # Stage 2: Executor (DAG wave execution)
        # ------------------------------------------------------------------
        print("[MultiAgent] Stage 2: Executor (DAG)", flush=True)
        exec_result = self._run_executor_dag(
            questions, parsed_plans, gen_batch, global_steps
        )

        # Log executor outputs
        self._log_executor_outputs(global_steps, questions, exec_result)

        # Propagate GRPO group index
        if agent_grpo_idx is not None:
            plan_outputs.non_tensor_batch["agent_grpo_idx"] = np.array(
                agent_grpo_idx, dtype=object
            )
            if exec_result.todo_mapping:
                exec_grpo_idx = [
                    agent_grpo_idx[q_idx] for q_idx in exec_result.todo_mapping
                ]
                exec_result.exec_outputs.non_tensor_batch["agent_grpo_idx"] = (
                    np.array(exec_grpo_idx, dtype=object)
                )

        return MultiAgentResult(
            planner_outputs=plan_outputs,
            executor_outputs=exec_result.exec_outputs,
            queries=questions,
            plan_texts=plan_texts,
            parsed_plans=parsed_plans,
            final_answers=exec_result.final_answers,
            all_findings=exec_result.all_findings,
            todo_mapping=exec_result.todo_mapping,
        )

    # -------------------------------------------------------------------------
    # DAG-based Executor Execution
    # -------------------------------------------------------------------------

    def _run_executor_dag(
        self,
        questions: List[str],
        parsed_plans: List[List[SubTask]],
        ref_batch: DataProto,
        global_steps: int,
    ) -> ExecutorDAGResult:
        """Execute sub-questions in DAG wave order with cross-question batching.

        For each wave level, collects tasks from ALL questions at that wave
        depth and batches them together for GPU efficiency.
        """
        num_questions = len(questions)

        # Compute wave schedules for all questions
        all_wave_schedules = [schedule_waves(plan) for plan in parsed_plans]
        max_waves = max((len(ws) for ws in all_wave_schedules), default=0)

        # Per-question findings: {q_idx: {task_index: {"question": ..., "answer": ...}}}
        all_findings: Dict[int, Dict[int, Dict[str, str]]] = {
            i: {} for i in range(num_questions)
        }

        all_exec_output_parts = []
        all_exec_msg_strings = []
        todo_mapping = []

        for wave_idx in range(max_waves):
            # Collect all tasks at this wave level across all questions
            wave_tasks = []  # List of (q_idx, task_dict)
            for q_idx, wave_schedule in enumerate(all_wave_schedules):
                if wave_idx < len(wave_schedule):
                    for task in wave_schedule[wave_idx]:
                        wave_tasks.append((q_idx, task))

            if not wave_tasks:
                continue

            # Build prompts for this wave batch
            messages_list = []
            for q_idx, task in wave_tasks:
                # Build prior_findings from declared dependencies
                dep_findings = {}
                if task.deps:
                    for dep_idx in task.deps:
                        if dep_idx in all_findings[q_idx]:
                            dep_findings[dep_idx] = all_findings[q_idx][dep_idx]

                prior_findings_str = format_prior_findings(dep_findings)
                messages = get_executor_prompt(
                    task.sub_question, prior_findings=prior_findings_str
                )
                messages_list.append(messages)
                todo_mapping.append(q_idx)

            # Tokenize and run executor for this wave
            wave_batch = self._tokenize_messages_to_batch(
                messages_list, ref_batch, tools=EXECUTOR_TOOLS
            )
            wave_msg_strings, wave_outputs = self.run_llm_loop(
                wave_batch, global_steps, lora_adapter_name="executor"
            )

            # Extract findings and store by (q_idx, task_index)
            for i, (q_idx, task) in enumerate(wave_tasks):
                if i < len(wave_msg_strings):
                    answer = self._extract_answer(wave_msg_strings[i])
                    all_findings[q_idx][task.index] = {
                        "question": task.sub_question,
                        "answer": answer,
                    }

            all_exec_msg_strings.extend(wave_msg_strings)
            all_exec_output_parts.append(wave_outputs)

        # Combine all wave outputs into a single DataProto
        if all_exec_output_parts:
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = 0
            padded_parts = _pad_dataproto_for_concat(
                all_exec_output_parts, pad_token_id
            )
            exec_outputs = DataProto.concat(padded_parts)
        else:
            exec_outputs = DataProto.from_dict({
                "input_ids": torch.zeros((0, 1), dtype=torch.long),
                "attention_mask": torch.zeros((0, 1), dtype=torch.long),
                "position_ids": torch.zeros((0, 1), dtype=torch.long),
            })

        # Determine final answers (from the is_final task or last task)
        final_answers = []
        for q_idx in range(num_questions):
            plan = parsed_plans[q_idx]
            if not plan:
                final_answers.append("")
                continue

            final_task = next(
                (t for t in reversed(plan) if t.is_final), plan[-1]
            )
            finding = all_findings[q_idx].get(final_task.index, {})
            final_answers.append(finding.get("answer", ""))

        return ExecutorDAGResult(
            exec_outputs=exec_outputs,
            final_answers=final_answers,
            all_findings=all_findings,
            todo_mapping=todo_mapping,
            exec_msg_strings=all_exec_msg_strings,
        )

    # -------------------------------------------------------------------------
    # Question extraction
    # -------------------------------------------------------------------------

    def _extract_questions(self, gen_batch: DataProto) -> List[str]:
        """Extract question strings from a DataProto batch."""
        return self.parse_question(gen_batch.batch["input_ids"])

    # -------------------------------------------------------------------------
    # Batch builders
    # -------------------------------------------------------------------------

    def _build_planner_batch(
        self, questions: List[str], ref_batch: DataProto
    ) -> DataProto:
        """Build a DataProto batch of planner prompts."""
        messages_list = [get_planner_prompt(q) for q in questions]
        return self._tokenize_messages_to_batch(messages_list, ref_batch)

    # -------------------------------------------------------------------------
    # Tokenization helper
    # -------------------------------------------------------------------------

    def _tokenize_messages_to_batch(
        self,
        messages_list: List[List[Dict]],
        ref_batch: DataProto,
        tools: Optional[List[Dict]] = None,
    ) -> DataProto:
        """Tokenize a list of chat message sequences into a padded DataProto.

        Uses tokenizer.apply_chat_template for proper chat formatting,
        then creates input_ids, attention_mask, and position_ids tensors.
        """
        template_kwargs = {
            "add_generation_prompt": True,
            "tokenize": False,
        }
        if tools is not None:
            template_kwargs["tools"] = tools

        formatted_strings = self.tokenizer.apply_chat_template(
            messages_list, **template_kwargs
        )

        tokenized = self.tokenizer(
            formatted_strings,
            return_tensors="pt",
            padding=True,
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # Left-pad: sort so that pad tokens come first
        pad_mask = input_ids != self.tokenizer.pad_token_id
        sorted_indices = pad_mask.to(torch.int64).argsort(dim=1, stable=True)
        input_ids = input_ids.gather(1, sorted_indices)
        attention_mask = attention_mask.gather(1, sorted_indices)

        position_ids = self.tensor_fn.create_position_ids(attention_mask)

        return DataProto.from_dict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        })

    # -------------------------------------------------------------------------
    # Output decoding & answer extraction
    # -------------------------------------------------------------------------

    def _decode_outputs(self, outputs: DataProto) -> List[str]:
        """Decode token IDs from output DataProto to strings."""
        if "responses" not in outputs.batch.keys():
            print(
                "[MultiAgent] WARNING: _decode_outputs found no 'responses' "
                "key in batch, returning empty list",
                flush=True,
            )
            return []

        responses = outputs.batch["responses"]
        decoded = self.tokenizer.batch_decode(
            responses, skip_special_tokens=False
        )
        decoded = [
            text.replace(self.tokenizer.pad_token or "<|endoftext|>", "").strip()
            for text in decoded
        ]
        return decoded

    @staticmethod
    def _extract_answer(msg: str) -> str:
        """Extract the answer from an executor message string.

        Looks for <answer>...</answer> tags first, then falls back to
        the last meaningful content.
        """
        answer_match = re.search(r"<answer>(.*?)</answer>", msg, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()

        # Fallback: content after last </think> tag
        parts = msg.split("</think>")
        if len(parts) > 1:
            tail = parts[-1].strip()
            if tail:
                return tail[:500]

        return msg[-300:].strip() if msg else ""

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def _log_stage_outputs(
        self,
        stage_name: str,
        global_steps: int,
        questions: List[str],
        texts: List[str],
        parsed_plans: Optional[List[List[SubTask]]] = None,
    ):
        """Log stage outputs to a JSON file for debugging."""
        output_dir = (
            f"./outputs/{self.config.project_name}/"
            f"{self.config.experiment_name}/rollout"
        )
        os.makedirs(output_dir, exist_ok=True)

        log_data = []
        for i, (question, text) in enumerate(zip(questions, texts)):
            entry = {
                "idx": i,
                "question": question,
                "output": text,
            }
            if parsed_plans is not None and i < len(parsed_plans):
                entry["parsed_plan"] = [
                    {"index": t.index, "sub_question": t.sub_question,
                     "deps": t.deps, "is_final": t.is_final}
                    for t in parsed_plans[i]
                ]
            log_data.append(entry)

        filepath = os.path.join(
            output_dir, f"{stage_name}_step_{global_steps}.json"
        )
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4, ensure_ascii=False)

        print(
            f"[MultiAgent] {stage_name}_step_{global_steps}.json saved",
            flush=True,
        )

    def _log_executor_outputs(
        self,
        global_steps: int,
        questions: List[str],
        exec_result: ExecutorDAGResult,
    ):
        """Log executor DAG execution results."""
        output_dir = (
            f"./outputs/{self.config.project_name}/"
            f"{self.config.experiment_name}/rollout"
        )
        os.makedirs(output_dir, exist_ok=True)

        log_data = []
        for q_idx, question in enumerate(questions):
            findings = exec_result.all_findings.get(q_idx, {})
            log_data.append({
                "idx": q_idx,
                "question": question,
                "findings": {
                    str(k): v for k, v in findings.items()
                },
                "final_answer": exec_result.final_answers[q_idx],
            })

        filepath = os.path.join(
            output_dir, f"executor_step_{global_steps}.json"
        )
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4, ensure_ascii=False)

        print(
            f"[MultiAgent] executor_step_{global_steps}.json saved",
            flush=True,
        )
