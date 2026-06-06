"""Hi-IGPO Phase 2b: interleaved Planner <-> frozen Executor generation.

Outer loop: the Planner (trained LoRA) proposes one subtask per turn, or emits
<answer> to stop (<= max_planner_turns). Inner: each subtask is executed by the
FROZEN Executor (Phase-1 model, no grad) to produce findings, which feed the next
Planner turn. After each Planner turn a belief Bel_t is computed; IG_t = Bel_t - Bel_{t-1}.

`run_loop_pure` is the tensor-free control flow — unit-testable and the skeleton for the
real `run_loop`. `run_loop` reuses MultiAgentGenerationManager's dual-LoRA rollout helpers
(_build_planner_batch / _generate_with_gpu_padding(lora=) / run_llm_loop(lora=) /
_tokenize_messages_to_batch / _decode_outputs) and is iterated on the server (GPU-only).
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class InterleavedGenerationManager:
    """Pure-python control flow (no torch) — see run_loop_pure. The tensor rollout lives in
    InterleavedRolloutManager (subclasses MultiAgentGenerationManager) below to avoid importing
    heavy deps when only the control flow is needed (unit tests)."""

    def __init__(self, planner, executor, max_planner_turns=5, **kw):
        self.planner = planner
        self.executor = executor
        self.max_planner_turns = max_planner_turns

    def run_loop_pure(self, question):
        """Pure-python control flow (no tensors). Returns the interleaved trace.

        planner.step(ctx) -> (output_str, is_answer);  executor.run(subtask, ctx) -> findings_str
        Returns dict(answer, subtasks, findings).
        """
        subtasks, findings = [], []
        answer = None
        ctx = {"question": question, "findings": findings}
        for _ in range(self.max_planner_turns):
            out, is_answer = self.planner.step(ctx)
            if is_answer:
                answer = out.replace("<answer>", "").replace("</answer>", "").strip()
                break
            subtasks.append(out)
            findings.append(self.executor.run(out, ctx))
        return {"answer": answer, "subtasks": subtasks, "findings": findings}


@dataclass
class InterleavedTrace:
    """Per-sample interleaved rollout record (assembled into tensors by run_loop)."""
    question: str
    planner_texts: List[str] = field(default_factory=list)   # planner generation per turn (trained)
    subtasks: List[str] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)        # executor output per subtask (a:answer + b:evidence)
    answer: Optional[str] = None
    # token-level (filled during assembly): each planner turn's end-token position in the planner seq
    turn_end_positions: List[int] = field(default_factory=list)
    beliefs: List[float] = field(default_factory=list)       # Bel_0..Bel_T


def assemble_planner_sequence(planner_turn_ids, findings_ids):
    """Interleave Planner per-turn token ids with Executor findings into one response frame.

    planner_turn_ids: List[List[int]] — token ids of each Planner generation turn, in order;
        entries [0..T-1] are subtask turns, the LAST entry is the answer turn (T subtasks total).
    findings_ids: List[List[int]] — Executor findings observation per subtask
        (len == len(planner_turn_ids) - 1; the answer turn has no findings after it).

    Layout (response frame), findings masked out of the loss:
        [subtask_0][findings_0][subtask_1][findings_1]...[subtask_{T-1}][findings_{T-1}][answer]

    Returns (response_ids, loss_mask, turn_end_positions):
      - loss_mask: 1 on Planner tokens (trained), 0 on findings observations.
      - turn_end_positions: last-token index of each Planner turn in the response frame,
        ordered [subtask_0_end, ..., subtask_{T-1}_end, answer_end]; length == #subtasks + 1,
        matching scatter_planner_token_rewards (T IG turn-ends + 1 F1/answer position).
    """
    if not planner_turn_ids:
        raise ValueError("planner_turn_ids must contain at least the answer turn")
    if len(findings_ids) != len(planner_turn_ids) - 1:
        raise ValueError(
            f"findings_ids ({len(findings_ids)}) must equal #subtask turns "
            f"({len(planner_turn_ids) - 1}) = len(planner_turn_ids) - 1"
        )

    response_ids, loss_mask, turn_end_positions = [], [], []
    for t, turn_ids in enumerate(planner_turn_ids):
        if len(turn_ids) == 0:
            raise ValueError(f"planner turn {t} is empty")
        response_ids.extend(turn_ids)
        loss_mask.extend([1] * len(turn_ids))
        turn_end_positions.append(len(response_ids) - 1)   # last planner token of this turn
        # findings observation follows every subtask turn (not the final answer turn)
        if t < len(findings_ids):
            obs = findings_ids[t]
            response_ids.extend(obs)
            loss_mask.extend([0] * len(obs))
    return response_ids, loss_mask, turn_end_positions


def _parse_planner_turn(text: str):
    """(is_answer, payload). <answer>..</answer> -> answer; else the text is a subtask."""
    if "<answer>" in text and "</answer>" in text:
        return True, text.split("<answer>")[1].split("</answer>")[0].strip()
    # else: treat reasoning+content as the subtask instruction (align with planner prompt on server)
    return False, text.strip()


def build_interleaved_rollout_manager(tokenizer, actor_rollout_wg, config, lora_save_dir,
                                      max_planner_turns=5, gt_computer=None):
    """Factory: returns an InterleavedRolloutManager (defined lazily to avoid importing
    MultiAgentGenerationManager / torch when only run_loop_pure is needed)."""
    from scrl.llm_agent.multi_agent_generation import MultiAgentGenerationManager
    from research_agent.prompts.planner import get_planner_prompt
    from research_agent.prompts.executor import get_executor_prompt

    class InterleavedRolloutManager(MultiAgentGenerationManager):
        """Real interleaved rollout (WIP — iterated on server). Reuses the dual-LoRA helpers
        of MultiAgentGenerationManager; replaces its one-shot DAG `run_multi_agent_loop`."""

        def __init__(self):
            super().__init__(tokenizer=tokenizer, actor_rollout_wg=actor_rollout_wg,
                             config=config, lora_save_dir=lora_save_dir)
            self.max_planner_turns = max_planner_turns
            self.gt_computer = gt_computer

        def run_loop(self, gen_batch, global_steps, ground_truths):
            """Interleaved Planner(LoRA)<->frozen Executor rollout.

            Returns (planner_output: DataProto, turn_end_positions: List[List[int]],
                     beliefs: List[List[float]], answers: List[str]).
            Consumed by Task 7: per-sample scatter_planner_token_rewards -> (bs,L)
            token_level_rewards/turn_boundary_mask -> compute_igpo_turn_advantage.
            """
            import numpy as np  # noqa

            questions = self._extract_questions(gen_batch)
            n = len(questions)
            traces = [InterleavedTrace(question=q) for q in questions]
            # running planner chat per sample (question prompt + assistant/user turns)
            planner_msgs: List[List[Dict]] = [list(get_planner_prompt(q)) for q in questions]
            active = list(range(n))

            for t in range(self.max_planner_turns):
                if not active:
                    break
                # --- Planner: one generation for each active sample ---
                batch = self._tokenize_messages_to_batch([planner_msgs[i] for i in active], gen_batch)
                plan_out = self._generate_with_gpu_padding(batch, lora_adapter_name="planner")
                texts = self._decode_outputs(plan_out)

                next_active, subtask_rows = [], []
                for k, i in enumerate(active):
                    is_answer, payload = _parse_planner_turn(texts[k])
                    traces[i].planner_texts.append(texts[k])
                    planner_msgs[i].append({"role": "assistant", "content": texts[k]})
                    if is_answer:
                        traces[i].answer = payload
                    else:
                        traces[i].subtasks.append(payload)
                        subtask_rows.append(i)
                        next_active.append(i)

                # --- Executor (FROZEN): run each subtask, inject findings back ---
                if subtask_rows:
                    exec_msgs = [get_executor_prompt(traces[i].subtasks[-1]) for i in subtask_rows]
                    exec_batch = self._tokenize_messages_to_batch(
                        exec_msgs, gen_batch, tools=getattr(self, "tools", None))
                    _, exec_out = self.run_llm_loop(
                        exec_batch, global_steps, lora_adapter_name="executor")
                    findings = self._decode_outputs(exec_out)   # a:answer + b:evidence (per design §3)
                    for j, i in enumerate(subtask_rows):
                        traces[i].findings.append(findings[j])
                        planner_msgs[i].append({"role": "user", "content": findings[j]})
                active = next_active

            # ---- Belief + tensor assembly (WIP: iterate on server) ----
            # Per design §4/§5: build each sample's Planner sequence (planner_texts interleaved with
            # findings observations; response_mask=1 only on planner_texts), record each planner turn's
            # end-token position into trace.turn_end_positions, and compute Bel_0..Bel_T via
            # self.gt_computer.compute_all_turns_vectorized(... ground_truth ...). IG_t = Bel diffs.
            # Then Task 7 calls scatter_planner_token_rewards(beliefs, f1, turn_end_positions, L).
            planner_output, turn_end_positions, beliefs, answers = self._assemble_planner_tensors(
                traces, ground_truths, gen_batch)
            return planner_output, turn_end_positions, beliefs, answers

        def _assemble_planner_tensors(self, traces, ground_truths, gen_batch):
            """Assemble the Planner sequence DataProto + turn_end_positions + beliefs.

            WIP — the tensor bookkeeping (concat planner turns + findings observations into one
            (bs, L) sequence with response_mask, recording turn-end token positions, and the
            vectorized belief forward) is finalized against the live runtime on the server.
            """
            raise NotImplementedError(
                "_assemble_planner_tensors: planner-sequence assembly + vectorized belief is the "
                "server-iteration step (needs live DataProto/tokenizer/model). Control flow above "
                "(planner<->frozen-executor interleave, findings injection) is complete."
            )

    return InterleavedRolloutManager()
