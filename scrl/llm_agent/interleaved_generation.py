"""Hi-IGPO Phase 2b: interleaved Planner <-> frozen Executor generation.

Outer loop: the Planner (trained LoRA) proposes one subtask per turn, or emits
<answer> to stop (<= max_planner_turns). Inner: each subtask is executed by the
FROZEN Executor (Phase-1 model, no grad) to produce findings, which feed the next
Planner turn. After each Planner turn a belief Bel_t is computed; IG_t = Bel_t - Bel_{t-1}.

`run_loop_pure` is the tensor-free control flow — unit-testable and the skeleton for the
real `run_loop` (which wraps the same flow with DataProto / rollout / belief tensors).
"""


class InterleavedGenerationManager:
    def __init__(self, planner, executor, max_planner_turns=5, **kw):
        self.planner = planner
        self.executor = executor
        self.max_planner_turns = max_planner_turns

    def run_loop_pure(self, question):
        """Pure-python control flow (no tensors). Returns the interleaved trace.

        planner.step(ctx) -> (output_str, is_answer)
        executor.run(subtask, ctx) -> findings_str

        Returns dict:
            answer:   final answer string (None if the Planner never answered within the cap)
            subtasks: list of subtask strings the Planner proposed
            findings: list of Executor findings (one per subtask, same order)
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

    def run_loop(self, batch):
        """Real rollout (Task 5 Step 5 — server/GPU).

        Wraps run_loop_pure's control flow with tensors:
          - outer Planner turns generated via the actor (LoRA), one subtask or <answer> each;
          - each subtask executed by the FROZEN Executor (reuse LLMGenerationManager.run_llm_loop,
            requires_grad=False) to produce findings appended to the rolling context;
          - after each Planner turn, compute Bel_t via compute_all_turns_vectorized;
          - record each Planner turn's end-token position (turn_end_positions) + the Bel_t sequence.
        Output: DataProto with the Planner sequence + turn_end_positions + beliefs, consumed by
        Task 6 (scatter_planner_token_rewards) -> token_level_rewards / turn_boundary_mask, then
        the same adv_estimator='igpo' advantage path. Does NOT emit turn_records.
        """
        raise NotImplementedError(
            "run_loop (real rollout) is Task 5 Step 5 — implemented/validated on the server; "
            "the tensor-free control flow lives in run_loop_pure."
        )
