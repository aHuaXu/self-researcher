"""Task 5: interleaved Planner <-> frozen Executor control-flow (pure-python skeleton).

run_loop_pure is the tensor-free control flow (unit-testable + skeleton for run_loop):
outer Planner loop emits a subtask or <answer> (<= max_planner_turns); each subtask is
run by the frozen Executor to produce findings.
"""
from scrl.llm_agent.interleaved_generation import InterleavedGenerationManager


class FakePlanner:
    """Turn 1 emits a subtask, turn 2 emits <answer> (stop)."""

    def __init__(self):
        self.calls = 0

    def step(self, ctx):
        self.calls += 1
        if self.calls >= 2:
            return ("<answer>done</answer>", True)
        return (f"subtask-{self.calls}", False)


class FakeExecutor:
    def run(self, subtask, ctx):
        return f"findings-for-{subtask}"


def test_loop_stops_on_answer_and_caps_turns():
    mgr = InterleavedGenerationManager(planner=FakePlanner(), executor=FakeExecutor(),
                                       max_planner_turns=5)
    trace = mgr.run_loop_pure(question="q")
    assert trace["answer"] == "done"
    assert len(trace["subtasks"]) == 1          # turn 2 was the answer
    assert trace["findings"] == ["findings-for-subtask-1"]


class NeverAnswerPlanner:
    """Always emits a subtask (never answers) -> loop must be capped by max_planner_turns."""

    def __init__(self):
        self.calls = 0

    def step(self, ctx):
        self.calls += 1
        return (f"subtask-{self.calls}", False)


def test_loop_capped_at_max_planner_turns():
    mgr = InterleavedGenerationManager(planner=NeverAnswerPlanner(), executor=FakeExecutor(),
                                       max_planner_turns=3)
    trace = mgr.run_loop_pure(question="q")
    assert trace["answer"] is None              # never answered
    assert len(trace["subtasks"]) == 3          # capped
    assert len(trace["findings"]) == 3
