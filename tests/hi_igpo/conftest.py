"""
conftest.py for hi_igpo tests.

verl/__init__.py and several verl utility modules pull in heavy dependencies
(pandas, ray, tensordict, flash_attn, etc.) that are not available in the
lightweight .venv-test environment.

Strategy: pre-register stub modules for the verl package hierarchy so Python
can import verl.trainer.ppo.core_algos directly without executing any of the
heavy __init__.py code.  We also stub out verl.utils.torch_functional with
lightweight pure-torch implementations of the few helpers that core_algos.py
actually uses.

The stubs must be installed BEFORE any test-module import runs, which is
guaranteed by conftest.py's execution order relative to test collection.
"""
import sys
import types
import os

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_pkg_stub(name: str, path: str) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__path__ = [path]
    m.__package__ = name
    m.__spec__ = None
    return m


# ── verl top-level (bypass heavy verl/__init__.py) ────────────────────────
if "verl" not in sys.modules:
    verl_stub = _make_pkg_stub("verl", os.path.join(_ROOT, "verl"))
    sys.modules["verl"] = verl_stub

# ── verl.trainer ──────────────────────────────────────────────────────────
if "verl.trainer" not in sys.modules:
    trainer_stub = _make_pkg_stub("verl.trainer", os.path.join(_ROOT, "verl", "trainer"))
    sys.modules["verl.trainer"] = trainer_stub

# ── verl.trainer.ppo ──────────────────────────────────────────────────────
if "verl.trainer.ppo" not in sys.modules:
    ppo_stub = _make_pkg_stub("verl.trainer.ppo", os.path.join(_ROOT, "verl", "trainer", "ppo"))
    sys.modules["verl.trainer.ppo"] = ppo_stub

# ── verl.utils ────────────────────────────────────────────────────────────
if "verl.utils" not in sys.modules:
    utils_stub = _make_pkg_stub("verl.utils", os.path.join(_ROOT, "verl", "utils"))
    sys.modules["verl.utils"] = utils_stub

# ── verl.utils.torch_functional — lightweight stub ────────────────────────
# core_algos.py imports this as `verl_F` and uses: masked_whiten, masked_mean,
# entropy_from_logits, clip_by_value.  compute_igpo_turn_advantage does NOT
# call any verl_F functions, so the stub bodies don't need to be correct for
# our test — they just need to exist so the module-level import succeeds.
if "verl.utils.torch_functional" not in sys.modules:
    tf_stub = types.ModuleType("verl.utils.torch_functional")

    def masked_mean(values, mask, axis=None):
        return (values * mask).sum() / mask.sum().clamp(min=1)

    def masked_whiten(values, mask, shift_mean=True):
        mean = masked_mean(values, mask)
        var = masked_mean((values - mean) ** 2, mask)
        whitened = (values - mean) * torch.rsqrt(var + 1e-8)
        return whitened * mask

    def entropy_from_logits(logits):
        pd = torch.softmax(logits, dim=-1)
        return -torch.sum(pd * torch.log(pd + 1e-8), dim=-1)

    def clip_by_value(x, lo, hi):
        return torch.clamp(x, lo, hi)

    tf_stub.masked_mean = masked_mean
    tf_stub.masked_whiten = masked_whiten
    tf_stub.entropy_from_logits = entropy_from_logits
    tf_stub.clip_by_value = clip_by_value

    sys.modules["verl.utils.torch_functional"] = tf_stub
    # Also attach to parent stub
    sys.modules["verl.utils"].torch_functional = tf_stub
