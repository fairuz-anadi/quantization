"""The FT arm must not be the Base arm wearing a different filename.

This file exists because of a run that had none of these checks. A full English
P1 fine-tune, merge and three-precision evaluation completed -- roughly 99
GPU-minutes -- and the "fine-tuned" logits came back bit-identical to the base
model's on all 900 BELEBELE items at every precision, maximum difference
0.000000.

Nothing failed, because nothing compared the two arms. The pipeline's checks
were all structural: `assert_adapter_applied` proves LoRA layers EXIST, the
dtype check proves the merged checkpoint is FP16, and the smoke test's check 7
proves the three PRECISIONS differ from each other. None of them asks whether
the fine-tuned weights differ from the base weights.

Two root causes, and one test class for each.

  * `evaluate_cell` had no way to load a merged checkpoint at all, so the one
    full P1 evaluation had to be run by ad-hoc code outside the repo. The
    archived filenames carry `model_alias=qwen2.5-3b-instruct` -- the BASE alias
    -- rather than `ft_alias(...)`, which is what that code scored.

  * `merge_and_save` never verified that merging moved any weight, so an adapter
    whose B matrices were still at their zero initialisation would produce a
    checkpoint identical to the base model and pass every check.

A bit-identical result is not evidence that fine-tuning had no effect. Merging a
trained LoRA perturbs FP16 weights; even an adapter that learned to reproduce
the base model would differ in the low bits. Exactly 0.000000 across 900 items
is the signature of the base weights being scored.
"""

import inspect
import subprocess
import sys
from pathlib import Path

import pytest

from quantlang import evaluate, finetune, model as model_mod

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# the FT arm has a sanctioned path
# --------------------------------------------------------------------------- #

def test_evaluate_cell_can_load_a_merged_checkpoint():
    """Without this parameter there is no way to evaluate an FT cell in-repo."""
    sig = inspect.signature(evaluate.evaluate_cell)
    assert "local_checkpoint" in sig.parameters
    assert sig.parameters["local_checkpoint"].default is None, (
        "defaulting to anything but None would change P0's behaviour")


def test_model_load_still_defaults_to_the_hub():
    """P0's call path must be byte-identical to what produced its results."""
    sig = inspect.signature(model_mod.load)
    assert sig.parameters["local_checkpoint"].default is None


def test_run_eval_exposes_the_ft_arm():
    """The FT arm is reachable from the command line, so it is never ad-hoc."""
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "run_eval.py"), "--help"],
        capture_output=True, text=True, check=True).stdout
    assert "--local-checkpoint" in out
    assert "--ft-lang" in out


def test_a_missing_checkpoint_stops_the_run(tmp_path):
    """Silently falling back to the base model is the failure being prevented."""
    with pytest.raises(FileNotFoundError, match="not a directory"):
        evaluate.evaluate_cell(
            {"scoring": {"max_input_tokens": 10, "method": "letter_logit"}},
            hf_id="x", model_alias="x", revision="x", precision="fp16",
            lang="eng_Latn", outdir=tmp_path, tag="t",
            local_checkpoint=str(tmp_path / "does_not_exist"))


@pytest.mark.parametrize("args", [
    ["--ft-lang", "eng_Latn"],                       # ft-lang without checkpoint
    ["--local-checkpoint", "nowhere"],               # checkpoint without ft-lang
])
def test_half_specified_ft_arms_are_refused(tmp_path, args):
    """Either flag alone would mislabel a cell."""
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "run_eval.py"),
         "--precision", "fp16", "--outdir", str(tmp_path), "--tag", "smoke",
         "--langs", "eng_Latn", *args],
        capture_output=True, text=True)
    assert proc.returncode != 0
    assert "FATAL" in (proc.stdout + proc.stderr)


def test_an_ft_cell_cannot_collide_with_a_base_cell():
    """The alias goes in the filename, so the two arms cannot overwrite."""
    base = "qwen2.5-3b-instruct"
    for lang in ("eng_Latn", "ben_Beng"):
        alias = finetune.ft_alias(base, lang)
        assert alias != base
        assert lang in alias
        assert "__" not in alias, "'__' is the field separator in raw filenames"


def test_the_raw_record_says_which_weights_were_scored():
    """The archived v1 filenames could not distinguish the arms. These can.

    `weights_from` and `arm` are written into every JSONL row and every meta
    manifest, so a Base row and an FT row are distinguishable in the raw output
    without reference to anything else.
    """
    src = (REPO / "quantlang" / "evaluate.py").read_text(encoding="utf-8")
    assert src.count('"weights_from": local_checkpoint or "hub"') == 2, (
        "weights provenance must be recorded in BOTH the per-item record and "
        "the meta manifest")
    assert src.count('"arm": "base" if local_checkpoint is None '
                     'else "finetuned"') == 2


# --------------------------------------------------------------------------- #
# merging must actually move the weights
# --------------------------------------------------------------------------- #

class _FakeParam:
    def __init__(self, tensor):
        self.tensor = tensor

    def detach(self):
        return self

    def to(self, *_args, **_kwargs):
        return self.tensor


class _FakeMerged:
    """Just enough of a model to exercise assert_merge_moved offline."""

    def __init__(self, weights):
        self._weights = weights

    def named_parameters(self):
        return [(f"{k}.weight", _FakeParam(v)) for k, v in self._weights.items()]


def _tensor(*values):
    import torch
    return torch.tensor(values, dtype=torch.float32)


def test_an_unmoved_merge_is_rejected():
    """The zero-delta case: exactly what the invalid English run produced."""
    before = {"layer.0": _tensor(1.0, 2.0, 3.0)}
    merged = _FakeMerged({"layer.0": _tensor(1.0, 2.0, 3.0)})
    with pytest.raises(finetune.FineTuneError, match="changed nothing"):
        finetune.assert_merge_moved(merged, before)


def test_a_real_merge_is_accepted_and_measured():
    before = {"layer.0": _tensor(1.0, 2.0, 3.0),
              "layer.1": _tensor(0.0, 0.0)}
    merged = _FakeMerged({"layer.0": _tensor(1.0, 2.5, 3.0),
                          "layer.1": _tensor(0.0, 0.0)})
    out = finetune.assert_merge_moved(merged, before)
    assert out["n_modules_checked"] == 2
    assert out["max_abs_weight_delta"] == pytest.approx(0.5)
    assert out["largest_movement_in"] == "layer.0"


def test_a_merge_that_cannot_be_verified_is_rejected():
    """An unverifiable merge is treated as a failed one, not waved through."""
    before = {"layer.0": _tensor(1.0)}
    merged = _FakeMerged({"something.else": _tensor(9.0)})
    with pytest.raises(finetune.FineTuneError, match="cannot be verified"):
        finetune.assert_merge_moved(merged, before)


def test_the_threshold_only_separates_moved_from_identical():
    """It is not a "did it learn enough" judgement, and must not become one.

    Any real LoRA update clears this by orders of magnitude; an untrained
    adapter, whose B matrices are still zero, produces exactly 0.0.
    """
    assert finetune.MIN_MERGE_DELTA <= 1e-6


def test_merge_and_save_verifies_before_writing():
    """The check has to run on the path that produces real checkpoints."""
    src = inspect.getsource(finetune.merge_and_save)
    assert "_target_weight_snapshot" in src
    assert "assert_merge_moved" in src
    assert src.index("assert_merge_moved") < src.index("save_pretrained"), (
        "verify the merge before writing the checkpoint, not after")


def test_the_smoke_test_compares_the_two_arms():
    """Check 9. Checks 1-8 all passed on the invalid run."""
    src = (REPO / "scripts" / "run_p1_smoke.py").read_text(encoding="utf-8")
    assert "9_ft_arm_differs_from_base_arm" in src
    assert "max_logit_delta_vs_base" in src


# --------------------------------------------------------------------------- #
# the environment must be able to build a LoRA layer at all
# --------------------------------------------------------------------------- #

def test_an_unusable_peft_dispatch_is_reported_clearly(monkeypatch):
    """Kaggle ships torchao 0.10.0; a current PEFT wants >= 0.16.0.

    PEFT builds a fixed dispatcher list for every LoRA layer, and
    `dispatch_torchao` calls `is_torchao_available()` whether or not torchao is
    used. That helper RAISES on an out-of-range version rather than returning
    False, so no adapter can attach and fine-tuning cannot run -- with a
    traceback per target module that blames torchao instead of the mismatch.

    Nothing in this project uses torchao; INT8 and NF4 are both bitsandbytes.
    """
    import peft.import_utils as iu

    def boom():
        raise ImportError(
            "Found an incompatible version of torchao. Found version 0.10.0, "
            "but only versions above 0.16.0 are supported")

    monkeypatch.setattr(iu, "is_torchao_available", boom)
    with pytest.raises(finetune.FineTuneError) as excinfo:
        finetune.assert_peft_dispatch_is_usable()
    message = str(excinfo.value)
    assert "pip uninstall -y torchao" in message, (
        "the error must name the fix, not just the symptom")
    assert "bitsandbytes" in message


def test_a_usable_dispatch_passes_silently():
    finetune.assert_peft_dispatch_is_usable()


def test_the_guard_runs_before_any_adapter_is_built():
    """Checked up front, so the failure is one line and not one per layer."""
    src = inspect.getsource(finetune.attach_adapter)
    assert "assert_peft_dispatch_is_usable()" in src
    assert src.index("assert_peft_dispatch_is_usable") < src.index("get_peft_model("), \
        "the guard must fire before get_peft_model is called"


def test_probe_env_fails_the_session_on_a_broken_dispatch():
    """Session A's cheapest cell must catch this, not hour one of training."""
    src = (REPO / "scripts" / "probe_env.py").read_text(encoding="utf-8")
    assert "peft_lora_dispatch" in src
    assert 'out["peft_lora_dispatch"] == "ok"' in src, (
        "the dispatch check must gate ok_for_experiment")
    assert src.index("peft_lora_dispatch") < src.index("no CUDA device visible"), (
        "the check needs no GPU and must not hide behind the CUDA early exit")


def test_torchao_version_is_recorded_in_run_provenance():
    """So a future breakage is diagnosable from the manifest alone."""
    for module in ("evaluate.py", "finetune.py"):
        src = (REPO / "quantlang" / module).read_text(encoding="utf-8")
        assert '"torchao"' in src, module


def test_the_notebooks_only_read_keys_the_smoke_test_writes():
    """A rename in the writer must not leave the reader pointing at nothing.

    This exact bug shipped: `max_logit_delta_vs_base` was renamed to
    `*_fp16` in run_p1_smoke.py, the notebook's report-reader cell was not
    updated, and a passing smoke run died on a KeyError while displaying its
    own result. Cheap to catch, annoying to hit mid-session.
    """
    import re
    gen = (REPO / "scripts" / "make_p1_notebooks.py").read_text(encoding="utf-8")
    smoke = (REPO / "scripts" / "run_p1_smoke.py").read_text(encoding="utf-8")

    # Keys the notebooks pull off a smoke check dict.
    referenced = set(re.findall(r'd\["([a-z0-9_]+)"\]', gen))
    assert referenced, "expected the generator to read some smoke-report keys"
    for key in referenced:
        assert f'"{key}"' in smoke, (
            f"notebooks read check key {key!r}, which run_p1_smoke.py never "
            f"writes")
