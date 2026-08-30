"""A second model must be addable without disturbing P0 by one byte.

`models` sits inside freeze_p0.P0_CONFIG_KEYS. Adding an entry there would
change the P0 config digest and break the freeze that gates every notebook --
for a model P0 never ran. So a replication model lives under a separate
top-level key and is reached only after `models` has been searched and missed.

These tests pin that boundary. The expensive failure they prevent is silent: a
replication cell that looks like a P0 cell, or a P0 selection path that quietly
changed behaviour when the second model was added.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from quantlang import config as cfg_mod

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# the frozen boundary
# --------------------------------------------------------------------------- #

def test_p0_freeze_still_passes():
    """The whole point of the separate key. If this fails, nothing else matters."""
    proc = subprocess.run([sys.executable, str(REPO / "scripts" / "freeze_p0.py")],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "P0 freeze intact" in proc.stdout


def test_replication_models_is_not_inside_the_frozen_subtree():
    """A new top-level key leaves the P0 digest untouched; editing `models` does not."""
    src = (REPO / "scripts" / "freeze_p0.py").read_text(encoding="utf-8")
    line = next(l for l in src.splitlines() if l.startswith("P0_CONFIG_KEYS"))
    assert "replication_models" not in line, (
        "replication_models must NOT join the frozen subtree -- that would make "
        "every future model addition break the P0 freeze")
    assert '"models"' in line, "the P0 `models` list must stay frozen"


def test_the_frozen_models_list_still_holds_exactly_p0s_model():
    cfg = cfg_mod.load()
    assert [m["alias"] for m in cfg["models"]] == ["qwen2.5-3b-instruct"]
    assert sum(1 for m in cfg["models"] if m.get("role") == "primary") == 1


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #

def _select(alias=None):
    """Reproduce run_eval's selection without importing its main()."""
    cfg = cfg_mod.load()
    models = cfg_mod.require(cfg, "models")
    replication = cfg.get("replication_models") or []
    role = "p0"
    if alias:
        chosen = [m for m in models if m["alias"] == alias]
        if not chosen:
            chosen = [m for m in replication if m["alias"] == alias]
            role = "replication"
    else:
        chosen = [m for m in models if m.get("role") == "primary"]
    return chosen, role


def test_p0_selection_with_no_flag_is_unchanged():
    """The path that produced P0's results must resolve identically."""
    chosen, role = _select()
    assert len(chosen) == 1
    assert chosen[0]["alias"] == "qwen2.5-3b-instruct"
    assert chosen[0]["hf_id"] == "Qwen/Qwen2.5-3B-Instruct"
    assert role == "p0"


def test_p0_selection_by_explicit_alias_is_unchanged():
    chosen, role = _select("qwen2.5-3b-instruct")
    assert len(chosen) == 1 and role == "p0", (
        "a frozen alias must never fall through to replication_models")


def test_a_replication_alias_resolves():
    chosen, role = _select("bloomz-3b")
    assert len(chosen) == 1
    assert chosen[0]["hf_id"] == "bigscience/bloomz-3b"
    assert role == "replication"
    assert chosen[0].get("role") == "replication"


def test_frozen_models_are_searched_before_replication_models():
    """Order matters: a shadowing alias must resolve to the FROZEN entry."""
    src = (REPO / "scripts" / "run_eval.py").read_text(encoding="utf-8")
    i_frozen = src.index('chosen = [m for m in models if m["alias"] == args.model_alias]')
    i_rep = src.index('chosen = [m for m in replication if m["alias"] == args.model_alias]')
    assert i_frozen < i_rep


def test_an_unknown_alias_fails_clearly(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "run_eval.py"),
         "--precision", "fp16", "--langs", "eng_Latn",
         "--model-alias", "no-such-model",
         "--outdir", str(tmp_path), "--tag", "smoke"],
        capture_output=True, text=True)
    assert proc.returncode != 0
    out = proc.stdout + proc.stderr
    assert "FATAL" in out
    assert "no-such-model" in out
    assert "bloomz-3b" in out, "the error must list what IS available"


# --------------------------------------------------------------------------- #
# reproducibility and the pre-flight
# --------------------------------------------------------------------------- #

def test_every_replication_model_is_preflighted_or_marked_unverified():
    """Listing a model is not the same as clearing it."""
    for m in cfg_mod.load().get("replication_models", []):
        note = (m.get("note") or "").lower()
        if not m.get("revision"):
            assert "not yet" in note or "unverified" in note or "gated" in note, (
                f"{m['alias']} is unpinned, so its note must say it is not ready")


def test_every_listed_replication_model_is_pinned():
    """An unpinned run is not reproducible and its numbers cannot go in a paper.

    This replaces a test that used gemma-2-2b-it as the unpinned example. Gemma
    is now pinned, so the example was gone -- but the invariant it protected is
    the one that matters, and it is stronger stated over every model.
    """
    for m in cfg_mod.load().get("replication_models", []):
        assert m.get("revision"), f"{m['alias']} has no pinned revision"
        assert len(m["revision"]) == 40, (
            f"{m['alias']} revision is not a full 40-char commit SHA")


def test_the_generator_still_refuses_an_unpinned_model():
    """The guard itself must survive, even with nothing currently unpinned."""
    src = (REPO / "scripts" / "make_replication_notebook.py").read_text(encoding="utf-8")
    assert 'if not entry.get("revision")' in src
    assert "no pinned revision" in src


def test_the_generated_gate_notebook_is_fp16_only():
    """The floor gate exists to stop us buying uninterpretable INT8/NF4 cells."""
    p = REPO / "notebooks" / "kaggle_rep_bloomz-3b_fp16.ipynb"
    if not p.exists():
        pytest.skip("notebook not generated in this tree")
    cells = json.loads(p.read_text(encoding="utf-8"))["cells"]
    code = "\n".join("".join(c["source"])
                     for c in cells if c["cell_type"] == "code")
    prose = "\n".join("".join(c["source"])
                      for c in cells if c["cell_type"] == "markdown")

    assert '"--precision", "fp16"' in code
    assert "--all-precisions" not in code
    # Only the CODE may not request a quantized precision. The prose says "does
    # not run INT8 or NF4", which is the constraint being documented -- an
    # earlier version of this test matched that sentence and failed the notebook
    # for explaining itself.
    for banned in ('"nf4"', '"int8_llmint8"', "--all-precisions"):
        assert banned not in code, f"gate notebook must not request {banned}"
    assert "NF4" in prose, "the notebook should say what it is deliberately not running"

    assert "0.30" in code, "the pre-registered floor threshold must be applied"
    assert "freeze_p0.py" in code, "a replication run must re-prove the P0 freeze"
    assert "probe_model_compat.py" in code, "the tokenizer pre-flight must run first"


def test_the_preflight_still_blocks_an_incompatible_tokenizer():
    src = (REPO / "scripts" / "probe_model_compat.py").read_text(encoding="utf-8")
    assert "letters_single_token" in src
    assert "do NOT change the scoring method" in src


def test_h5_is_recorded_before_any_replication_result():
    """H5 must be fixed in advance and must not claim causality."""
    doc = (REPO / "docs" / "H5_PREREGISTRATION.md").read_text(encoding="utf-8")
    assert "No causal claim is made" in doc
    assert "Wilson 95% lower bound > 0.30" in doc
    for lang in ("ben_Beng", "npi_Deva", "asm_Beng", "sin_Sinh"):
        assert lang in doc, f"{lang} must have a stated directional prediction"
    assert "not adjusted to accommodate an outcome" in doc
