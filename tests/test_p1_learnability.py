"""The gates that would have caught algorithm_version 1 before the GPU time.

Version 1 shipped an item set solvable without reading the passage, and the
pipeline was green throughout: four options present, gold letter consistent,
adapter attaching, quantization applying, digests stable, manifest reproducible.
Every one of those checks is structural. None of them asks whether the task is
learnable, or whether it is already solved.

Two behavioural gates were added, and this file pins both:

  * the item set carries no substring shortcut, checked on the FROZEN manifest
    rather than only at build time, so a manifest written by an older build is
    condemned rather than trusted;
  * the base model has headroom on the P1 task, measured against a ceiling read
    from P0's own results rather than invented here.

Also pinned: the training partitions are the same size in both languages, so
"English gained more from fine-tuning" cannot be confounded with "English took
more gradient steps".
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from quantlang import config as cfg_mod
from quantlang import p1data

REPO = Path(__file__).resolve().parent.parent


def _load_gate_module():
    spec = importlib.util.spec_from_file_location(
        "check_p1_learnability", REPO / "scripts" / "check_p1_learnability.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cfg():
    return cfg_mod.load()


@pytest.fixture(scope="module")
def manifest():
    return p1data.load_split_manifest()


@pytest.fixture(scope="module")
def gate():
    return _load_gate_module()


# --------------------------------------------------------------------------- #
# the ceiling is read, not invented
# --------------------------------------------------------------------------- #

def test_the_ceiling_comes_from_p0s_own_results(gate):
    """A threshold picked by taste is exactly what this repo forbids.

    The stop condition is P0's best MEASURED cell -- the easiest thing the
    benchmark showed the base model. If the base model does better than that on
    P1 training items, the P1 task is easier than the benchmark it supports and
    fine-tuning has no room to move anything.
    """
    ceiling, cell = gate.p0_ceiling()
    assert 0.0 < ceiling <= 1.0
    assert "/" in cell, "the ceiling must name the cell it came from"

    # It really is the maximum of the frozen table, not a number typed in.
    import csv
    with gate.P0_ACCURACY_TABLE.open(encoding="utf-8") as fh:
        accs = [float(r["accuracy"]) for r in csv.DictReader(fh)]
    assert ceiling == max(accs)


def test_the_ceiling_is_currently_english_fp16(gate):
    """Recorded so a change in P0's results is visible here, not silent."""
    ceiling, cell = gate.p0_ceiling()
    assert cell == "eng_Latn/fp16"
    assert ceiling == pytest.approx(0.8955555555555555)


# --------------------------------------------------------------------------- #
# the shortcut gate runs against the frozen manifest
# --------------------------------------------------------------------------- #

def test_the_frozen_manifest_passes_the_shortcut_gate(gate, manifest, cfg):
    scope = cfg_mod.require(cfg, "finetune.final_scope_languages")
    out = gate.check_shortcut(manifest, scope)
    for key, result in out.items():
        assert result["pass"], key
        assert result["lexical_shortcut_accuracy"] == pytest.approx(0.25)
        assert result["presence_gap"] == 0.0


def test_a_manifest_without_diagnostics_is_condemned(gate, manifest):
    """The v1 manifest had no shortcut measurement at all, and that was the bug.

    A manifest frozen by a build that never measured the shortcut must be
    rejected rather than assumed innocent.
    """
    stripped = {**manifest, "languages": {
        lang: {k: v for k, v in entry.items()
               if k != "construction_diagnostics"}
        for lang, entry in manifest["languages"].items()}}
    with pytest.raises(SystemExit, match="no construction_diagnostics"):
        gate.check_shortcut(stripped, list(manifest["languages"]))


def test_a_shortcut_ridden_manifest_is_rejected(gate, manifest):
    """v1's actual numbers, fed back in: 0.96 shortcut on 100% gold presence."""
    lang = next(iter(manifest["languages"]))
    poisoned = {**manifest, "languages": dict(manifest["languages"])}
    entry = {**poisoned["languages"][lang]}
    diags = {k: dict(v) for k, v in entry["construction_diagnostics"].items()}
    diags["train"].update({
        "lexical_shortcut_accuracy": 0.96,
        "gold_in_context_rate": 1.0,
        "distractor_in_context_rate": 0.10,
    })
    entry["construction_diagnostics"] = diags
    poisoned["languages"][lang] = entry
    with pytest.raises(SystemExit, match="substring presence alone"):
        gate.check_shortcut(poisoned, [lang])


def test_the_gate_runs_without_a_gpu():
    """The CPU half must be runnable before any session is booked."""
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "check_p1_learnability.py"),
         "--no-gpu"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "shortcut=0.2500" in proc.stdout
    # ...and it must say plainly that half a gate is not a gate.
    assert "only half satisfied" in proc.stdout


# --------------------------------------------------------------------------- #
# the two arms train on the same amount of data
# --------------------------------------------------------------------------- #

def test_the_training_partitions_are_equalised(cfg, manifest):
    scope = cfg_mod.require(cfg, "finetune.final_scope_languages")
    if not cfg_mod.require(cfg, "finetune.equalise_train_partition"):
        pytest.skip("equalisation is off")
    sizes = {l: manifest["languages"][l]["n_train_items_equalised"]
             for l in scope}
    assert len(set(sizes.values())) == 1, (
        f"training sizes differ across the final scope: {sizes}. 'English "
        f"gained more from fine-tuning' would then be confounded with 'English "
        f"took more gradient steps'.")


def test_the_cap_is_the_smallest_built_partition(cfg, manifest):
    scope = cfg_mod.require(cfg, "finetune.final_scope_languages")
    cap = p1data.train_equalise_cap(cfg, manifest)
    assert cap == min(manifest["languages"][l]["n_train_items"] for l in scope)
    for lang in scope:
        assert manifest["languages"][lang]["n_train_items_equalised"] == cap


def test_the_cap_ignores_languages_outside_the_final_scope(cfg, manifest):
    """A provenance language rebuilt alongside the scope must not move the cap."""
    padded = {**manifest, "languages": {
        **manifest["languages"],
        "sin_Sinh": {"n_train_items": 12},
    }}
    assert (p1data.train_equalise_cap(cfg, padded)
            == p1data.train_equalise_cap(cfg, manifest))


def test_the_subsample_is_deterministic(cfg):
    items = [{"item_id": f"Article {i:04d}#0"} for i in range(500)]
    first = p1data.select_equalised_train(cfg, "eng_Latn", items, 100)
    second = p1data.select_equalised_train(cfg, "eng_Latn", items, 100)
    assert first == second == sorted(first)
    assert len(first) == 100
    assert set(first) <= {it["item_id"] for it in items}


def test_the_subsample_differs_between_languages(cfg):
    """Seeded per language, like every other selection in this pipeline."""
    items = [{"item_id": f"Article {i:04d}#0"} for i in range(500)]
    assert (p1data.select_equalised_train(cfg, "eng_Latn", items, 100)
            != p1data.select_equalised_train(cfg, "ben_Beng", items, 100))


def test_a_partition_at_or_under_the_cap_is_untouched(cfg):
    items = [{"item_id": f"Article {i:04d}#0"} for i in range(50)]
    assert (p1data.select_equalised_train(cfg, "eng_Latn", items, 100)
            == sorted(it["item_id"] for it in items))


def test_load_partition_applies_the_cap_not_the_caller():
    """Every training run goes through load_partition, so none can miss the trim.

    Doing the trim in the caller would mean any future entry point could train
    on the untrimmed set without anything noticing.
    """
    import inspect
    src = inspect.getsource(p1data.load_partition)
    assert "train_equalise_cap" in src
    assert "select_equalised_train" in src

    # `train_full` stays reachable for diagnostics, and must be asked for by
    # name so it can never be selected by accident.
    assert '"train_full"' in src


# --------------------------------------------------------------------------- #
# low headroom is a WARNING, and passing it is on the record
# --------------------------------------------------------------------------- #

def test_low_headroom_cannot_be_passed_silently():
    """The stop became a warning AFTER it fired, which deserves suspicion.

    The justification is that the original rationale -- "do not fine-tune a task
    the base model already solves" -- tests for accuracy headroom, while P1's
    estimand is whether a language-adapted model QUANTIZES differently. Those
    come apart: check 9 measured a 1.14 FT-vs-Base logit delta at matched
    precision from three optimizer steps, against 0.000000 for the invalid run.

    What must never happen is the warning becoming invisible. Proceeding
    requires an explicit flag.
    """
    src = (REPO / "scripts" / "check_p1_learnability.py").read_text(encoding="utf-8")
    assert "--acknowledge-low-headroom" in src
    assert "if not acknowledge:" in src, (
        "low headroom must still raise unless explicitly acknowledged")


def test_the_acknowledgement_is_written_into_the_report():
    """A reader must be able to tell the limitation was accepted, not missed."""
    src = (REPO / "scripts" / "check_p1_learnability.py").read_text(encoding="utf-8")
    assert 'report["low_headroom_acknowledged"]' in src
    assert 'report["low_headroom_languages"]' in src


def test_the_warning_names_the_consequence_for_rq3():
    """Low headroom bounds an actual claim, and the claim is named."""
    src = (REPO / "scripts" / "check_p1_learnability.py").read_text(encoding="utf-8")
    assert "RQ3" in src, (
        "the warning must say which research question it limits")


def test_the_decision_and_its_evidence_are_recorded_in_the_source():
    """Changing a gate after seeing it fire has to leave a trail."""
    src = (REPO / "scripts" / "check_p1_learnability.py").read_text(encoding="utf-8")
    for token in ("0.970", "0.900", "1.14", "0.000000"):
        assert token in src, (
            f"the measurement {token} that justified the change must be recorded")


def test_the_smoke_delta_says_which_row_is_comparable():
    """Only fp16-vs-fp16 isolates fine-tuning from quantization.

    The baseline is the base model at FP16, so the int8 and nf4 rows measure the
    fine-tuning effect AND the quantization effect together and rise with
    aggressiveness for that reason alone. Reporting them unlabelled invites
    reading 5.31 as "NF4 fine-tuning changed more".
    """
    src = (REPO / "scripts" / "run_p1_smoke.py").read_text(encoding="utf-8")
    assert "max_logit_delta_vs_base_fp16" in src
    assert '"comparable_row": "fp16"' in src
