"""P0 must be exactly what it was before P1 existed.

P0's numbers are already in the paper's chain of custody. P1 is additive by
construction, but "additive by construction" is a claim, and this file is what
turns it into a check that runs on every `pytest`.

Two mechanisms, because one is not enough:

  * BYTES, for the files that define P0 outright -- the item manifest, the
    pinned revisions, the P0 modules, the result tables.
  * BEHAVIOUR, for the three files section 11 of the brief permits P1 to extend
    (`config.py`, `model.py`, `evaluate.py`). Byte-freezing those would forbid
    the local-checkpoint argument P1 is required to add, so instead the
    P0-relevant contract they implement is pinned directly.

The missing P0 raw provenance is handled honestly: those digests are null, and
the tests below require the registry to keep SAYING they are missing rather
than letting the gap quietly disappear.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import freeze_p0  # noqa: E402

from quantlang import config as cfg_mod  # noqa: E402
from quantlang import schema, statistics  # noqa: E402
from quantlang.config import REPO_ROOT  # noqa: E402


@pytest.fixture(scope="module")
def registry():
    return freeze_p0.load_registry()


# --------------------------------------------------------------------------- #
# bytes
# --------------------------------------------------------------------------- #

def test_registry_exists():
    assert freeze_p0.FREEZE_PATH.exists(), (
        "configs/p0_freeze.json is missing. Run "
        "`python scripts/freeze_p0.py --register` before doing P1 work.")


def test_p0_is_intact():
    """The whole guard, in one assertion."""
    problems = freeze_p0.check()
    assert not problems, "P0 FREEZE VIOLATED:\n  " + "\n  ".join(problems)


def test_p0_config_subtree_digest_is_unchanged(registry):
    assert freeze_p0.config_p0_digest() == registry["config_p0_subtree"]["sha256"]


def test_adding_the_p1_block_did_not_disturb_p0(registry):
    """The finetune block exists AND the P0 digest still matches.

    If this passes, the boundary works: P1 configuration lives in the same file
    without touching anything P0 reads.
    """
    cfg = cfg_mod.load()
    assert cfg.get("finetune") is not None, "the P1 block is missing"
    assert freeze_p0.config_p0_digest(cfg) == registry["config_p0_subtree"]["sha256"]


def test_every_strict_file_is_registered_and_present(registry):
    for rel, entry in registry["strict_files"].items():
        assert entry["sha256"] is not None, f"{rel} was never registered"
        assert (REPO_ROOT / rel).exists(), f"{rel} has gone missing"


def test_p0_result_tables_are_covered_by_the_guard(registry):
    """The tables are the P0 numbers; they must be inside the fence."""
    covered = set(registry["strict_files"])
    for name in ("tidy", "accuracy", "degradation", "interaction", "latency"):
        assert f"results/ALL_P0_RESULTS/tables/{name}.csv" in covered


# --------------------------------------------------------------------------- #
# missing provenance -- declared, never fabricated
# --------------------------------------------------------------------------- #

def test_missing_p0_raw_provenance_is_declared_not_invented(registry):
    """P0's per-item output is not committed. The registry must keep saying so.

    This is a pre-existing gap, so it does not fail the suite -- but it must
    stay visible. A null digest here means NOT YET KNOWN, exactly as a null in
    experiment.yaml does.
    """
    prov = registry["p0_raw_provenance"]["files"]
    assert prov, "the registry does not track P0 raw provenance at all"
    for rel, entry in prov.items():
        if entry["sha256"] is None:
            assert entry["status"] == "MISSING", rel
            assert not (REPO_ROOT / rel).exists(), (
                f"{rel} now EXISTS on disk but is still registered as MISSING. "
                f"Re-run `python scripts/freeze_p0.py --register` so the real "
                f"file is pinned instead of being silently trusted."
            )


def test_restored_provenance_files_must_match_their_registration(registry):
    """Once a raw file is registered, it is frozen like everything else."""
    for rel, entry in registry["p0_raw_provenance"]["files"].items():
        if entry["sha256"] is not None:
            path = REPO_ROOT / rel
            assert path.exists(), f"{rel} was registered but is now absent"
            assert freeze_p0.sha256_text(path) == entry["sha256"], rel


def test_expected_raw_filenames_match_what_evaluate_actually_writes():
    """Guards the naming drift noted in results/raw/README.md.

    evaluate.py writes `{tag}__{alias}__{lang}__{precision}`; the README still
    documents an older `{model}__{precision}__{lang}.csv`. The registry follows
    the code, not the README.
    """
    expected = freeze_p0.expected_p0_raw_files()
    assert len(expected) == 30, "5 languages x 3 precisions x (jsonl + meta)"
    assert any(
        e.endswith("p0__qwen2.5-3b-instruct__eng_Latn__nf4.jsonl") for e in expected)


# --------------------------------------------------------------------------- #
# behaviour, for the files P1 is allowed to extend
# --------------------------------------------------------------------------- #

def test_tidy_schema_is_unchanged():
    """P1 reuses validate_tidy for its BELEBELE arm; the columns must not move."""
    assert schema.TIDY_COLUMNS == (
        "model", "model_revision", "precision", "lang", "item_id",
        "pred", "gold", "correct",
    )
    assert schema.VALID_ANSWERS == (1, 2, 3, 4)


def test_precision_names_are_unchanged():
    assert cfg_mod.VALID_PRECISIONS == ("fp16", "int8_llmint8", "nf4")


def test_p0_scoring_contract_is_unchanged():
    """The scored token, the template, the method: all read by P1 verbatim."""
    cfg = cfg_mod.load()
    assert cfg_mod.require(cfg, "scoring.method") == "letter_logit"
    assert cfg_mod.require(cfg, "scoring.option_letters") == ["A", "B", "C", "D"]
    assert cfg_mod.require(cfg, "scoring.option_prefix") == " "
    assert cfg_mod.require(cfg, "scoring.truncation_side") == "left"
    assert cfg_mod.require(cfg, "scoring.max_input_tokens") == 4096
    template = cfg_mod.require(cfg, "scoring.prompt_template")
    assert template.rstrip("\n").endswith("Answer:")


def test_p0_grid_is_unchanged():
    cfg = cfg_mod.load()
    assert cfg_mod.require(cfg, "benchmark.languages") == [
        "eng_Latn", "ben_Beng", "sin_Sinh", "asm_Beng", "npi_Deva"]
    assert cfg_mod.require(cfg, "precisions") == ["fp16", "int8_llmint8", "nf4"]
    assert cfg_mod.require(cfg, "benchmark.n_items_per_lang") == 900
    assert cfg_mod.require(cfg, "benchmark.reference_language") == "eng_Latn"


def test_p1_did_not_add_a_model():
    """Section 29: no second model. One primary, still Qwen2.5-3B."""
    models = cfg_mod.require(cfg_mod.load(), "models")
    assert len(models) == 1
    assert models[0]["hf_id"] == "Qwen/Qwen2.5-3B-Instruct"
    assert models[0]["revision"] == "aa8e72537993ba99e69dfaafa59ed015b17504d1"


def test_p1_did_not_add_fp8_or_another_precision():
    cfg = cfg_mod.load()
    for name in cfg_mod.require(cfg, "precisions"):
        assert "fp8" not in name.lower()
    assert "fp8" not in str(cfg.get("finetune", {})).lower()


def test_belebele_item_manifest_still_has_900_items():
    m = schema.load_manifest()
    assert m["n_items"] == 900
    assert len(m["item_ids"]) == 900
    assert m["item_id_key"] == ["link", "question_number"]


def test_paired_bootstrap_still_produces_the_same_numbers():
    """P1's primary estimand reuses this function; pin its output numerically.

    A refactor that changed the resampling scheme would move every P0 and P1
    interval at once, and nothing else in the suite would notice.
    """
    rng = np.random.default_rng(7)
    a, b, c, d = (rng.integers(0, 2, size=400) for _ in range(4))
    out = statistics.paired_bootstrap_interaction(a, b, c, d, n_boot=500, seed=11)
    assert out["n_boot"] == 500
    assert out["delta_interaction"] == pytest.approx(
        (a.mean() - b.mean()) - (c.mean() - d.mean()))
    assert out["ci_low"] < out["delta_interaction"] < out["ci_high"]
    repeat = statistics.paired_bootstrap_interaction(a, b, c, d, n_boot=500, seed=11)
    assert repeat == out, "the bootstrap is no longer reproducible from its seed"


def test_holm_is_still_step_down():
    """Sorted p = a .01, c .03, b .04 over m=3 contrasts:
        a -> 3*.01 = .03
        c -> 2*.03 = .06
        b -> 1*.04 = .04, raised to .06 by the running maximum
    The last step is the monotonicity Holm requires; without it an adjusted p
    could fall below one from a smaller raw p.
    """
    adjusted = statistics.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["c"] == pytest.approx(0.06)
    assert adjusted["b"] == pytest.approx(0.06)
    assert adjusted["a"] <= adjusted["c"] <= adjusted["b"]


# --------------------------------------------------------------------------- #
# the guard has to be able to fail
# --------------------------------------------------------------------------- #

def test_guard_detects_a_changed_p0_value(tmp_path, registry):
    """A guard that cannot fail is decoration."""
    import yaml
    cfg = yaml.safe_load(cfg_mod.CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["scoring"]["option_prefix"] = ""      # the ' A' vs 'A' bug, reintroduced
    assert freeze_p0.config_p0_digest(cfg) != registry["config_p0_subtree"]["sha256"]


def test_guard_ignores_a_changed_p1_value(registry):
    """P1 edits must not trip the P0 guard, or nobody will keep it green."""
    import yaml
    cfg = yaml.safe_load(cfg_mod.CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["finetune"]["lora"]["r"] = 999
    assert freeze_p0.config_p0_digest(cfg) == registry["config_p0_subtree"]["sha256"]


def test_guard_detects_an_edited_strict_file(tmp_path):
    original = (REPO_ROOT / "configs" / "item_id_manifest.json").read_text(
        encoding="utf-8")
    tampered = tmp_path / "item_id_manifest.json"
    tampered.write_text(original.replace('"n_items": 900', '"n_items": 899'),
                        encoding="utf-8")
    assert freeze_p0.sha256_text(tampered) != freeze_p0.sha256_text(
        REPO_ROOT / "configs" / "item_id_manifest.json")


def test_digest_is_newline_normalised(tmp_path):
    """This checkout mixes CRLF and LF; a raw-byte digest would be unstable."""
    lf = tmp_path / "lf.txt"
    crlf = tmp_path / "crlf.txt"
    lf.write_bytes(b"alpha\nbeta\n")
    crlf.write_bytes(b"alpha\r\nbeta\r\n")
    assert freeze_p0.sha256_text(lf) == freeze_p0.sha256_text(crlf)
