"""The final 2 x 2 x 3 grid, and the verification that gates entry to it.

The point of this file is the REJECTION paths. Section 21 of the brief asks for
completeness, model, revision, language, precision, item count, truncation and
provenance to be verified, and for an incomplete cell to be rejected rather than
silently filled. Every one of those has a test that feeds it a bad cell and
checks it is refused.

The FT cells here are SYNTHETIC. They exercise the plumbing and are built to a
fixed seed so the assertions are about structure, never about an effect size. No
number produced in this file is a result, and nothing here writes outside tmp.

The Base arm is real: it comes from P0's frozen tidy.csv, which is what the
analysis will use for those six cells.
"""

import hashlib
import json
import random

import pandas as pd
import pytest

from quantlang import config as cfg_mod
from quantlang import p1analysis
from quantlang.config import REPO_ROOT
from quantlang.finetune import ft_alias
from quantlang.p1analysis import P1AnalysisError
from quantlang.schema import load_manifest

P0_TIDY = REPO_ROOT / "results" / "ALL_P0_RESULTS" / "tables" / "tidy.csv"


@pytest.fixture(scope="module")
def cfg():
    return cfg_mod.load()


@pytest.fixture(scope="module")
def manifest():
    return load_manifest()


@pytest.fixture(scope="module")
def p0_tidy():
    return pd.read_csv(P0_TIDY)


def _primary(cfg):
    return [m for m in cfg_mod.require(cfg, "models")
            if m.get("role") == "primary"][0]


def _meta(cfg, lang, precision, overrides=None):
    """One cell manifest in exactly evaluate_cell's shape.

    Overrides go through a dict rather than **kwargs so a test can override
    `precision` or `lang` -- which are also positional here -- without a name
    collision. `alias` is the one key that is not a manifest field: it stands in
    for model_alias in both the run_id and the field.
    """
    overrides = dict(overrides or {})
    alias = overrides.pop("alias", None)
    prim = _primary(cfg)
    meta = {
        "run_id": f"main__{alias or ft_alias(prim['alias'], lang)}__{lang}__{precision}",
        "tag": "main",
        "model": prim["hf_id"],
        "weights_from": f"/kaggle/working/p1/merged/{lang}__seed20260828",
        "arm": "finetuned",
        "model_alias": alias or ft_alias(prim["alias"], lang),
        "model_revision": prim["revision"],
        "precision": precision,
        "lang": lang,
        "n_items": 900,
        "n_correct": 600,
        "accuracy": 600 / 900,
        "n_truncated": 0,
        "median_latency_ms": 42.0,
        "peak_memory_reserved_gb": 6.2,
        "scoring_method": cfg_mod.require(cfg, "scoring.method"),
        "prompt_template_sha256": hashlib.sha256(
            cfg_mod.require(cfg, "scoring.prompt_template").encode("utf-8")
        ).hexdigest(),
        "warmup": 5, "repeats": 1, "device": "cuda:0", "limit": None,
    }
    meta.update(overrides)
    return meta


def _write_cell(outdir, cfg, manifest, lang, precision, accuracy=0.66, seed=7,
                n_items=None, overrides=None):
    """One synthetic FT cell in exactly evaluate_cell's output format."""
    overrides = dict(overrides or {})
    meta = _meta(cfg, lang, precision, overrides)
    run = meta["run_id"]
    gold_by_id = manifest["gold_by_item_id"]
    ids = list(manifest["item_ids"])[: n_items or len(manifest["item_ids"])]
    rng = random.Random(seed)
    n_correct = 0
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / f"{run}.jsonl").open("w", encoding="utf-8") as fh:
        for iid in ids:
            gold = int(gold_by_id[iid])
            ok = rng.random() < accuracy
            pred = gold if ok else rng.choice([v for v in (1, 2, 3, 4) if v != gold])
            n_correct += int(ok)
            fh.write(json.dumps({
                "run_id": run, "model": meta["model"],
                "weights_from": meta["weights_from"], "arm": meta["arm"],
                "model_alias": meta["model_alias"],
                "model_revision": meta["model_revision"],
                "precision": precision, "lang": lang, "item_id": iid,
                "pred": pred, "gold": gold, "correct": int(ok),
                "letter_logits": [0.1, 0.2, 0.3, 0.4], "latency_ms": 42.0,
                "input_tokens": 500, "truncated": False, "prompt_sha256": "x",
            }) + "\n")
    if "n_items" not in overrides:
        meta["n_items"] = len(ids)
    meta["n_correct"] = n_correct
    meta["accuracy"] = n_correct / len(ids)
    (outdir / f"{run}.meta.json").write_text(
        json.dumps(meta, indent=1), encoding="utf-8")
    return meta


@pytest.fixture
def ft_dir(tmp_path, cfg, manifest):
    """A complete, valid FT arm: two languages x three precisions."""
    out = tmp_path / "raw"
    for lang, acc in (("eng_Latn", 0.90), ("ben_Beng", 0.65)):
        for precision in cfg_mod.require(cfg, "precisions"):
            _write_cell(out, cfg, manifest, lang, precision, accuracy=acc)
    return out


# --------------------------------------------------------------------------- #
# the arm is recoverable from the alias, which is the only place it lives
# --------------------------------------------------------------------------- #

def test_arm_is_derived_from_the_alias(cfg):
    langs = cfg_mod.require(cfg, "finetune.final_scope_languages")
    base = _primary(cfg)["alias"]
    assert p1analysis.arm_of(base, base, langs) == ("base", None)
    for lang in langs:
        assert p1analysis.arm_of(ft_alias(base, lang), base, langs) == ("ft", lang)


def test_an_unknown_alias_is_not_analysed(cfg):
    langs = cfg_mod.require(cfg, "finetune.final_scope_languages")
    with pytest.raises(P1AnalysisError, match="unrecognised experiment"):
        p1analysis.arm_of("some-other-model", _primary(cfg)["alias"], langs)


# --------------------------------------------------------------------------- #
# section 21: verify, and REJECT rather than fill
# --------------------------------------------------------------------------- #

def test_a_clean_cell_is_accepted(cfg, manifest, ft_dir):
    records, verified, rejected = p1analysis.load_ft_cells(ft_dir, cfg)
    assert len(verified) == 6, [r["problems"] for r in rejected]
    assert not rejected
    assert len(records) == 6 * 900


@pytest.mark.parametrize("override, expected", [
    ({"arm": "base"}, "scored the base model"),
    ({"weights_from": "hub"}, "must name the merged checkpoint"),
    ({"model_revision": "0" * 40}, "different base revision"),
    ({"model": "meta-llama/Llama-3-8B"}, "not the pinned model"),
    ({"precision": "fp8"}, "not in the frozen set"),
    ({"n_items": 450}, "never padded or rescaled"),
    ({"limit": 20}, "limited run cannot form a cell"),
    ({"n_truncated": 3}, "cut prompt"),
    ({"scoring_method": "option_loglik"}, "frozen method"),
    ({"prompt_template_sha256": "deadbeef"}, "prompted differently"),
])
def test_a_bad_cell_is_rejected(tmp_path, cfg, manifest, override, expected):
    """One test per item on the brief's verification list."""
    out = tmp_path / "raw"
    _write_cell(out, cfg, manifest, "eng_Latn", "fp16", overrides=override)
    _, verified, rejected = p1analysis.load_ft_cells(out, cfg)
    assert not verified, "a bad cell must not be accepted"
    assert len(rejected) == 1
    assert any(expected in p for p in rejected[0]["problems"]), \
        rejected[0]["problems"]


def test_a_cross_lingual_cell_is_rejected(tmp_path, cfg, manifest):
    """An eng checkpoint scored on ben would be a different experiment."""
    out = tmp_path / "raw"
    _write_cell(out, cfg, manifest, "ben_Beng", "fp16",
                overrides={"alias": ft_alias(_primary(cfg)["alias"], "eng_Latn")})
    _, verified, rejected = p1analysis.load_ft_cells(out, cfg)
    assert not verified
    assert any("cross-lingual" in p for p in rejected[0]["problems"])


def test_a_manifest_without_its_items_is_rejected(tmp_path, cfg, manifest):
    """A meta.json is a claim; the per-item file is the evidence."""
    out = tmp_path / "raw"
    meta = _write_cell(out, cfg, manifest, "eng_Latn", "fp16")
    (out / f"{meta['run_id']}.jsonl").unlink()
    _, verified, rejected = p1analysis.load_ft_cells(out, cfg)
    assert not verified
    assert any("without its items" in p for p in rejected[0]["problems"])


def test_base_arm_files_in_the_same_directory_are_ignored(tmp_path, cfg, manifest):
    """A Base cell sitting beside the FT cells must not be double-counted."""
    out = tmp_path / "raw"
    _write_cell(out, cfg, manifest, "eng_Latn", "fp16")
    _write_cell(out, cfg, manifest, "eng_Latn", "fp16",
                overrides={"alias": _primary(cfg)["alias"], "arm": "base",
                           "weights_from": "hub"})
    _, verified, rejected = p1analysis.load_ft_cells(out, cfg)
    assert len(verified) == 1
    assert not rejected


def test_an_empty_directory_is_not_silently_an_empty_grid(tmp_path, cfg):
    (tmp_path / "raw").mkdir()
    with pytest.raises(P1AnalysisError, match="no FT number to report"):
        p1analysis.load_ft_cells(tmp_path / "raw", cfg)


# --------------------------------------------------------------------------- #
# the grid
# --------------------------------------------------------------------------- #

def test_the_base_arm_comes_from_p0_and_is_not_re_run(cfg, p0_tidy, ft_dir):
    """Six of the twelve cells are already measured; re-running them would give
    a cell that has one number a second one."""
    records, *_ = p1analysis.load_ft_cells(ft_dir, cfg)
    grid = p1analysis.build_grid(p0_tidy, records, cfg)
    base = grid[grid.arm == "base"]
    assert set(base.model.unique()) == {_primary(cfg)["alias"]}
    assert set(base.lang.unique()) == set(
        cfg_mod.require(cfg, "finetune.final_scope_languages"))
    assert len(base) == 6 * 900


def test_the_grid_has_exactly_twelve_complete_cells(cfg, p0_tidy, ft_dir):
    records, *_ = p1analysis.load_ft_cells(ft_dir, cfg)
    grid = p1analysis.build_grid(p0_tidy, records, cfg)
    comp = p1analysis.grid_completeness(grid, cfg)
    assert comp["n_expected_cells"] == 12
    assert comp["complete"]
    assert len(comp["present"]) == 12
    assert not comp["missing"] and not comp["short"]


def test_a_missing_ft_cell_is_reported_not_filled(tmp_path, cfg, manifest, p0_tidy):
    out = tmp_path / "raw"
    for precision in cfg_mod.require(cfg, "precisions"):
        _write_cell(out, cfg, manifest, "eng_Latn", precision)
    _write_cell(out, cfg, manifest, "ben_Beng", "fp16")     # ben int8/nf4 absent
    records, *_ = p1analysis.load_ft_cells(out, cfg)
    comp = p1analysis.grid_completeness(
        p1analysis.build_grid(p0_tidy, records, cfg), cfg)
    assert not comp["complete"]
    assert ("ft", "ben_Beng", "nf4") in comp["missing"]
    assert len(comp["present"]) == 10


def test_disagreeing_gold_between_arms_is_fatal(cfg, p0_tidy, ft_dir):
    """If the two arms disagree on the answer key they are not the same
    benchmark, and no paired comparison between them means anything."""
    records, *_ = p1analysis.load_ft_cells(ft_dir, cfg)
    records = records.copy()
    records.loc[records.index[0], "gold"] = 1 + (int(records.loc[records.index[0],
                                                                "gold"]) % 4)
    with pytest.raises(P1AnalysisError, match="not scored on the same"):
        p1analysis.build_grid(p0_tidy, records, cfg)


def test_a_base_arm_at_the_wrong_revision_is_fatal(cfg, p0_tidy, ft_dir):
    records, *_ = p1analysis.load_ft_cells(ft_dir, cfg)
    tampered = p0_tidy.copy()
    tampered["model_revision"] = "0" * 40
    with pytest.raises(P1AnalysisError, match="not the pinned"):
        p1analysis.build_grid(tampered, records, cfg)


# --------------------------------------------------------------------------- #
# the analyses run on frozen statistics, unmodified
# --------------------------------------------------------------------------- #

@pytest.fixture
def analysed(cfg, p0_tidy, ft_dir, manifest):
    records, *_ = p1analysis.load_ft_cells(ft_dir, cfg)
    grid = p1analysis.build_grid(p0_tidy, records, cfg)
    return p1analysis.analyse(grid, cfg, manifest)


def test_accuracy_covers_every_cell_with_an_interval(analysed):
    acc = analysed["accuracy"]
    assert len(acc) == 12
    assert (acc.n == 900).all()
    assert (acc.ci95_low <= acc.accuracy).all()
    assert (acc.accuracy <= acc.ci95_high).all()


def test_the_base_arm_accuracy_matches_p0_exactly(analysed, p0_tidy, cfg):
    """The analysis must reproduce the published P0 numbers, not re-derive them."""
    acc = analysed["accuracy"]
    for lang in cfg_mod.require(cfg, "finetune.final_scope_languages"):
        for precision in cfg_mod.require(cfg, "precisions"):
            got = acc[(acc.arm == "base") & (acc.lang == lang)
                      & (acc.precision == precision)]["accuracy"].iloc[0]
            sub = p0_tidy[(p0_tidy.lang == lang)
                          & (p0_tidy.precision == precision)]
            assert got == pytest.approx(sub["correct"].mean())


def test_degradation_is_measured_against_the_same_arms_fp16(analysed):
    """A degradation against the OTHER arm's baseline would be meaningless."""
    deg = analysed["degradation"]
    acc = analysed["accuracy"].set_index(["arm", "lang", "precision"])
    for _, r in deg.iterrows():
        own_fp16 = acc.loc[(r["arm"], r["lang"], "fp16"), "accuracy"]
        assert r["acc_fp16"] == pytest.approx(own_fp16)
        assert r["delta_acc"] == pytest.approx(r["acc_fp16"] - r["acc_quant"])


def test_rq1_base_arm_reproduces_p0s_published_interaction(analysed):
    """The Base half of RQ1 is P0's measurement and must come out identical.

    Same items, same estimator, same seed -- so the point estimate and the
    bootstrap interval must match `interaction.csv` exactly. Holm-adjusted p is
    deliberately NOT compared: P0 corrected across four languages and this
    analysis corrects across two arms, so the family differs and the adjustment
    must differ with it.
    """
    published = pd.read_csv(
        REPO_ROOT / "results" / "ALL_P0_RESULTS" / "tables" / "interaction.csv")
    mine = analysed["rq1_language_interaction"]
    mine = mine[mine.arm == "base"]
    assert len(mine) > 0
    for _, r in mine.iterrows():
        ref = published[(published.lang == r["lang"])
                        & (published.precision == r["precision"])]
        assert len(ref) == 1, (r["lang"], r["precision"])
        ref = ref.iloc[0]
        assert r["delta_interaction"] == pytest.approx(ref["delta_interaction"])
        assert r["ci_low"] == pytest.approx(ref["ci_low"])
        assert r["ci_high"] == pytest.approx(ref["ci_high"])
        assert r["p_bootstrap"] == pytest.approx(ref["p_bootstrap"])


def test_rq1_is_computed_within_each_arm(analysed, cfg):
    """The language contrast is meaningless pooled across arms."""
    df = analysed["rq1_language_interaction"]
    assert set(df.arm.unique()) == {"base", "ft"}
    ref = cfg_mod.require(cfg, "benchmark.reference_language")
    assert (df.reference == ref).all()
    assert (df.lang != ref).all()


def test_rq2_is_computed_within_each_language(analysed, cfg):
    df = analysed["rq2_arm_interaction"]
    assert set(df.lang.unique()) == set(
        cfg_mod.require(cfg, "finetune.final_scope_languages"))
    assert "fp16" not in set(df.precision.unique()), (
        "fp16 vs fp16 is not a degradation contrast")


def test_rq3_carries_the_fp16_baseline_on_every_row(analysed):
    """'Recovered' is not a statement without saying recovered relative to what."""
    df = analysed["rq3_recovery"]
    assert {"acc_base_fp16", "base_quantization_cost"} <= set(df.columns)
    assert df["acc_base_fp16"].notna().all()


def test_the_recovery_ratio_is_undefined_when_there_is_no_gap(analysed):
    """A ratio over a zero or negative denominator is not a quantity."""
    df = analysed["rq3_recovery"]
    no_gap = df[df.base_quantization_cost <= 0]
    assert no_gap["recovers_fp16_gap"].isna().all()
    assert df[df.precision == "fp16"]["recovers_fp16_gap"].isna().all()


def test_every_family_is_holm_corrected(analysed):
    for name, col in (("degradation", "mcnemar_p_holm"),
                      ("rq1_language_interaction", "p_holm"),
                      ("rq2_arm_interaction", "p_holm"),
                      ("rq3_recovery", "mcnemar_p_holm")):
        df = analysed[name]
        assert col in df.columns, name
        raw = "mcnemar_p" if "mcnemar" in col else "p_bootstrap"
        assert (df[col] >= df[raw] - 1e-12).all(), name


def test_incomplete_grids_still_analyse_what_they_can(tmp_path, cfg, manifest,
                                                      p0_tidy):
    """A half-finished session must yield the contrasts it supports and no more."""
    out = tmp_path / "raw"
    for precision in cfg_mod.require(cfg, "precisions"):
        _write_cell(out, cfg, manifest, "eng_Latn", precision)
    records, *_ = p1analysis.load_ft_cells(out, cfg)
    res = p1analysis.analyse(p1analysis.build_grid(p0_tidy, records, cfg), cfg)
    assert not res["completeness"]["complete"]
    # Base-arm RQ1 needs no FT cell at all, so it is still available...
    assert "base" in set(res["rq1_language_interaction"].arm.unique())
    # ...but the FT-arm language contrast needs Bangla FT, which is absent.
    assert "ft" not in set(res["rq1_language_interaction"].arm.unique())
    # RQ2/RQ3 for English are computable; for Bangla they are not.
    assert set(res["rq2_arm_interaction"].lang.unique()) == {"eng_Latn"}
    assert set(res["rq3_recovery"].lang.unique()) == {"eng_Latn"}


def test_no_p0_file_is_written_by_the_analysis(analysed):
    """P0's tables are strict-frozen; the P1 analysis reads and never writes."""
    import inspect
    src = inspect.getsource(p1analysis)
    assert "to_csv" not in src, (
        "p1analysis must not write files; the driver script owns output paths")
