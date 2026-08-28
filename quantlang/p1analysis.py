"""The final 2 x 2 x 3 grid: Base/FT x English/Bangla x FP16/INT8/NF4.

Nothing in this module modifies P0. `quantlang/statistics.py` and
`quantlang/schema.py` are frozen strict files and are used exactly as they
stand: the fine-tuned arm reaches them through the `model` column, because
`finetune.ft_alias` gives each FT checkpoint its own alias. `Aligned` therefore
pairs Base and FT cells on the canonical BELEBELE item order without knowing
that an "arm" exists.

WHERE THE TWELVE CELLS COME FROM

    Base x {eng, ben} x {fp16, int8, nf4}   ->  the P0 run, already measured
    FT   x {eng, ben} x {fp16, int8, nf4}   ->  the P1 sessions

The Base arm is NOT re-run. It was measured once on the pinned model, the pinned
revision and the frozen 900-item manifest, and re-running it would produce a
second number for a cell that already has one.

THE THREE QUESTIONS, AND WHY TWO OF THEM ARE THE SAME ARITHMETIC

    RQ1  Does quantization cost Bangla more than English?
         d = [fp16(ben) - q(ben)] - [fp16(eng) - q(eng)],  within an arm.

    RQ2  Does fine-tuning change how much quantization costs?
         d = [fp16(base) - q(base)] - [fp16(ft) - q(ft)],  within a language.

Both are differences of differences over item-paired binary vectors, so both use
`paired_bootstrap_interaction` unchanged. It is documented in terms of languages
because that is what P0 needed; the arithmetic does not care which factor is
being crossed, and reusing it keeps every interval in the paper on one estimator.

    RQ3  Does fine-tuning recover what quantization costs?
         A direct paired contrast, FT vs Base at one precision -- NOT a
         difference of differences. It is reported alongside the FP16 baseline
         because "recovered" is meaningless without saying recovered relative to
         what.

WHAT IS DELIBERATELY NOT CONFLATED

Absolute accuracy, quantization degradation, fine-tuning recovery, latency and
memory are five different quantities. A model can lose less to NF4 simply by
being worse in FP16, so degradation is always reported next to the FP16 baseline
it is measured from, and latency never enters an accuracy table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config as cfg_mod
from .finetune import ft_alias
from .schema import SchemaError, load_manifest
from .statistics import (Aligned, holm, mcnemar_exact,
                         paired_bootstrap_interaction, wilson_ci)

ARMS = ("base", "ft")


class P1AnalysisError(RuntimeError):
    """Raised when the artifacts do not support the analysis being asked for."""


# --------------------------------------------------------------------------- #
# arm bookkeeping
# --------------------------------------------------------------------------- #

def arm_of(model_alias: str, base_alias: str, langs: list[str]
           ) -> tuple[str, str | None]:
    """(arm, fine-tuning language) for a result alias.

    The alias is the only place the arm is recorded in the frozen tidy schema,
    which is exactly why `ft_alias` exists and why it is distinct from the base
    alias: an FT cell and a Base cell can never collide in a filename or in a
    groupby.
    """
    if model_alias == base_alias:
        return "base", None
    for lang in langs:
        if model_alias == ft_alias(base_alias, lang):
            return "ft", lang
    raise P1AnalysisError(
        f"result alias {model_alias!r} is neither the base alias "
        f"{base_alias!r} nor ft_alias(base, L) for any L in {langs}. An "
        f"unrecognised alias is an unrecognised experiment; it is not analysed."
    )


# --------------------------------------------------------------------------- #
# verification -- reject, never fill
# --------------------------------------------------------------------------- #

def verify_cell_meta(meta: dict[str, Any], *, cfg: dict[str, Any],
                     base_alias: str, revision: str, langs: list[str],
                     n_expected: int) -> dict[str, Any]:
    """Everything that must be true of one FT cell's manifest.

    Each failure returns a reason rather than raising, so a whole session can be
    reported at once instead of one problem per run of this script. The caller
    refuses to build the grid if any reason survives.
    """
    problems: list[str] = []
    run = meta.get("run_id", "<unknown>")

    if meta.get("arm") != "finetuned":
        problems.append(
            f"arm={meta.get('arm')!r}, expected 'finetuned'. This cell scored "
            f"the base model; it is a duplicate of the Base arm, not an FT cell.")
    if meta.get("weights_from") in (None, "hub"):
        problems.append(
            f"weights_from={meta.get('weights_from')!r}. An FT cell must name "
            f"the merged checkpoint it scored.")
    if meta.get("model_revision") != revision:
        problems.append(
            f"model_revision={meta.get('model_revision')!r}, expected "
            f"{revision!r}. A different base revision is a different experiment.")
    if meta.get("model") != cfg_mod.require(cfg, "models")[0]["hf_id"]:
        problems.append(f"model={meta.get('model')!r} is not the pinned model.")

    lang = meta.get("lang")
    if lang not in langs:
        problems.append(f"lang={lang!r} is outside the final scope {langs}.")
    else:
        expected_alias = ft_alias(base_alias, lang)
        if meta.get("model_alias") != expected_alias:
            problems.append(
                f"model_alias={meta.get('model_alias')!r}, expected "
                f"{expected_alias!r}. The FT checkpoint and the evaluation "
                f"language disagree, which would be a cross-lingual cell -- a "
                f"different experiment from the one declared.")

    precision = meta.get("precision")
    if precision not in cfg_mod.require(cfg, "precisions"):
        problems.append(f"precision={precision!r} is not in the frozen set.")

    if meta.get("n_items") != n_expected:
        problems.append(
            f"n_items={meta.get('n_items')}, expected {n_expected}. A short "
            f"cell is reported as incomplete, never padded or rescaled.")
    if meta.get("limit") is not None:
        problems.append(
            f"limit={meta.get('limit')} -- a limited run cannot form a cell.")
    if meta.get("n_truncated"):
        problems.append(
            f"n_truncated={meta['n_truncated']}: {meta['n_truncated']} prompt(s) "
            f"hit max_input_tokens, so those items were scored on a cut prompt.")

    if meta.get("scoring_method") != cfg_mod.require(cfg, "scoring.method"):
        problems.append(
            f"scoring_method={meta.get('scoring_method')!r} differs from the "
            f"frozen method; results cannot be compared across a change of it.")

    import hashlib
    want_template = hashlib.sha256(
        cfg_mod.require(cfg, "scoring.prompt_template").encode("utf-8")).hexdigest()
    if meta.get("prompt_template_sha256") != want_template:
        problems.append(
            "prompt_template_sha256 differs from the frozen template. The FT "
            "arm was prompted differently from the Base arm.")

    return {"run_id": run, "lang": lang, "precision": precision,
            "model_alias": meta.get("model_alias"),
            "weights_from": meta.get("weights_from"),
            "n_items": meta.get("n_items"),
            "accuracy": meta.get("accuracy"),
            "median_latency_ms": meta.get("median_latency_ms"),
            "peak_memory_reserved_gb": meta.get("peak_memory_reserved_gb"),
            "device": meta.get("device"), "warmup": meta.get("warmup"),
            "repeats": meta.get("repeats"),
            "problems": problems, "ok": not problems}


def load_ft_cells(raw_dir: Path, cfg: dict[str, Any]
                  ) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    """Read every FT cell under `raw_dir`. Returns (records, verified, rejected).

    Cells are routed by their MODEL ALIAS, not by their `arm` field. A file
    whose alias is the base alias is a Base cell and is ignored here (P0 already
    supplies that half of the grid); a file whose alias is `ft_alias(base, L)`
    is an FT cell and is verified, including that its `arm` really does say
    finetuned.

    Routing on `arm` instead would let a mislabelled FT cell disappear rather
    than be rejected -- and the one invalid P1 evaluation on record is exactly
    that shape: it carries the BASE alias because the ad-hoc code that produced
    it scored the base model. Under this rule such a file is skipped as a Base
    duplicate and its FT cell is then reported MISSING, which is the truth.
    """
    raw_dir = Path(raw_dir)
    metas = sorted(raw_dir.rglob("*.meta.json"))
    if not metas:
        raise P1AnalysisError(
            f"no *.meta.json under {raw_dir}. Nothing has been measured; there "
            f"is no FT number to report.")

    models = cfg_mod.require(cfg, "models")
    primary = [m for m in models if m.get("role") == "primary"][0]
    base_alias, revision = primary["alias"], primary["revision"]
    langs = cfg_mod.require(cfg, "finetune.final_scope_languages")
    n_expected = int(cfg_mod.require(cfg, "benchmark.n_items_per_lang"))

    verified: list[dict] = []
    rejected: list[dict] = []
    rows: list[dict] = []
    for meta_path in metas:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        alias = meta.get("model_alias")
        if alias == base_alias:
            continue                      # a Base cell; P0 supplies that half
        if alias not in {ft_alias(base_alias, l) for l in langs}:
            rejected.append({
                "run_id": meta.get("run_id"), "meta_path": str(meta_path),
                "lang": meta.get("lang"), "precision": meta.get("precision"),
                "model_alias": alias, "ok": False,
                "problems": [f"model_alias={alias!r} is neither the base alias "
                             f"nor ft_alias(base, L) for any L in {langs}. An "
                             f"unrecognised alias is an unrecognised experiment."],
            })
            continue
        result = verify_cell_meta(meta, cfg=cfg, base_alias=base_alias,
                                  revision=revision, langs=langs,
                                  n_expected=n_expected)
        result["meta_path"] = str(meta_path)
        if not result["ok"]:
            rejected.append(result)
            continue

        jsonl = meta_path.with_suffix("").with_suffix(".jsonl")
        if not jsonl.exists():
            jsonl = meta_path.parent / (meta_path.name[:-len(".meta.json")] + ".jsonl")
        if not jsonl.exists():
            result["problems"] = [f"per-item file missing next to {meta_path.name}. "
                                  f"A manifest without its items is not admissible."]
            result["ok"] = False
            rejected.append(result)
            continue

        with jsonl.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        verified.append(result)

    return pd.DataFrame(rows), verified, rejected


# --------------------------------------------------------------------------- #
# the grid
# --------------------------------------------------------------------------- #

def build_grid(p0_tidy: pd.DataFrame, ft_records: pd.DataFrame,
               cfg: dict[str, Any]) -> pd.DataFrame:
    """The twelve cells, per item, with `arm` made explicit.

    The Base half comes from P0's frozen tidy.csv rather than from a fresh run.
    The FT half comes from this session's raw output. Both carry the same
    columns the frozen `Aligned` reads, plus `arm` for reporting.
    """
    models = cfg_mod.require(cfg, "models")
    primary = [m for m in models if m.get("role") == "primary"][0]
    base_alias, revision = primary["alias"], primary["revision"]
    langs = list(cfg_mod.require(cfg, "finetune.final_scope_languages"))

    base = p0_tidy[(p0_tidy.model == base_alias) & (p0_tidy.lang.isin(langs))].copy()
    if base.empty:
        raise P1AnalysisError(
            f"P0 tidy holds no rows for {base_alias} in {langs}. The Base arm "
            f"of the grid is the P0 measurement; without it there is nothing to "
            f"compare the FT arm against.")
    bad_rev = set(base.model_revision.unique()) - {revision}
    if bad_rev:
        raise P1AnalysisError(
            f"P0 tidy carries revision(s) {bad_rev}, not the pinned {revision}.")
    base["arm"] = "base"

    ft = pd.DataFrame(columns=base.columns)
    if len(ft_records):
        ft = pd.DataFrame({
            "model": ft_records["model_alias"],
            "model_revision": ft_records["model_revision"],
            "precision": ft_records["precision"],
            "lang": ft_records["lang"],
            "item_id": ft_records["item_id"],
            "pred": ft_records["pred"].astype(int),
            "gold": ft_records["gold"].astype(int),
            "correct": ft_records["correct"].astype(int),
        })
        ft["arm"] = "ft"

    grid = pd.concat([base, ft], ignore_index=True)

    # Gold must agree item-for-item between the arms, or the two are not the
    # same benchmark and no paired comparison is meaningful.
    gold = grid.groupby(["lang", "item_id"])["gold"].nunique()
    if (gold > 1).any():
        offenders = gold[gold > 1].index.tolist()[:3]
        raise P1AnalysisError(
            f"the gold answer differs between arms for {len(gold[gold > 1])} "
            f"item(s), e.g. {offenders}. The arms were not scored on the same "
            f"benchmark.")
    return grid


def grid_completeness(grid: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Which of the twelve cells exist, and at what n. Missing is reported."""
    langs = list(cfg_mod.require(cfg, "finetune.final_scope_languages"))
    precisions = list(cfg_mod.require(cfg, "precisions"))
    n_expected = int(cfg_mod.require(cfg, "benchmark.n_items_per_lang"))

    present, missing, short = [], [], []
    counts = grid.groupby(["arm", "lang", "precision"])["item_id"].nunique()
    for arm in ARMS:
        for lang in langs:
            for precision in precisions:
                key = (arm, lang, precision)
                n = int(counts.get(key, 0))
                if n == 0:
                    missing.append(key)
                elif n != n_expected:
                    short.append((*key, n))
                else:
                    present.append(key)
    return {
        "n_expected_cells": len(ARMS) * len(langs) * len(precisions),
        "n_items_per_cell": n_expected,
        "present": present, "missing": missing, "short": short,
        "complete": not missing and not short,
    }


# --------------------------------------------------------------------------- #
# the analyses
# --------------------------------------------------------------------------- #

def _alias(cfg: dict[str, Any], arm: str, lang: str) -> str:
    primary = [m for m in cfg_mod.require(cfg, "models")
               if m.get("role") == "primary"][0]
    return (primary["alias"] if arm == "base"
            else ft_alias(primary["alias"], lang))


def accuracy_table(grid: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Absolute accuracy with Wilson intervals. One row per cell."""
    rows = []
    for (arm, lang, precision), grp in grid.groupby(["arm", "lang", "precision"]):
        k, n = int(grp["correct"].sum()), len(grp)
        lo, hi = wilson_ci(k, n)
        rows.append({"arm": arm, "lang": lang, "precision": precision,
                     "n": n, "correct": k, "accuracy": k / n,
                     "ci95_low": lo, "ci95_high": hi})
    return pd.DataFrame(rows).sort_values(
        ["arm", "lang", "precision"]).reset_index(drop=True)


def degradation_table(aligned: Aligned, cfg: dict[str, Any],
                      cells: list[tuple[str, str, str]]) -> pd.DataFrame:
    """FP16 -> quantized, within each (arm, language). RQ1's raw material.

    Reported next to the FP16 baseline it is measured from: a model can lose
    less to NF4 by being worse in FP16, and a bare delta hides that.
    """
    langs = list(cfg_mod.require(cfg, "finetune.final_scope_languages"))
    quantized = [p for p in cfg_mod.require(cfg, "precisions") if p != "fp16"]
    have = set(cells)

    rows, pvals = [], {}
    for arm in ARMS:
        for lang in langs:
            alias = _alias(cfg, arm, lang)
            if (alias, lang, "fp16") not in have:
                continue
            fp16 = aligned.get(alias, lang, "fp16")
            for precision in quantized:
                if (alias, lang, precision) not in have:
                    continue
                q = aligned.get(alias, lang, precision)
                key = f"{arm}/{lang}/{precision}"
                p = mcnemar_exact(fp16, q)
                pvals[key] = p
                rows.append({
                    "key": key, "arm": arm, "lang": lang, "precision": precision,
                    "acc_fp16": float(fp16.mean()),
                    "acc_quant": float(q.mean()),
                    "delta_acc": float(fp16.mean() - q.mean()),
                    "n_fp16_only": int(((fp16 == 1) & (q == 0)).sum()),
                    "n_quant_only": int(((fp16 == 0) & (q == 1)).sum()),
                    "mcnemar_p": p,
                })
    if not rows:
        return pd.DataFrame()
    adjusted = holm(pvals)
    df = pd.DataFrame(rows)
    df["mcnemar_p_holm"] = df["key"].map(adjusted)
    return df.drop(columns=["key"]).reset_index(drop=True)


def rq1_language_interaction(aligned: Aligned, cfg: dict[str, Any],
                             cells: list[tuple[str, str, str]]) -> pd.DataFrame:
    """RQ1: does quantization cost Bangla more than English, within an arm?

        d = [fp16(ben) - q(ben)] - [fp16(eng) - q(eng)]

    Positive means the non-reference language pays more. This -- not the raw
    accuracy table -- is the claim the paper's main hypothesis makes.
    """
    ref = cfg_mod.require(cfg, "benchmark.reference_language")
    langs = [l for l in cfg_mod.require(cfg, "finetune.final_scope_languages")
             if l != ref]
    quantized = [p for p in cfg_mod.require(cfg, "precisions") if p != "fp16"]
    stats = cfg_mod.require(cfg, "stats")
    have = set(cells)

    rows, pvals = [], {}
    for arm in ARMS:
        ref_alias = _alias(cfg, arm, ref)
        for lang in langs:
            alias = _alias(cfg, arm, lang)
            for precision in quantized:
                needed = [(ref_alias, ref, "fp16"), (ref_alias, ref, precision),
                          (alias, lang, "fp16"), (alias, lang, precision)]
                if any(c not in have for c in needed):
                    continue
                out = paired_bootstrap_interaction(
                    aligned.get(alias, lang, "fp16"),
                    aligned.get(alias, lang, precision),
                    aligned.get(ref_alias, ref, "fp16"),
                    aligned.get(ref_alias, ref, precision),
                    n_boot=int(stats["bootstrap_iterations"]),
                    seed=int(stats["seed"]),
                    percentiles=tuple(stats["bootstrap_percentiles"]),
                )
                key = f"{arm}/{lang}vs{ref}/{precision}"
                pvals[key] = out["p_bootstrap"]
                rows.append({"key": key, "arm": arm, "lang": lang,
                             "reference": ref, "precision": precision, **out})
    if not rows:
        return pd.DataFrame()
    adjusted = holm(pvals)
    df = pd.DataFrame(rows)
    df["p_holm"] = df["key"].map(adjusted)
    return df.drop(columns=["key"]).reset_index(drop=True)


def rq2_arm_interaction(aligned: Aligned, cfg: dict[str, Any],
                        cells: list[tuple[str, str, str]]) -> pd.DataFrame:
    """RQ2: does fine-tuning change quantization sensitivity, within a language?

        d = [fp16(base) - q(base)] - [fp16(ft) - q(ft)]

    Positive means quantization costs the BASE model more than it costs the
    fine-tuned one -- i.e. fine-tuning made the model more robust to it.

    Same estimator as RQ1: a difference of differences over item-paired binary
    vectors. Only the factor being crossed changes.
    """
    langs = list(cfg_mod.require(cfg, "finetune.final_scope_languages"))
    quantized = [p for p in cfg_mod.require(cfg, "precisions") if p != "fp16"]
    stats = cfg_mod.require(cfg, "stats")
    have = set(cells)

    rows, pvals = [], {}
    for lang in langs:
        base_alias, ft_a = _alias(cfg, "base", lang), _alias(cfg, "ft", lang)
        for precision in quantized:
            needed = [(base_alias, lang, "fp16"), (base_alias, lang, precision),
                      (ft_a, lang, "fp16"), (ft_a, lang, precision)]
            if any(c not in have for c in needed):
                continue
            out = paired_bootstrap_interaction(
                aligned.get(base_alias, lang, "fp16"),
                aligned.get(base_alias, lang, precision),
                aligned.get(ft_a, lang, "fp16"),
                aligned.get(ft_a, lang, precision),
                n_boot=int(stats["bootstrap_iterations"]),
                seed=int(stats["seed"]),
                percentiles=tuple(stats["bootstrap_percentiles"]),
            )
            key = f"{lang}/{precision}"
            pvals[key] = out["p_bootstrap"]
            rows.append({"key": key, "lang": lang, "precision": precision,
                         "contrast": "base_minus_ft_degradation", **out})
    if not rows:
        return pd.DataFrame()
    adjusted = holm(pvals)
    df = pd.DataFrame(rows)
    df["p_holm"] = df["key"].map(adjusted)
    return df.drop(columns=["key"]).reset_index(drop=True)


def rq3_recovery(aligned: Aligned, cfg: dict[str, Any],
                 cells: list[tuple[str, str, str]]) -> pd.DataFrame:
    """RQ3: does fine-tuning recover accuracy lost to quantization?

    A DIRECT paired contrast, FT vs Base at one precision -- not a difference of
    differences. The FP16 baselines of both arms are carried on every row,
    because "recovered" without a reference point is not a statement.

    `recovers_fp16_gap` is the share of the Base FP16 -> Base quantized drop that
    the FT model at that precision closes. It is NaN wherever the base model did
    not actually lose accuracy to quantization, because there is then no gap to
    recover and the ratio is not a quantity.

    Even when it is defined it is a RATIO OF TWO SMALL NUMBERS and goes unstable
    as the denominator shrinks: a 0.6pp base drop under a 0.9pp FT gain prints
    "1.6" and means very little. Read it next to `base_quantization_cost` and
    next to that contrast's McNemar p in the degradation table; where the
    denominator is not distinguishable from zero, the ratio is noise and the
    paper should quote the absolute deltas instead.
    """
    langs = list(cfg_mod.require(cfg, "finetune.final_scope_languages"))
    precisions = list(cfg_mod.require(cfg, "precisions"))
    have = set(cells)

    rows, pvals = [], {}
    for lang in langs:
        base_alias, ft_a = _alias(cfg, "base", lang), _alias(cfg, "ft", lang)
        if (base_alias, lang, "fp16") not in have:
            continue
        base_fp16 = aligned.get(base_alias, lang, "fp16")
        for precision in precisions:
            if any(c not in have for c in [(base_alias, lang, precision),
                                           (ft_a, lang, precision)]):
                continue
            b = aligned.get(base_alias, lang, precision)
            f = aligned.get(ft_a, lang, precision)
            key = f"{lang}/{precision}"
            p = mcnemar_exact(b, f)
            pvals[key] = p
            quant_cost = float(base_fp16.mean() - b.mean())
            gain = float(f.mean() - b.mean())
            rows.append({
                "key": key, "lang": lang, "precision": precision,
                "acc_base": float(b.mean()), "acc_ft": float(f.mean()),
                "delta_ft_minus_base": gain,
                "acc_base_fp16": float(base_fp16.mean()),
                "base_quantization_cost": quant_cost,
                # Undefined unless quantization actually cost the base model
                # something. A negative or zero cost means there is no gap.
                "recovers_fp16_gap": (gain / quant_cost if quant_cost > 0
                                      else float("nan")),
                "n_ft_only": int(((f == 1) & (b == 0)).sum()),
                "n_base_only": int(((f == 0) & (b == 1)).sum()),
                "mcnemar_p": p,
            })
    if not rows:
        return pd.DataFrame()
    adjusted = holm(pvals)
    df = pd.DataFrame(rows)
    df["mcnemar_p_holm"] = df["key"].map(adjusted)
    return df.drop(columns=["key"]).reset_index(drop=True)


def efficiency_table(verified: list[dict], p0_latency: pd.DataFrame | None
                     ) -> pd.DataFrame:
    """Latency and memory, kept out of every accuracy table on purpose.

    These are per-run efficiency metrics with their own comparability rules
    (same session, same GPU, same warmup/repeats protocol), not per-item
    outcomes. Mixing them into a paired accuracy analysis would invite exactly
    the conflation the brief forbids.
    """
    rows = [{"arm": "ft", "lang": v["lang"], "precision": v["precision"],
             "median_latency_ms": v["median_latency_ms"],
             "peak_memory_reserved_gb": v["peak_memory_reserved_gb"],
             "device": v["device"], "warmup": v["warmup"],
             "repeats": v["repeats"]}
            for v in verified]
    if p0_latency is not None and len(p0_latency):
        base = p0_latency.copy()
        base.insert(0, "arm", "base")
        rows_df = pd.DataFrame(rows)
        common = [c for c in rows_df.columns if c in base.columns]
        return pd.concat([base[common], rows_df[common]], ignore_index=True)
    return pd.DataFrame(rows)


def analyse(grid: pd.DataFrame, cfg: dict[str, Any],
            manifest: dict | None = None) -> dict[str, Any]:
    """Run every analysis the grid can support. Missing cells are skipped, loudly."""
    manifest = manifest or load_manifest()
    frozen_cols = ["model", "model_revision", "precision", "lang",
                   "item_id", "pred", "gold", "correct"]
    aligned = Aligned(grid[frozen_cols], manifest)
    cells = list(aligned.keys())
    return {
        "completeness": grid_completeness(grid, cfg),
        "accuracy": accuracy_table(grid, cfg),
        "degradation": degradation_table(aligned, cfg, cells),
        "rq1_language_interaction": rq1_language_interaction(aligned, cfg, cells),
        "rq2_arm_interaction": rq2_arm_interaction(aligned, cfg, cells),
        "rq3_recovery": rq3_recovery(aligned, cfg, cells),
    }
