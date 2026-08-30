"""Test H5 prediction 1: does lower tokenization burden mean less NF4 damage?

Compares a replication model's FP16->quantized degradation with Qwen's, per
language, paired on the same 900 BELEBELE items. Both models see identical
items, so the per-item degradation indicators are comparable; the two models
are different models, so the contrast between them is bootstrapped over items
rather than treated as a within-item paired test.

Hypothesis, predictions, the floor gate and this analysis were fixed in
docs/H5_PREREGISTRATION.md before any replication result existed.

    python scripts/analyze_replication.py --alias bloomz-3b \
        --raw results/results --outdir results/REP_ANALYSIS
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quantlang.config import REPO_ROOT  # noqa: E402

P0_TIDY = REPO_ROOT / "results" / "ALL_P0_RESULTS" / "tables" / "tidy.csv"
CHANCE = 0.25
FLOOR = 0.30          # pre-registered Wilson lower-bound gate


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return centre - half, centre + half


def load_cells(raw: Path, alias: str) -> dict:
    """Every {lang, precision} cell the replication run wrote."""
    out: dict = {}
    for f in sorted(raw.glob("*.jsonl")):
        parts = f.stem.split("__")
        if len(parts) != 4 or parts[1] != alias:
            continue
        _, _, lang, prec = parts
        rows = [json.loads(l) for l in f.open(encoding="utf-8")]
        if len(rows) != 900:
            raise SystemExit(f"FATAL: {f.name} has {len(rows)} items, not 900")
        if {r["arm"] for r in rows} != {"base"}:
            raise SystemExit(f"FATAL: {f.name} is not a base cell")
        df = pd.DataFrame(rows)[["item_id", "correct"]]
        out.setdefault(lang, {})[prec] = df.rename(columns={"correct": prec})
    return out


def mcnemar(df: pd.DataFrame, hi: str, lo: str) -> tuple[int, int, float]:
    b = int(((df[hi] == 1) & (df[lo] == 0)).sum())
    c = int(((df[hi] == 0) & (df[lo] == 1)).sum())
    p = stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p


def holm(pvals: dict) -> dict:
    m = len(pvals)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(sorted(pvals.items(), key=lambda kv: kv[1])):
        running = max(running, min(1.0, p * (m - i)))
        out[k] = running
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--outdir", default="results/REP_ANALYSIS")
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260830)
    args = ap.parse_args()

    cells = load_cells(Path(args.raw), args.alias)
    if not cells:
        raise SystemExit(f"FATAL: no cells for alias {args.alias!r} under {args.raw}")

    tidy = pd.read_csv(P0_TIDY)
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, 900, size=(args.n_boot, 900))

    gate_rows, deg_rows, cross_rows = [], [], []

    # ---- the floor gate, on FP16 alone -------------------------------------
    for lang, precs in sorted(cells.items()):
        if "fp16" not in precs:
            continue
        d = precs["fp16"]
        k = int(d["fp16"].sum())
        lo, hi = wilson(k, 900)
        gate_rows.append(dict(lang=lang, n=900, correct=k, acc=k / 900,
                              ci_low=lo, ci_high=hi, passes_floor=lo > FLOOR))

    floored = {r["lang"] for r in gate_rows if not r["passes_floor"]}

    # ---- degradation within the replication model ---------------------------
    deg_p = {}
    for lang, precs in sorted(cells.items()):
        if lang in floored or "fp16" not in precs:
            continue
        for prec, df in precs.items():
            if prec == "fp16":
                continue
            m = precs["fp16"].merge(df, on="item_id")
            b, c, p = mcnemar(m, "fp16", prec)
            deg_p[(lang, prec)] = p
            deg_rows.append(dict(lang=lang, precision=prec,
                                 delta=(m["fp16"] - m[prec]).mean(),
                                 lost=b, gained=c, mcnemar_p=p))
    adj = holm(deg_p)
    for r in deg_rows:
        r["mcnemar_p_holm"] = adj[(r["lang"], r["precision"])]

    # ---- prediction 1: this model vs Qwen, same items -----------------------
    cross_p = {}
    for lang, precs in sorted(cells.items()):
        if lang in floored or "fp16" not in precs:
            continue
        q = tidy[tidy.lang == lang].pivot(index="item_id", columns="precision",
                                          values="correct").reset_index()
        for prec in [p for p in precs if p != "fp16"]:
            m = q[["item_id", "fp16", prec]].rename(
                columns={"fp16": "Q_fp16", prec: "Q_q"})
            m = m.merge(precs["fp16"], on="item_id").merge(precs[prec], on="item_id")
            if len(m) != 900:
                raise SystemExit(f"FATAL: {lang}/{prec} joined {len(m)} items")
            diff = ((m["fp16"] - m[prec]) - (m["Q_fp16"] - m["Q_q"])).to_numpy(float)
            bs = diff[idx].mean(axis=1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
            cross_p[(lang, prec)] = p
            cross_rows.append(dict(
                lang=lang, precision=prec,
                qwen_degradation=(m["Q_fp16"] - m["Q_q"]).mean(),
                rep_degradation=(m["fp16"] - m[prec]).mean(),
                difference=diff.mean(), ci_low=lo, ci_high=hi, p_bootstrap=p,
                ci_excludes_zero=not (lo <= 0 <= hi)))
    # docs/H5_PREREGISTRATION.md fixes the family as "the three directional
    # predictions" -- NF4, on the non-control languages. English is the control
    # (its burden is unchanged across the two tokenizers) and INT8 is not part
    # of prediction 1, so neither belongs in the primary family.
    #
    # The wider correction across all eight cross-model tests is ALSO reported,
    # because a reader is entitled to see what a more conservative family does
    # to the result, and because choosing the family after seeing the p-values
    # is exactly the move the pre-registration exists to prevent.
    CONTROL = "eng_Latn"
    prereg = {k: v for k, v in cross_p.items()
              if k[1] == "nf4" and k[0] != CONTROL}
    adj_prereg, adj_all = holm(prereg), holm(cross_p)
    for r in cross_rows:
        key = (r["lang"], r["precision"])
        r["in_prereg_family"] = key in prereg
        r["p_holm_prereg"] = adj_prereg.get(key)
        r["p_holm_all_tests"] = adj_all[key]

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("fp16_gate", gate_rows), ("degradation", deg_rows),
                       ("prediction1_vs_qwen", cross_rows)):
        pd.DataFrame(rows).to_csv(out / f"{args.alias}__{name}.csv", index=False)
        print(f"\n=== {name} ===")
        print(pd.DataFrame(rows).to_string(index=False))
    if floored:
        print(f"\nFLOORED (no degradation claim, either direction): {sorted(floored)}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
