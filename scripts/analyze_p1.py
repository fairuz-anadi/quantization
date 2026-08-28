"""Turn the P1 artifacts into the paper's tables and figures.

    python scripts/analyze_p1.py --p1-raw results/raw --outdir results/P1
    python scripts/analyze_p1.py --p1-raw results/raw --outdir results/P1 --strict

Reads two things and invents neither:

  * the BASE arm from P0's frozen `tidy.csv`, which was measured once on the
    pinned model, the pinned revision and the frozen 900-item manifest. It is
    not re-run: a cell that already has a number does not get a second one.
  * the FT arm from the raw per-item output of the P1 Kaggle sessions.

Every FT cell is verified before it is allowed into the grid -- model, revision,
language, precision, item count, truncation, scoring method, prompt template,
and that it actually scored a merged checkpoint rather than the base model.
A cell that fails is REJECTED and reported as missing. It is never repaired,
rescaled or filled in, and `--strict` makes an incomplete grid a non-zero exit
so this cannot be run past by accident in a notebook.

Nothing here writes to results/raw/, which is append-only and populated solely
by `kaggle kernels output`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from quantlang import config as cfg_mod  # noqa: E402
from quantlang import p1analysis  # noqa: E402
from quantlang.config import REPO_ROOT  # noqa: E402

P0_TIDY = REPO_ROOT / "results" / "ALL_P0_RESULTS" / "tables" / "tidy.csv"
P0_LATENCY = REPO_ROOT / "results" / "ALL_P0_RESULTS" / "tables" / "latency.csv"


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(Path(p).resolve())


def figures(res: dict, outdir: Path) -> None:
    """Three figures, each answering exactly one of the three questions."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib unavailable; figures skipped")
        return

    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    acc = res["accuracy"]
    if acc.empty:
        return
    precisions = list(dict.fromkeys(acc["precision"]))
    langs = list(dict.fromkeys(acc["lang"]))
    x = range(len(precisions))

    # fig1 -- absolute accuracy. Base and FT side by side, per language.
    fig, axes = plt.subplots(1, len(langs), figsize=(4.2 * len(langs), 4),
                             sharey=True)
    axes = axes if len(langs) > 1 else [axes]
    for ax, lang in zip(axes, langs):
        for arm, marker in (("base", "o"), ("ft", "s")):
            sub = acc[(acc.lang == lang) & (acc.arm == arm)].set_index("precision")
            if sub.empty:
                continue
            ys = [sub.loc[p, "accuracy"] for p in precisions if p in sub.index]
            lo = [sub.loc[p, "accuracy"] - sub.loc[p, "ci95_low"]
                  for p in precisions if p in sub.index]
            hi = [sub.loc[p, "ci95_high"] - sub.loc[p, "accuracy"]
                  for p in precisions if p in sub.index]
            ax.errorbar(list(x)[:len(ys)], ys, yerr=[lo, hi], marker=marker,
                        capsize=3, label=arm)
        ax.axhline(0.25, ls=":", lw=0.8, color="grey")
        ax.set_xticks(list(x)); ax.set_xticklabels(precisions, rotation=15)
        ax.set_title(lang); ax.set_xlabel("precision")
    axes[0].set_ylabel("accuracy (95% Wilson CI)")
    axes[0].legend(title="arm", fontsize=8)
    fig.suptitle("Absolute accuracy. Dotted line is the 0.25 chance floor.",
                 fontsize=9)
    fig.tight_layout(); fig.savefig(figdir / "fig1_accuracy_grid.pdf")
    plt.close(fig)

    # fig2 -- degradation from each arm's OWN fp16, which is the only baseline
    # a degradation is meaningful against.
    deg = res["degradation"]
    if not deg.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4))
        labels, vals = [], []
        for _, r in deg.sort_values(["lang", "arm", "precision"]).iterrows():
            labels.append(f"{r['lang']}\n{r['arm']}/{r['precision']}")
            vals.append(r["delta_acc"])
        ax.bar(range(len(vals)), vals)
        ax.axhline(0, lw=0.8, color="black")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("accuracy lost vs that arm's own FP16")
        ax.set_title("Quantization degradation, measured within each arm")
        fig.tight_layout(); fig.savefig(figdir / "fig2_degradation.pdf")
        plt.close(fig)

    # fig3 -- the two interaction terms with their bootstrap intervals. This is
    # the figure the hypotheses actually live in.
    rows = []
    for name, df in (("RQ1 lang", res["rq1_language_interaction"]),
                     ("RQ2 arm", res["rq2_arm_interaction"])):
        for _, r in df.iterrows():
            tag = r.get("arm", "") or ""
            rows.append((f"{name} {r['lang']}/{r['precision']} {tag}".strip(),
                         r["delta_interaction"], r["ci_low"], r["ci_high"]))
    if rows:
        fig, ax = plt.subplots(figsize=(7.2, 0.45 * len(rows) + 1.8))
        ys = range(len(rows))
        ax.errorbar([r[1] for r in rows], list(ys),
                    xerr=[[r[1] - r[2] for r in rows],
                          [r[3] - r[1] for r in rows]],
                    fmt="o", capsize=3)
        ax.axvline(0, lw=0.8, color="black")
        ax.set_yticks(list(ys)); ax.set_yticklabels([r[0] for r in rows], fontsize=7)
        ax.set_xlabel("interaction (difference of differences), 95% bootstrap CI")
        ax.set_title("An interval crossing zero is not evidence of an effect")
        fig.tight_layout(); fig.savefig(figdir / "fig3_interactions.pdf")
        plt.close(fig)
    print(f"  figures -> {_rel(figdir)}")


def summary(res: dict, verified: list[dict], rejected: list[dict]) -> str:
    """A plain-language reading of the tables, with no claim the data cannot bear."""
    out: list[str] = []
    comp = res["completeness"]
    out.append("=== GRID ===")
    out.append(f"{len(comp['present'])}/{comp['n_expected_cells']} cells complete "
               f"at {comp['n_items_per_cell']} items each")
    for key in comp["missing"]:
        out.append(f"  MISSING  {'/'.join(key)}")
    for *key, n in comp["short"]:
        out.append(f"  SHORT    {'/'.join(key)}  ({n} items) -- rejected, not padded")
    for r in rejected:
        out.append(f"  REJECTED {r.get('run_id')}: {r['problems'][0]}")

    out.append("\n=== RQ1: does quantization cost the two languages differently? ===")
    df = res["rq1_language_interaction"]
    if df.empty:
        out.append("  not computable: the grid is missing cells this needs")
    for _, r in df.iterrows():
        crosses = r["ci_low"] <= 0 <= r["ci_high"]
        out.append(
            f"  {r['arm']:<4} {r['lang']} vs {r['reference']} / {r['precision']:<13} "
            f"d={r['delta_interaction']:+.4f} "
            f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] "
            f"p_holm={r['p_holm']:.4f}"
            + ("  -- interval crosses zero" if crosses else "  -- interval excludes zero"))

    out.append("\n=== RQ2: does fine-tuning change quantization sensitivity? ===")
    df = res["rq2_arm_interaction"]
    if df.empty:
        out.append("  not computable: the grid is missing cells this needs")
    for _, r in df.iterrows():
        crosses = r["ci_low"] <= 0 <= r["ci_high"]
        out.append(
            f"  {r['lang']} / {r['precision']:<13} "
            f"d(base-ft degradation)={r['delta_interaction']:+.4f} "
            f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] "
            f"p_holm={r['p_holm']:.4f}"
            + ("  -- interval crosses zero" if crosses else "  -- interval excludes zero"))

    out.append("\n=== RQ3: does fine-tuning recover what quantization costs? ===")
    df = res["rq3_recovery"]
    if df.empty:
        out.append("  not computable: the grid is missing cells this needs")
    for _, r in df.iterrows():
        ratio = ("n/a" if pd.isna(r["recovers_fp16_gap"])
                 else f"{r['recovers_fp16_gap']:.2f}x")
        out.append(
            f"  {r['lang']} / {r['precision']:<13} "
            f"base={r['acc_base']:.4f} ft={r['acc_ft']:.4f} "
            f"delta={r['delta_ft_minus_base']:+.4f} "
            f"(base fp16 {r['acc_base_fp16']:.4f}, quantization cost "
            f"{r['base_quantization_cost']:+.4f}, closes {ratio}) "
            f"McNemar p_holm={r['mcnemar_p_holm']:.4f}")

    out.append(
        "\nRead with care. An interval that crosses zero is not evidence of no "
        "effect and is certainly not evidence of one; the recovery ratio is a "
        "ratio of two small numbers and is unstable wherever the base model "
        "lost little to quantization. This is one model, one benchmark and two "
        "languages -- it is a focused comparison, not a claim about LLMs in "
        "general.")
    out.append(
        "\nOn the Base-arm RQ1 rows: their point estimates and bootstrap "
        "intervals are IDENTICAL to P0's published interaction table, because "
        "they are the same measurement on the same items. Their Holm-adjusted "
        "p-values are not, and should not be: P0 corrected across four "
        "languages, this analysis corrects across two arms. The family changed, "
        "so the adjustment changed. Quote one table or the other, and say which.")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1-raw", required=True,
                    help="directory of FT *.jsonl / *.meta.json from Kaggle")
    ap.add_argument("--outdir", default=str(REPO_ROOT / "results" / "P1"))
    ap.add_argument("--p0-tidy", default=str(P0_TIDY))
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero unless all twelve cells are complete")
    args = ap.parse_args()

    cfg = cfg_mod.load()
    outdir = Path(args.outdir)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)

    p0 = pd.read_csv(args.p0_tidy)
    records, verified, rejected = p1analysis.load_ft_cells(Path(args.p1_raw), cfg)

    print(f"FT cells accepted : {len(verified)}")
    for r in rejected:
        print(f"FT cell REJECTED  : {r.get('run_id')} ({_rel(Path(r['meta_path']))})")
        for problem in r["problems"]:
            print(f"    - {problem}")

    grid = p1analysis.build_grid(p0, records, cfg)
    res = p1analysis.analyse(grid, cfg)

    grid.to_csv(outdir / "tables" / "grid_tidy.csv", index=False)
    for name in ("accuracy", "degradation", "rq1_language_interaction",
                 "rq2_arm_interaction", "rq3_recovery"):
        res[name].to_csv(outdir / "tables" / f"{name}.csv", index=False)

    latency = None
    if Path(P0_LATENCY).exists():
        latency = pd.read_csv(P0_LATENCY)
        latency = latency[latency.lang.isin(
            cfg_mod.require(cfg, "finetune.final_scope_languages"))]
    p1analysis.efficiency_table(verified, latency).to_csv(
        outdir / "tables" / "efficiency.csv", index=False)

    text = summary(res, verified, rejected)
    (outdir / "tables" / "summary.txt").write_text(text + "\n", encoding="utf-8")
    (outdir / "tables" / "provenance.json").write_text(json.dumps({
        "verified_cells": verified, "rejected_cells": rejected,
        "completeness": {k: v for k, v in res["completeness"].items()
                         if k != "present"},
        "p0_tidy": _rel(Path(args.p0_tidy)),
        "p1_raw": _rel(Path(args.p1_raw)),
    }, indent=2, default=str), encoding="utf-8")

    figures(res, outdir)
    print(f"\n{text}\n")
    print(f"wrote {_rel(outdir / 'tables')}")

    if not res["completeness"]["complete"]:
        msg = ("\nThe grid is INCOMPLETE. The missing cells above are reported "
               "as missing and were not filled in.")
        if args.strict:
            print(msg + " --strict: failing.", file=sys.stderr)
            return 1
        print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
