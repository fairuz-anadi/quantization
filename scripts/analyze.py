"""Every number and figure in the paper is produced here, from tidy.csv.

    python scripts/analyze.py

Outputs to results/tables/ and results/figures/:
    accuracy.csv      Table 2  accuracy by language x precision, Wilson 95% CI
    degradation.csv   Table 3  dAcc = Acc(FP16) - Acc(quantized), exact McNemar
    interaction.csv   H1/H2    d_int vs the reference language, bootstrap CI
    latency.csv       Table 4  (written by build_tidy.py)
    summary.txt       the same thing, readable
    fig1..fig3        accuracy / degradation / latency

Nothing here is typed by hand into the paper, and nothing here fills a gap: a
cell that was not measured is absent from the output, not estimated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantlang import config as cfg_mod  # noqa: E402
from quantlang.config import REPO_ROOT  # noqa: E402
from quantlang.statistics import (  # noqa: E402
    Aligned, holm, mcnemar_exact, paired_bootstrap_interaction, wilson_ci,
)

LANG_NAME = {
    "eng_Latn": "English", "ben_Beng": "Bangla", "sin_Sinh": "Sinhala",
    "asm_Beng": "Assamese", "npi_Deva": "Nepali",
}
QUANTIZED = ("int8_llmint8", "nf4")
PREC_LABEL = {"fp16": "FP16", "int8_llmint8": "INT8", "nf4": "NF4"}


def _rel(p: Path) -> str:
    """Display path relative to the repo when possible; absolute otherwise."""
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(Path(p).resolve())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tidy", default=str(REPO_ROOT / "results" / "tables" / "tidy.csv"))
    ap.add_argument("--outdir", default=str(REPO_ROOT / "results" / "tables"))
    ap.add_argument("--figdir", default=str(REPO_ROOT / "results" / "figures"))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    cfg = cfg_mod.load()
    ref = cfg_mod.require(cfg, "benchmark.reference_language")
    n_boot = cfg_mod.require(cfg, "stats.bootstrap_iterations")
    seed = cfg_mod.require(cfg, "stats.seed")
    pcts = tuple(cfg_mod.require(cfg, "stats.bootstrap_percentiles"))
    correction = cfg_mod.require(cfg, "stats.multiple_comparison_correction")
    if correction != "holm":
        raise SystemExit(f"FATAL: unsupported correction {correction!r}")

    tidy_path = Path(args.tidy)
    if not tidy_path.exists():
        raise SystemExit(
            f"FATAL: {tidy_path} does not exist. Run scripts/build_tidy.py "
            f"first; there is nothing measured to analyse."
        )
    tidy = pd.read_csv(tidy_path)
    aligned = Aligned(tidy)

    outdir, figdir = Path(args.outdir), Path(args.figdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    models = sorted(tidy["model"].unique())
    langs_all = sorted(tidy["lang"].unique())
    lines: list[str] = []
    acc_rows, deg_rows, int_rows = [], [], []

    for model in models:
        langs = [l for l in langs_all if any(
            (model, l, p) in aligned for p in ("fp16",) + QUANTIZED)]

        # ---- Table 2: accuracy with Wilson CI ----------------------------- #
        for lang in langs:
            for prec in ("fp16",) + QUANTIZED:
                if (model, lang, prec) not in aligned:
                    continue
                v = aligned.get(model, lang, prec)
                k, n = int(v.sum()), len(v)
                lo, hi = wilson_ci(k, n)
                acc_rows.append({
                    "model": model, "lang": lang, "precision": prec,
                    "n": n, "correct": k, "accuracy": k / n,
                    "ci95_low": lo, "ci95_high": hi,
                })

        # ---- Table 3: degradation + exact McNemar ------------------------- #
        raw_p: dict[str, float] = {}
        for lang in langs:
            if (model, lang, "fp16") not in aligned:
                continue
            base = aligned.get(model, lang, "fp16")
            for prec in QUANTIZED:
                if (model, lang, prec) not in aligned:
                    continue
                q = aligned.get(model, lang, prec)
                key = f"{lang}|{prec}"
                p = mcnemar_exact(base, q)
                raw_p[key] = p
                deg_rows.append({
                    "model": model, "lang": lang, "precision": prec,
                    "acc_fp16": float(base.mean()), "acc_quant": float(q.mean()),
                    "delta_acc": float(base.mean() - q.mean()),
                    "n_fp16_only": int(np.sum((base == 1) & (q == 0))),
                    "n_quant_only": int(np.sum((base == 0) & (q == 1))),
                    "mcnemar_p": p,
                })
        adj = holm(raw_p) if raw_p else {}
        for row in deg_rows:
            if row["model"] == model:
                row["mcnemar_p_holm"] = adj.get(f"{row['lang']}|{row['precision']}")

        # ---- H1/H2: interaction vs the reference language ----------------- #
        raw_pi: dict[str, float] = {}
        pending = []
        for prec in QUANTIZED:
            if (model, ref, "fp16") not in aligned or (model, ref, prec) not in aligned:
                continue
            fe, qe = aligned.get(model, ref, "fp16"), aligned.get(model, ref, prec)
            for lang in langs:
                if lang == ref:
                    continue
                if (model, lang, "fp16") not in aligned or (model, lang, prec) not in aligned:
                    continue
                fl, ql = aligned.get(model, lang, "fp16"), aligned.get(model, lang, prec)
                res = paired_bootstrap_interaction(
                    fl, ql, fe, qe, n_boot=n_boot, seed=seed, percentiles=pcts)
                key = f"{lang}|{prec}"
                raw_pi[key] = res["p_bootstrap"]
                pending.append({
                    "model": model, "lang": lang, "precision": prec,
                    "reference": ref, **res,
                })
        adj_i = holm(raw_pi) if raw_pi else {}
        for row in pending:
            key = f"{row['lang']}|{row['precision']}"
            row["p_bootstrap_holm"] = adj_i.get(key)
            # Significance is read off the CI, which is the interval actually
            # reported; the p-value is supporting information, not the verdict.
            row["ci_excludes_zero"] = bool(row["ci_low"] > 0 or row["ci_high"] < 0)
        int_rows.extend(pending)

    acc = pd.DataFrame(acc_rows)
    deg = pd.DataFrame(deg_rows)
    inter = pd.DataFrame(int_rows)
    acc.to_csv(outdir / "accuracy.csv", index=False)
    deg.to_csv(outdir / "degradation.csv", index=False)
    if len(inter):
        inter.to_csv(outdir / "interaction.csv", index=False)

    # ---- readable summary -------------------------------------------------- #
    for model in models:
        lines.append(f"\n=== {model} ===")
        lines.append("\nTable 2 -- accuracy (95% Wilson CI); chance = 0.25")
        lines.append(f"{'language':<10}{'prec':<7}{'n':>5}{'acc':>9}{'ci95':>22}")
        for _, r in acc[acc.model == model].iterrows():
            lines.append(
                f"{LANG_NAME.get(r.lang, r.lang):<10}{PREC_LABEL[r.precision]:<7}"
                f"{r.n:>5}{r.accuracy:>9.4f}   [{r.ci95_low:.4f}, {r.ci95_high:.4f}]"
                + ("   <- CI touches chance" if r.ci95_low <= 0.25 else ""))

        lines.append("\nTable 3 -- dAcc = FP16 - quantized (exact McNemar, Holm-adjusted)")
        for _, r in deg[deg.model == model].iterrows():
            lines.append(
                f"{LANG_NAME.get(r.lang, r.lang):<10}{PREC_LABEL[r.precision]:<7}"
                f"d={r.delta_acc:+.4f}  p={r.mcnemar_p:.4g}  "
                f"p_holm={r.mcnemar_p_holm:.4g}")

        if len(inter):
            sub = inter[inter.model == model]
            lines.append(
                f"\nH1/H2 -- d_int vs {LANG_NAME.get(ref, ref)} "
                f"(>0 means quantization costs this language MORE)")
            for _, r in sub.iterrows():
                lines.append(
                    f"{LANG_NAME.get(r.lang, r.lang):<10}{PREC_LABEL[r.precision]:<7}"
                    f"d_int={r.delta_interaction:+.4f}  "
                    f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}]  "
                    f"p_holm={r.p_bootstrap_holm:.4g}"
                    + ("  [CI excl. 0, UNCORRECTED]" if r.ci_excludes_zero else ""))
            lines.append(
                "  Note: the CI flag is per-contrast and NOT corrected for the "
                f"{len(sub)} contrasts tested. Where it disagrees with p_holm, "
                "the Holm-adjusted p is the claim that survives multiplicity.")

    summary = "\n".join(lines)
    print(summary)
    (outdir / "summary.txt").write_text(summary, encoding="utf-8")
    print(f"\nwrote {_rel(outdir)}/"
          "{accuracy,degradation,interaction}.csv + summary.txt")

    if args.no_figures:
        return 0
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib unavailable; figures skipped")
        return 0

    lat_path = outdir / "latency.csv"
    lat = pd.read_csv(lat_path) if lat_path.exists() else None

    for model in models:
        a = acc[acc.model == model]
        langs = sorted(a["lang"].unique())
        x = np.arange(len(langs))
        w = 0.26

        # Figure 1 -- accuracy by language and precision
        fig, ax = plt.subplots(figsize=(7.2, 4))
        for i, prec in enumerate(("fp16",) + QUANTIZED):
            sub = a[a.precision == prec].set_index("lang").reindex(langs)
            vals = sub["accuracy"].to_numpy(dtype=float)
            err = np.vstack([vals - sub["ci95_low"].to_numpy(dtype=float),
                             sub["ci95_high"].to_numpy(dtype=float) - vals])
            ax.bar(x + (i - 1) * w, vals, w, yerr=err, capsize=3,
                   label=PREC_LABEL[prec])
        ax.axhline(0.25, ls="--", lw=1, color="grey")
        ax.annotate("chance", (len(langs) - 0.55, 0.258), fontsize=8, color="grey")
        ax.set_xticks(x, [LANG_NAME.get(l, l) for l in langs])
        ax.set_ylabel("BELEBELE accuracy")
        ax.set_title(f"Accuracy by language and precision - {model}")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figdir / f"fig1_accuracy_{model}.pdf")
        plt.close(fig)

        # Figure 2 -- the paper's key figure: degradation by language
        d = deg[deg.model == model]
        if len(d):
            fig, ax = plt.subplots(figsize=(7.2, 4))
            for i, prec in enumerate(QUANTIZED):
                sub = d[d.precision == prec].set_index("lang").reindex(langs)
                ax.bar(x + (i - 0.5) * 0.36,
                       sub["delta_acc"].to_numpy(dtype=float), 0.36,
                       label=PREC_LABEL[prec])
            ax.axhline(0, color="black", lw=1)
            ax.set_xticks(x, [LANG_NAME.get(l, l) for l in langs])
            ax.set_ylabel("dAccuracy (FP16 - quantized)")
            ax.set_title(f"Quantization degradation by language - {model}")
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(figdir / f"fig2_degradation_{model}.pdf")
            plt.close(fig)

        # Figure 3 -- latency by precision
        if lat is not None and len(lat[lat.model == model]):
            L = lat[lat.model == model]
            fig, ax = plt.subplots(figsize=(7.2, 4))
            for i, prec in enumerate(("fp16",) + QUANTIZED):
                sub = L[L.precision == prec].set_index("lang").reindex(langs)
                ax.bar(x + (i - 1) * w,
                       sub["median_latency_ms"].to_numpy(dtype=float), w,
                       label=PREC_LABEL[prec])
            ax.set_xticks(x, [LANG_NAME.get(l, l) for l in langs])
            ax.set_ylabel("median latency (ms / item)")
            gpus = sorted({g for g in L["gpu_name"].dropna().unique()})
            ax.set_title(f"Latency - {model} ({', '.join(gpus) or 'GPU unrecorded'}, "
                         f"within-session)")
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(figdir / f"fig3_latency_{model}.pdf")
            plt.close(fig)

    print(f"wrote figures -> {_rel(figdir)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
