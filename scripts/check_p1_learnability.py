"""Is the P1 task worth fine-tuning on? Answer BEFORE spending the GPU hours.

This gate exists because the pipeline did not have one. Every structural check
passed -- four options, adapter attached, merge clean, quantization applied,
digests stable -- while the item set was solvable without reading the passage,
and a full English fine-tune plus three evaluations (~99 minutes) went by before
anyone measured whether the base model already solved it.

Two questions are asked here, in the order that spends the least.

  1. CPU, seconds. Does the frozen item set still carry a lexical shortcut?
     `scripts/build_p1_splits.py` already refuses to freeze one, so this
     re-checks the manifest that is actually on disk rather than trusting that
     the build which wrote it was the build with the gate.

  2. GPU, a few minutes. How well does the BASE model already do on the P1 task,
     scored by the exact P0 evaluator? If it is at ceiling there is no headroom
     for fine-tuning to move anything, the FT arm reproduces the Base arm, and
     the run is not worth starting.

THE CEILING IS NOT INVENTED. It is P0's own best measured cell -- English FP16
on BELEBELE, read from results/ALL_P0_RESULTS/tables/accuracy.csv. If the base
model scores higher on P1 training items than on the easiest thing P0 measured,
the P1 task is easier than the benchmark it is meant to support, and no
quantization contrast on the FT arm would be measurable above it.

    python scripts/check_p1_learnability.py --langs eng_Latn ben_Beng \
        --outdir /kaggle/working/p1_gate
    python scripts/check_p1_learnability.py --no-gpu     # question 1 only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantlang import config as cfg_mod  # noqa: E402
from quantlang import p1data  # noqa: E402
from quantlang.config import REPO_ROOT  # noqa: E402
from quantlang.statistics import wilson_ci  # noqa: E402

P0_ACCURACY_TABLE = REPO_ROOT / "results" / "ALL_P0_RESULTS" / "tables" / "accuracy.csv"

# How many training items to score. Large enough that the Wilson interval is
# narrow enough to compare against the ceiling, small enough to stay a gate
# rather than an experiment. At n=300 the half-width at p=0.5 is ~5.7pp.
DEFAULT_SAMPLE = 300

# Restated from scripts/build_p1_splits.py, which is what actually enforces it
# at freeze time. Duplicated deliberately: this script must be able to condemn a
# manifest written by some earlier build.
SHORTCUT_ACCURACY_CEILING = 0.30


def p0_ceiling() -> tuple[float, str]:
    """P0's highest measured accuracy, and which cell it came from."""
    if not P0_ACCURACY_TABLE.exists():
        raise SystemExit(
            f"FATAL: {P0_ACCURACY_TABLE} is missing, so the ceiling would have "
            f"to be invented. It is P0's best measured cell and nothing else."
        )
    best = None
    with P0_ACCURACY_TABLE.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            acc = float(row["accuracy"])
            if best is None or acc > best[0]:
                best = (acc, f"{row['lang']}/{row['precision']}")
    if best is None:
        raise SystemExit(f"FATAL: {P0_ACCURACY_TABLE} has no rows.")
    return best


def check_shortcut(manifest: dict, langs: list[str]) -> dict:
    """Question 1: does the frozen item set carry a lexical shortcut?"""
    out: dict[str, dict] = {}
    failures: list[str] = []
    for lang in langs:
        entry = (manifest.get("languages") or {}).get(lang)
        if entry is None:
            raise SystemExit(
                f"FATAL: {lang} is not in the frozen P1 split manifest. It is "
                f"not part of the pinned P1 corpus.")
        diags = entry.get("construction_diagnostics")
        if not diags:
            raise SystemExit(
                f"FATAL: {lang} has no construction_diagnostics in the "
                f"manifest. It was frozen by a build that did not measure the "
                f"shortcut at all, which is exactly the condition that let "
                f"algorithm_version 1 through. Rebuild with "
                f"scripts/build_p1_splits.py.")
        for partition in ("train", "heldout"):
            d = diags[partition]
            shortcut = d["lexical_shortcut_accuracy"]
            gap = abs(d["gold_in_context_rate"] - d["distractor_in_context_rate"])
            ok = shortcut <= SHORTCUT_ACCURACY_CEILING and gap <= 0.01
            out[f"{lang}/{partition}"] = {
                "lexical_shortcut_accuracy": shortcut,
                "gold_in_context_rate": d["gold_in_context_rate"],
                "distractor_in_context_rate": d["distractor_in_context_rate"],
                "presence_gap": gap,
                "pass": ok,
            }
            print(f"  {lang}/{partition:<8} shortcut={shortcut:.4f}  "
                  f"gold_present={d['gold_in_context_rate']:.1%}  "
                  f"distractor_present={d['distractor_in_context_rate']:.1%}  "
                  f"[{'PASS' if ok else 'FAIL'}]")
            if not ok:
                failures.append(f"{lang}/{partition}")
    if failures:
        raise SystemExit(
            f"FATAL: {failures} can be solved by substring presence alone. This "
            f"is the failure that invalidated algorithm_version 1. Do not "
            f"fine-tune on this item set."
        )
    return out


def check_base_accuracy(cfg: dict, langs: list[str], sample: int,
                        device: str, ceiling: float, ceiling_cell: str) -> dict:
    """Question 2: does the base model already solve the P1 task?"""
    import torch

    from quantlang import model as model_mod
    from quantlang.evaluate import _forward_letter_logits

    models = cfg_mod.require(cfg, "models")
    primary = [m for m in models if m.get("role") == "primary"]
    if len(primary) != 1:
        raise SystemExit(
            f"FATAL: expected exactly one role=primary model, found "
            f"{len(primary)}.")
    entry = primary[0]

    max_len = cfg_mod.require(cfg, "scoring.max_input_tokens")
    print(f"\nloading {entry['hf_id']} @ {entry['revision'][:12]} in fp16...",
          flush=True)
    tok, model, _ = model_mod.load(cfg, entry["hf_id"], entry["revision"],
                                   "fp16", device)
    option_ids = model_mod.option_token_ids(cfg, tok)
    dev = torch.device(device)

    out: dict[str, dict] = {}
    over: list[str] = []
    for lang in langs:
        items = p1data.load_partition(cfg, lang, "train")
        scored = items[:sample]
        n_correct = 0
        for it in scored:
            prompt = p1data.build_p1_prompt(cfg, it)
            ids = tok(prompt, return_tensors="pt", truncation=True,
                      max_length=max_len).input_ids.to(dev)
            logits = _forward_letter_logits(model, ids, option_ids)
            pred = int(max(range(len(option_ids)),
                           key=lambda i: logits[i])) + 1
            n_correct += int(pred == int(it["gold"]))
        acc = n_correct / len(scored)
        lo, hi = wilson_ci(n_correct, len(scored))
        # The gate fires on the LOWER bound: we stop only when the base model is
        # confidently above the ceiling, never on a point estimate that happened
        # to land there.
        at_ceiling = lo > ceiling
        out[lang] = {
            "n_scored": len(scored),
            "n_train_items_available": len(items),
            "base_fp16_accuracy": acc,
            "ci95": [lo, hi],
            "p0_ceiling": ceiling,
            "p0_ceiling_cell": ceiling_cell,
            "at_ceiling": at_ceiling,
        }
        print(f"  {lang:<10} base fp16 acc={acc:.4f} "
              f"[{lo:.4f}, {hi:.4f}] on {len(scored)} items  "
              f"(P0 ceiling {ceiling:.4f} from {ceiling_cell})  "
              f"[{'STOP' if at_ceiling else 'OK'}]")
        if at_ceiling:
            over.append(lang)

    if over:
        raise SystemExit(
            f"FATAL: the base model already solves the P1 task for {over} -- "
            f"its 95% lower bound sits above P0's best measured cell "
            f"({ceiling:.4f}, {ceiling_cell}). Fine-tuning has no headroom, the "
            f"FT arm would reproduce the Base arm, and ~100 GPU-minutes per "
            f"language would buy nothing. Stop and report this rather than "
            f"training anyway."
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=None,
                    help="defaults to finetune.final_scope_languages")
    ap.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-gpu", action="store_true",
                    help="run the CPU shortcut check only")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    cfg = cfg_mod.load()
    langs = args.langs or cfg_mod.require(cfg, "finetune.final_scope_languages")
    manifest = p1data.load_split_manifest()

    ceiling, ceiling_cell = p0_ceiling()
    report = {
        "split_manifest_sha256": manifest["sha256"],
        "context_window": manifest["context_window"],
        "distractors": manifest["distractors"],
        "p0_ceiling": ceiling,
        "p0_ceiling_cell": ceiling_cell,
        "langs": langs,
    }

    print("=== 1. lexical shortcut (CPU, from the frozen manifest) ===")
    report["shortcut"] = check_shortcut(manifest, langs)

    if args.no_gpu:
        print("\n--no-gpu: base-model accuracy NOT measured. The learnability "
              "gate is only half satisfied; do not start a fine-tune on this.")
        report["base_accuracy"] = None
    else:
        print("\n=== 2. base-model accuracy on P1 training items (GPU) ===")
        report["base_accuracy"] = check_base_accuracy(
            cfg, langs, args.sample, args.device, ceiling, ceiling_cell)

    report["all_checks_passed"] = not args.no_gpu
    if args.outdir:
        out = Path(args.outdir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "p1_learnability_report.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        print(f"\nwrote {path}")

    if not args.no_gpu:
        print("\nGATE PASSED: the item set carries no substring shortcut and "
              "the base model has room to improve. Fine-tuning is worth "
              "starting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
