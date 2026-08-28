"""The mandatory 20-item P1 smoke test. Run this before any full language.

Section 9 of the P1 brief lists eight things that must be true before a
five-language run is worth starting. This script checks all eight and writes a
report; it returns non-zero if any of them fails, so it can gate a notebook.

  1. training completes
  2. the adapter is actually applied
  3. merging works
  4. the merged checkpoint loads
  5. it runs through the existing P0 evaluator
  6. answer-letter behaviour remains meaningful
  7. no silent FP16 fallback when quantization is requested
  8. no BELEBELE training data is involved

Output goes to --outdir, never to results/raw/. A 20-item run is a smoke test
and can never form a P1 cell.

    python scripts/run_p1_smoke.py --outdir /kaggle/working/p1_smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from quantlang import config as cfg_mod  # noqa: E402
from quantlang import data as data_mod  # noqa: E402
from quantlang import finetune  # noqa: E402
from quantlang import model as model_mod  # noqa: E402
from quantlang import p1data  # noqa: E402
from quantlang import schema  # noqa: E402
from quantlang.evaluate import _forward_letter_logits  # noqa: E402


class SmokeFailure(RuntimeError):
    """A stop condition fired. The design is not adjusted to make it pass."""


def _score(cfg, tok, model, option_ids, rows, device, max_len):
    """Score a handful of items exactly as evaluate_cell does."""
    dev = torch.device(device)
    out = []
    for row in rows:
        prompt = data_mod.build_prompt(cfg, row)
        ids = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=max_len).input_ids.to(dev)
        t0 = time.perf_counter()
        logits = _forward_letter_logits(model, ids, option_ids)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        pred = int(max(range(len(option_ids)), key=lambda i: logits[i])) + 1
        out.append({
            "item_id": row["item_id"],
            "pred": pred,
            "gold": int(row["gold"]),
            "correct": int(pred == int(row["gold"])),
            "letter_logits": logits,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "input_tokens": int(ids.shape[1]),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--lang", default="eng_Latn",
                    help="English by default: the first end-to-end validation "
                         "language named in the brief")
    ap.add_argument("--train-items", type=int, default=20)
    ap.add_argument("--eval-items", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = cfg_mod.load()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    lang = args.lang
    failures: list[str] = []
    report: dict = {"lang": lang, "checks": {}}

    models = cfg_mod.require(cfg, "models")
    entry = [m for m in models if m.get("role") == "primary"][0]
    revision = entry["revision"]
    max_len = cfg_mod.require(cfg, "scoring.max_input_tokens")

    # ---- check 8, before anything expensive -------------------------------- #
    manifest = p1data.load_split_manifest()
    train_items = p1data.load_partition(cfg, lang, "train", manifest)[:args.train_items]
    belebele_ids = set(schema.load_manifest()["item_ids"])
    leaked = [it["item_id"] for it in train_items if it["item_id"] in belebele_ids]
    train_ds = cfg_mod.require(cfg, "finetune.train_dataset")
    bench_ds = cfg_mod.require(cfg, "benchmark.hf_dataset")
    ok_no_belebele = not leaked and train_ds != bench_ds
    report["checks"]["8_no_belebele_in_training"] = {
        "pass": ok_no_belebele,
        "train_dataset": train_ds,
        "benchmark_dataset": bench_ds,
        "n_train_items": len(train_items),
        "leaked_item_ids": leaked[:5],
    }
    if not ok_no_belebele:
        raise SmokeFailure(
            f"BELEBELE data reached the training set: {leaked[:5]}. Stopping.")
    print(f"[8/8 pre-check] training corpus is {train_ds}, "
          f"{len(train_items)} items, no BELEBELE item ids present")

    # ---- checks 1-3: train, adapter applied, merge -------------------------- #
    print(f"\n=== training {args.train_items} items ({lang}) ===", flush=True)
    adapter_dir = outdir / "adapter"
    merged_dir = outdir / "merged"
    ft_meta = finetune.finetune_language(
        cfg, lang=lang, hf_id=entry["hf_id"], model_alias=entry["alias"],
        revision=revision, seed=cfg_mod.require(cfg, "finetune.seeds.main"),
        adapter_dir=adapter_dir, merged_dir=merged_dir,
        limit=args.train_items, device=args.device, tag="smoke",
    )
    report["finetune"] = ft_meta

    report["checks"]["1_training_completes"] = {
        "pass": ft_meta["training"]["n_optimizer_steps"] > 0,
        "optimizer_steps": ft_meta["training"]["n_optimizer_steps"],
        "loss_first_decile": ft_meta["training"]["loss_first_decile_mean"],
        "loss_last_decile": ft_meta["training"]["loss_last_decile_mean"],
        "train_seconds": ft_meta["training"]["train_seconds"],
    }
    report["checks"]["2_adapter_applied"] = {
        "pass": ft_meta["adapter_layer_counts"]["lora_layers"] > 0,
        **ft_meta["adapter_layer_counts"],
        **ft_meta["parameter_counts"],
    }
    report["checks"]["3_merge_succeeds"] = {
        "pass": ft_meta["merged_checkpoint"]["total_bytes"] > 0,
        **{k: v for k, v in ft_meta["merged_checkpoint"].items() if k != "path"},
    }

    # ---- baseline predictions, for comparison ------------------------------ #
    eval_rows = data_mod.load_language(cfg, lang)[: args.eval_items]
    heldout = p1data.load_partition(cfg, lang, "heldout_eval",
                                    manifest)[: args.eval_items]

    print("\n=== base model FP16, for comparison ===", flush=True)
    tok_b, base, _ = model_mod.load(cfg, entry["hf_id"], revision, "fp16",
                                    args.device)
    option_ids_base = model_mod.option_token_ids(cfg, tok_b)
    base_scores = _score(cfg, tok_b, base, option_ids_base, eval_rows,
                         args.device, max_len)
    del base, tok_b
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- checks 4-7: load merged at each precision -------------------------- #
    per_precision: dict[str, dict] = {}
    for precision in cfg_mod.require(cfg, "precisions"):
        print(f"\n=== merged checkpoint @ {precision} ===", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(torch.device(args.device))

        # model_mod.load raises PrecisionError if the quantizer did not take
        # effect, so reaching the next line IS check 7.
        tok, merged, layer_counts = model_mod.load(
            cfg, entry["hf_id"], revision, precision, args.device,
            local_checkpoint=str(merged_dir))
        option_ids = model_mod.option_token_ids(cfg, tok)

        bel = _score(cfg, tok, merged, option_ids, eval_rows, args.device, max_len)
        held = _score(cfg, tok, merged, option_ids, heldout, args.device, max_len)

        peak = (torch.cuda.max_memory_allocated(torch.device(args.device)) / 1024**3
                if torch.cuda.is_available() else None)
        per_precision[precision] = {
            "layer_counts": layer_counts,
            "option_token_ids": option_ids,
            "peak_memory_allocated_gb": peak,
            "belebele": bel,
            "heldout": held,
            "belebele_accuracy": sum(r["correct"] for r in bel) / len(bel),
            "heldout_accuracy": sum(r["correct"] for r in held) / len(held),
            "distinct_predicted_letters": sorted({r["pred"] for r in bel}),
            "all_logits_finite": all(
                all(v == v and abs(v) != float("inf") for v in r["letter_logits"])
                for r in bel + held),
            "changed_vs_base": sum(
                1 for a, b in zip(bel, base_scores) if a["pred"] != b["pred"]),
        }
        print(f"    belebele acc={per_precision[precision]['belebele_accuracy']:.2f} "
              f"heldout acc={per_precision[precision]['heldout_accuracy']:.2f} "
              f"peak={peak}")
        del merged, tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report["per_precision"] = per_precision
    report["base_fp16"] = {
        "belebele": base_scores,
        "belebele_accuracy": sum(r["correct"] for r in base_scores) / len(base_scores),
    }

    report["checks"]["4_merged_checkpoint_loads"] = {
        "pass": len(per_precision) == len(cfg_mod.require(cfg, "precisions")),
        "precisions_loaded": sorted(per_precision),
    }
    report["checks"]["5_runs_through_p0_evaluator"] = {
        "pass": all(len(v["belebele"]) == len(eval_rows)
                    for v in per_precision.values()),
        "scoring_method": cfg_mod.require(cfg, "scoring.method"),
        "n_belebele_items_scored": len(eval_rows),
        "n_heldout_items_scored": len(heldout),
    }

    # ---- check 6: answer-letter behaviour ---------------------------------- #
    letters_ok = all(v["all_logits_finite"] for v in per_precision.values())
    not_degenerate = any(len(v["distinct_predicted_letters"]) > 1
                         for v in per_precision.values())
    report["checks"]["6_answer_letter_behaviour"] = {
        "pass": letters_ok and not_degenerate,
        "all_logits_finite": letters_ok,
        "predicts_more_than_one_letter": not_degenerate,
        "per_precision_distinct_letters": {
            p: v["distinct_predicted_letters"] for p, v in per_precision.items()},
        "note": ("A collapse onto a single letter across every item means "
                 "fine-tuning destroyed the scored behaviour; the quantization "
                 "contrast would then be measured at the chance floor."),
    }

    # ---- check 7: quantization really applied, and the arms differ ---------- #
    int8_ok = per_precision.get("int8_llmint8", {}).get(
        "layer_counts", {}).get("Linear8bitLt", 0) > 0
    nf4_ok = per_precision.get("nf4", {}).get(
        "layer_counts", {}).get("Linear4bit", 0) > 0
    fp16_clean = (per_precision.get("fp16", {}).get("layer_counts", {})
                  .get("Linear8bitLt", 0) == 0
                  and per_precision.get("fp16", {}).get("layer_counts", {})
                  .get("Linear4bit", 0) == 0)
    logits_differ = len({
        tuple(round(v, 3) for v in p["belebele"][0]["letter_logits"])
        for p in per_precision.values()
    }) > 1
    report["checks"]["7_no_silent_fp16_fallback"] = {
        "pass": bool(int8_ok and nf4_ok and fp16_clean and logits_differ),
        "int8_has_Linear8bitLt": int8_ok,
        "nf4_has_Linear4bit": nf4_ok,
        "fp16_has_no_quantized_layers": fp16_clean,
        "precisions_produce_different_logits": logits_differ,
        "layer_counts": {p: v["layer_counts"] for p, v in per_precision.items()},
        "peak_memory_allocated_gb": {
            p: v["peak_memory_allocated_gb"] for p, v in per_precision.items()},
    }

    # ---- verdict ------------------------------------------------------------ #
    for name, check in report["checks"].items():
        if not check["pass"]:
            failures.append(name)

    report["all_checks_passed"] = not failures
    report["failed_checks"] = failures
    report["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["git_commit"] = finetune.git_commit()

    path = outdir / "p1_smoke_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                    encoding="utf-8")

    print("\n" + "=" * 62)
    for name, check in sorted(report["checks"].items()):
        print(f"  [{'PASS' if check['pass'] else 'FAIL'}] {name}")
    print("=" * 62)
    print(f"wrote {path}")

    if failures:
        print(f"\n*** SMOKE TEST FAILED: {failures} ***")
        print("Stop and fix this. Do not adjust the experimental design to make "
              "it pass, and do not start the five-language run.")
        return 2
    print("\nSmoke test passed. The five-language run is cleared to start.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SmokeFailure as exc:
        print(f"\n*** STOP CONDITION: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
