"""Run one or more (language x precision) cells for the pinned primary model.

Everything that defines the experiment -- model, revision, languages, prompt,
scoring method, item set -- comes from configs/experiment.yaml and the frozen
manifest. This script only chooses WHICH cells to run, never what a cell means.

On Kaggle:
    python scripts/run_eval.py --precision fp16 --outdir /kaggle/working --tag p0
    python scripts/run_eval.py --precision int8_llmint8 --outdir /kaggle/working --tag p0
    python scripts/run_eval.py --precision nf4 --outdir /kaggle/working --tag p0

Sanity check before committing a session to the full grid (~20 items, all
languages, all precisions, output kept out of the paper's provenance chain):
    python scripts/run_eval.py --all-precisions --limit 20 \
        --outdir results/smoke --tag smoke --store-prompts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantlang import config as cfg_mod  # noqa: E402
from quantlang import finetune  # noqa: E402
from quantlang.evaluate import evaluate_cell  # noqa: E402
from quantlang.model import PRECISIONS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-alias", default=None,
                    help="defaults to the config's role=primary model")
    ap.add_argument("--precision", action="append", choices=list(PRECISIONS),
                    help="repeatable; omit with --all-precisions")
    ap.add_argument("--all-precisions", action="store_true")
    ap.add_argument("--langs", nargs="+", default=None,
                    help="defaults to every language in the config")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap items per language. Sanity checks only -- a "
                         "limited run can never form a paper cell.")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=1,
                    help="forward passes per item; >1 for the latency protocol")
    ap.add_argument("--store-prompts", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--local-checkpoint", default=None,
                    help="P1 FT arm: score the merged fine-tuned checkpoint in "
                         "this directory instead of the pinned Hub model. The "
                         "result alias becomes finetune.ft_alias(...) "
                         "automatically, so an FT cell can never collide with a "
                         "Base cell.")
    ap.add_argument("--ft-epochs", type=int, default=1,
                    help="Adaptation strength of the checkpoint being scored. "
                         "Goes into the result alias, so a P1-Strong cell is "
                         "self-describing in its own filename and can never be "
                         "pooled with a P1-Standard cell. Default 1.")
    ap.add_argument("--ft-lang", default=None,
                    help="the language the checkpoint was fine-tuned ON. "
                         "Required with --local-checkpoint, and deliberately "
                         "separate from --langs: a cross-lingual cell would be "
                         "a different experiment, so naming both makes the "
                         "distinction explicit in the filename.")
    args = ap.parse_args()

    cfg = cfg_mod.load()

    models = cfg_mod.require(cfg, "models")
    if args.model_alias:
        chosen = [m for m in models if m["alias"] == args.model_alias]
        if not chosen:
            raise SystemExit(
                f"FATAL: no model with alias {args.model_alias!r} in the config. "
                f"Available: {[m['alias'] for m in models]}"
            )
    else:
        chosen = [m for m in models if m.get("role") == "primary"]
        if len(chosen) != 1:
            raise SystemExit(
                f"FATAL: expected exactly one role=primary model, found "
                f"{len(chosen)}. Name one explicitly with --model-alias."
            )
    entry = chosen[0]
    revision = entry.get("revision")
    if not revision:
        raise SystemExit(
            f"FATAL: {entry['alias']} has no pinned revision. Run "
            f"`python scripts/pin_revisions.py` first. An unpinned run is not "
            f"reproducible and its numbers cannot go in the paper."
        )

    precisions = list(PRECISIONS) if args.all_precisions else (args.precision or [])
    if not precisions:
        raise SystemExit("FATAL: pass --precision (repeatable) or --all-precisions")

    langs = args.langs or cfg_mod.require(cfg, "benchmark.languages")
    known = cfg_mod.require(cfg, "benchmark.languages")
    unknown = [l for l in langs if l not in known]
    if unknown:
        raise SystemExit(
            f"FATAL: language(s) {unknown} are not in the frozen language set "
            f"{known}. Adding a language is a design decision, not a flag."
        )

    if args.limit is not None and args.tag != "smoke":
        print(
            f"WARNING: --limit {args.limit} with tag {args.tag!r}. A limited run "
            f"cannot form a paper cell; build_tidy will reject it as incomplete.",
            file=sys.stderr,
        )

    # ---- the FT arm ---------------------------------------------------------
    # Before this existed there was no way to evaluate a fine-tuned checkpoint
    # through the sanctioned evaluator at all. The one full P1 evaluation was
    # therefore run by ad-hoc code, which loaded the base model from the Hub and
    # produced an "FT" cell whose logits were bit-identical to the Base arm's.
    # Every guard below exists to make that specific mistake impossible.
    result_alias = entry["alias"]
    local_checkpoint = args.local_checkpoint
    if local_checkpoint:
        if not args.ft_lang:
            raise SystemExit(
                "FATAL: --local-checkpoint requires --ft-lang, the language the "
                "checkpoint was fine-tuned on. It goes into the result alias, so "
                "an FT cell is self-describing in its own filename.")
        if args.ft_lang not in known:
            raise SystemExit(
                f"FATAL: --ft-lang {args.ft_lang} is not in the frozen language "
                f"set {known}.")
        ckpt = Path(local_checkpoint)
        if not ckpt.is_dir():
            raise SystemExit(
                f"FATAL: --local-checkpoint {local_checkpoint} is not a "
                f"directory. Stopping rather than silently falling back to the "
                f"base model, which would produce an FT cell identical to Base.")
        if not any(ckpt.glob("*.safetensors")):
            raise SystemExit(
                f"FATAL: {local_checkpoint} holds no *.safetensors weights. "
                f"That is not a merged checkpoint.")
        result_alias = finetune.ft_alias(entry["alias"], args.ft_lang,
                                         epochs=args.ft_epochs)
    elif args.ft_lang:
        raise SystemExit(
            "FATAL: --ft-lang without --local-checkpoint. Naming a fine-tuning "
            "language while scoring the base model would mislabel a Base cell "
            "as an FT one.")

    outdir = Path(args.outdir)
    print(f"model      : {entry['hf_id']}")
    print(f"revision   : {revision}")
    print(f"arm        : {'finetuned on ' + args.ft_lang if local_checkpoint else 'base'}")
    print(f"weights    : {local_checkpoint or 'hub @ ' + revision[:12]}")
    print(f"alias      : {result_alias}")
    print(f"precisions : {precisions}")
    print(f"languages  : {langs}")
    print(f"outdir     : {outdir}\n")

    summaries = []
    # Precision is the OUTER loop: each model load is expensive, and looping
    # languages inside one load keeps every language in a precision on the same
    # weights in the same session, which is what makes the latency comparable.
    for precision in precisions:
        for lang in langs:
            print(f"--- {precision} / {lang} ---", flush=True)
            meta = evaluate_cell(
                cfg,
                hf_id=entry["hf_id"],
                model_alias=result_alias,
                revision=revision,
                precision=precision,
                lang=lang,
                outdir=outdir,
                tag=args.tag,
                limit=args.limit,
                warmup=args.warmup,
                repeats=args.repeats,
                store_prompts=args.store_prompts,
                device=args.device,
                local_checkpoint=local_checkpoint,
            )
            print(f"    acc={meta['accuracy']:.4f} ({meta['n_correct']}/{meta['n_items']})"
                  f"  median={meta['median_latency_ms']:.1f}ms"
                  f"  peak={meta['peak_memory_reserved_gb']:.2f}GB"
                  f"  truncated={meta['n_truncated']}", flush=True)
            summaries.append(meta)

    print("\n=== session summary ===")
    for m in summaries:
        print(f"{m['precision']:<14}{m['lang']:<10}acc={m['accuracy']:.4f}  "
              f"median={m['median_latency_ms']:.1f}ms")
    (outdir / f"{args.tag}__session_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
