"""Train one language-specific LoRA and merge it into an FP16 checkpoint.

One language per invocation, because a P1 language is a self-contained unit of
work: train, merge, evaluate three precisions, keep the raw output. That fits a
Kaggle session and can be restarted without redoing the others.

Everything defining the experiment -- base model, revision, corpus, split, LoRA
configuration, seed -- comes from configs/experiment.yaml and the frozen
configs/p1_split_manifest.json. This script only chooses WHICH language.

    python scripts/run_finetune.py --lang eng_Latn --outdir /kaggle/working/p1
    python scripts/run_finetune.py --lang eng_Latn --limit 20 --tag smoke \
        --outdir /kaggle/working/p1_smoke

The second seed is a sensitivity probe, not another experimental cell, and is
only permitted for the languages named in finetune.seeds.sensitivity_languages:

    python scripts/run_finetune.py --lang eng_Latn --seed-role sensitivity ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantlang import config as cfg_mod  # noqa: E402
from quantlang import finetune  # noqa: E402


def resolve_seed(cfg: dict, lang: str, role: str) -> int:
    seeds = cfg_mod.require(cfg, "finetune.seeds")
    if role == "main":
        return int(seeds["main"])
    if role != "sensitivity":
        raise SystemExit(f"FATAL: unknown --seed-role {role!r}")
    allowed = seeds.get("sensitivity_languages") or []
    if lang not in allowed:
        raise SystemExit(
            f"FATAL: {lang} is not in finetune.seeds.sensitivity_languages "
            f"{allowed}. The second seed is a sensitivity probe on those "
            f"languages only; running it elsewhere would add experimental cells "
            f"that were never declared."
        )
    return int(seeds["sensitivity"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--outdir", required=True,
                    help="adapters/ and merged/ are created underneath")
    ap.add_argument("--tag", default="p1")
    ap.add_argument("--seed-role", default="main", choices=["main", "sensitivity"])
    ap.add_argument("--model-alias", default=None,
                    help="defaults to the config's role=primary model")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap training items. Smoke tests only -- a limited run "
                         "is not a P1 cell.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override finetune.training.epochs. This is the ONLY "
                         "factor P1-Strong varies; everything else stays "
                         "frozen. Artefacts and the result alias are marked "
                         "with the epoch count so a stronger run can never be "
                         "confused with the 1-epoch P1-Standard cells.")
    args = ap.parse_args()

    cfg = cfg_mod.load()

    known = cfg_mod.require(cfg, "benchmark.languages")
    if args.lang not in known:
        raise SystemExit(
            f"FATAL: {args.lang} is not in the frozen language set {known}. "
            f"Adding a language is a design decision, not a flag."
        )

    models = cfg_mod.require(cfg, "models")
    if args.model_alias:
        chosen = [m for m in models if m["alias"] == args.model_alias]
    else:
        chosen = [m for m in models if m.get("role") == "primary"]
    if len(chosen) != 1:
        raise SystemExit(
            f"FATAL: expected exactly one model, found {len(chosen)}. "
            f"Available: {[m['alias'] for m in models]}")
    entry = chosen[0]
    revision = entry.get("revision")
    if not revision:
        raise SystemExit(
            f"FATAL: {entry['alias']} has no pinned revision. Run "
            f"`python scripts/pin_revisions.py` first.")

    seed = resolve_seed(cfg, args.lang, args.seed_role)

    if args.limit is not None and args.tag == "p1":
        print(
            f"WARNING: --limit {args.limit} with tag 'p1'. A limited run is a "
            f"smoke test, not a P1 cell; give it --tag smoke so it can never be "
            f"mistaken for one.", file=sys.stderr)

    outdir = Path(args.outdir)
    # The epoch marker keeps P1-Strong artefacts beside P1-Standard's rather
    # than on top of them. One epoch keeps the original bare name so the
    # completed sessions B and C stay addressable by the paths they wrote.
    epochs = args.epochs
    suffix = "" if epochs in (None, 1) else f"__{epochs}ep"
    run_name = f"{args.lang}__seed{seed}{suffix}"
    adapter_dir = outdir / "adapters" / run_name
    merged_dir = outdir / "merged" / run_name

    print(f"base model : {entry['hf_id']} @ {revision}")
    print(f"language   : {args.lang}")
    print(f"seed       : {seed} ({args.seed_role})")
    print(f"epochs     : {epochs if epochs is not None else 'config default'}"
          f"{'   [P1-Strong]' if epochs not in (None, 1) else ''}")
    print(f"adapter    : {adapter_dir}")
    print(f"merged     : {merged_dir}\n")

    meta = finetune.finetune_language(
        cfg,
        lang=args.lang,
        hf_id=entry["hf_id"],
        model_alias=entry["alias"],
        revision=revision,
        seed=seed,
        adapter_dir=adapter_dir,
        merged_dir=merged_dir,
        limit=args.limit,
        device=args.device,
        tag=args.tag,
        epochs=epochs,
    )

    t = meta["training"]
    print("\n=== training summary ===")
    print(f"  examples            : {t['n_examples']}")
    print(f"  optimizer steps     : {t['n_optimizer_steps']}")
    print(f"  loss first decile   : {t['loss_first_decile_mean']:.4f}")
    print(f"  loss last decile    : {t['loss_last_decile_mean']:.4f}")
    print(f"  train seconds       : {t['train_seconds']:.1f}")
    print(f"  trainable params    : {meta['parameter_counts']['trainable_parameters']:,}")
    print(f"  merged checkpoint   : {meta['merged_checkpoint']['total_gb']:.2f} GB")
    print(f"  peak memory alloc.  : {meta['peak_memory_allocated_gb']}")

    summary = outdir / f"{args.tag}__{run_name}__finetune.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    print(f"\nwrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
