"""Freeze the P1 80/20 article-grouped split and the MCQ items built from it.

This is the P1 analogue of `scripts/verify_item_ids.py`: it establishes the
corpus contract once, empirically, and writes a manifest that every later run is
validated against. Training and held-out evaluation never infer their item set
from whatever a rebuild happened to produce.

What the manifest pins, per language:

  * which ARTICLES are train and which are held out (the split itself)
  * a digest over the exact MCQ items built from each partition -- item ids,
    gold letters, and full option text in order, so any change to distractor
    selection or gold placement is detectable
  * the frozen, capped held-out evaluation item ids and their gold answers,
    which is the secondary evaluation surface's contract

The split is grouped by article and the build refuses to continue if a context
turns out to span two articles, because grouping by a label only prevents
leakage if the label actually partitions the text.

    python scripts/build_p1_splits.py             # build and freeze
    python scripts/build_p1_splits.py --check     # rebuild, verify, write nothing
    python scripts/build_p1_splits.py --langs eng_Latn
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantlang import config as cfg_mod  # noqa: E402
from quantlang import p1data  # noqa: E402
from quantlang.config import REPO_ROOT  # noqa: E402

# The article split is 80/20 by ARTICLE COUNT, so the resulting ROW fractions
# only land near 0.8 because questions-per-article is fairly uniform. This band
# is wide on purpose: it is a tripwire for a broken split, not a target.
ROW_FRACTION_BAND = (0.70, 0.90)


def _rel(p: Path) -> str:
    """Display path relative to the repo when possible; absolute otherwise."""
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(Path(p).resolve())


def build_all(cfg: dict, langs: list[str]) -> dict:
    languages: dict[str, dict] = {}
    for lang in langs:
        print(f"--- {lang} ({p1data.lang_config(cfg, lang)}) ---", flush=True)
        built = p1data.build_language(cfg, lang)
        rep = built["report"]

        n_train_items = len(built["train_items"])
        n_heldout_items = len(built["heldout_items"])
        n_items = n_train_items + n_heldout_items
        row_fraction = n_train_items / n_items
        if not ROW_FRACTION_BAND[0] <= row_fraction <= ROW_FRACTION_BAND[1]:
            raise SystemExit(
                f"FATAL: {lang}: the article-grouped split put "
                f"{row_fraction:.1%} of ROWS in train, outside "
                f"{ROW_FRACTION_BAND}. Questions-per-article must be far more "
                f"skewed than assumed; inspect the corpus rather than adjusting "
                f"the band."
            )

        print(f"    source rows   : {rep['n_source_rows_total']} "
              f"(dropped empty: {rep['n_source_rows_dropped_empty']})")
        print(f"    articles      : {len(built['train_groups'])} train / "
              f"{len(built['heldout_groups'])} held out")
        print(f"    items         : {n_train_items} train / {n_heldout_items} "
              f"held out  ({row_fraction:.1%} of rows in train)")
        print(f"    heldout eval  : {len(built['heldout_eval_item_ids'])} items "
              f"(capped)")
        print(f"    surface homog.: train "
              f"{built['surface_homogeneity']['train']:.1%} / held out "
              f"{built['surface_homogeneity']['heldout']:.1%}"
              f"  (options sharing the gold's digit class)")
        w = built["window"]["train"]
        print(f"    window (train): prompt median {w['prompt_tokens_median']} "
              f"p90 {w['prompt_tokens_p90']} max {w['prompt_tokens_max']} tokens")
        nd = built["n_dropped_too_long"]
        if nd["train"] or nd["heldout"]:
            total_d = nd["train"] + nd["heldout"]
            print(f"    dropped       : {total_d} item(s) "
                  f"({total_d / built['report']['n_source_rows_used']:.2%}) whose "
                  f"question+options alone exceed max_seq_tokens")
        print(f"    evidence kept : train "
              f"{built['window']['train']['evidence_retained']:.1%} / held out "
              f"{built['window']['heldout']['evidence_retained']:.1%}")
        print(f"    train digest  : {built['train_digest'][:16]}...")

        languages[lang] = {
            "config": rep["config"],
            "n_source_rows_total": rep["n_source_rows_total"],
            "n_source_rows_dropped_empty": rep["n_source_rows_dropped_empty"],
            "n_source_rows_dropped_no_answer_span":
                rep["n_source_rows_dropped_no_answer_span"],
            "n_source_rows_used": rep["n_source_rows_used"],
            "n_articles": len(built["train_groups"]) + len(built["heldout_groups"]),
            "n_train_articles": len(built["train_groups"]),
            "n_heldout_articles": len(built["heldout_groups"]),
            "n_train_items": n_train_items,
            "n_heldout_items": n_heldout_items,
            "train_row_fraction": row_fraction,
            "option_surface_homogeneity": built["surface_homogeneity"],
            "train_items_sha256": built["train_digest"],
            "heldout_items_sha256": built["heldout_digest"],
            "train_choices_sha256": built["train_choices"],
            "heldout_choices_sha256": built["heldout_choices"],
            "context_window_stats": built["window"],
            "n_dropped_too_long": built["n_dropped_too_long"],
            "dropped_too_long_item_ids": [d["item_id"]
                                          for d in built["dropped_too_long"]],
            "train_articles": built["train_groups"],
            "heldout_articles": built["heldout_groups"],
            "heldout_eval_item_ids": built["heldout_eval_item_ids"],
            "heldout_eval_gold": built["heldout_eval_gold"],
        }

    ft = cfg_mod.require(cfg, "finetune")
    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "built_by": "scripts/build_p1_splits.py",
        "dataset": ft["train_dataset"],
        "revision": ft["hf_revision"],
        "split": ft["split"],
        "license": ft.get("license"),
        "group_key": ft["group_key"],
        "item_id_separator": ft["item_id_separator"],
        "train_fraction": ft["train_fraction"],
        "split_seed": ft["split_seed"],
        "n_options": ft["n_options"],
        "option_letters": cfg_mod.require(cfg, "scoring.option_letters"),
        "distractors": ft["distractors"],
        "heldout_eval_cap": ft["heldout_eval_cap"],
        "context_window": dict(ft["context_window"]),
        "context_window_tokenizer": p1data.tokenizer_identity(cfg),
        "max_seq_tokens": ft["training"]["max_seq_tokens"],
        "prompt_template_sha256": hashlib.sha256(
            cfg_mod.require(cfg, "scoring.prompt_template").encode("utf-8")
        ).hexdigest(),
        "note": (
            "The split is grouped by article, never by row. Distractors come "
            "from other articles within the SAME partition only. Passages are "
            "answer-centred token windows measured with the tokenizer named "
            "above -- never left-truncated, which lost the evidence for 61% of "
            "items and did so at rates from 20% (English) to 78% (Assamese). "
            "Items are rebuilt deterministically from (dataset revision, "
            "split_seed, tokenizer) and checked against the digests here; they "
            "are not stored inline. `*_choices_sha256` covers item ids, gold "
            "letters and option text only, so it is invariant to the passage "
            "representation and pins the selection policy on its own."
        ),
        "languages": languages,
    }
    manifest["sha256"] = p1data.manifest_payload_digest(manifest)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=None,
                    help="defaults to every language in benchmark.languages")
    ap.add_argument("--check", action="store_true",
                    help="rebuild and verify against the frozen manifest; "
                         "writes nothing")
    ap.add_argument("--out", default=str(p1data.SPLIT_MANIFEST_PATH))
    args = ap.parse_args()

    cfg = cfg_mod.load()
    known = cfg_mod.require(cfg, "benchmark.languages")
    langs = args.langs or known
    unknown = [l for l in langs if l not in known]
    if unknown:
        raise SystemExit(
            f"FATAL: language(s) {unknown} are not in the frozen language set "
            f"{known}.")

    print(f"dataset  : {cfg_mod.require(cfg, 'finetune.train_dataset')}")
    print(f"revision : {cfg_mod.require(cfg, 'finetune.hf_revision')}")
    print(f"group by : {cfg_mod.require(cfg, 'finetune.group_key')}")
    print(f"seed     : {cfg_mod.require(cfg, 'finetune.split_seed')}\n")

    if args.check:
        manifest = p1data.load_split_manifest(Path(args.out))
        for lang in langs:
            built = p1data.build_language(cfg, lang)
            p1data.verify_against_manifest(built, manifest)
            print(f"  {lang}: rebuild matches the frozen manifest")
        recomputed = p1data.manifest_payload_digest(manifest)
        if recomputed != manifest["sha256"]:
            raise SystemExit(
                f"FATAL: manifest sha256 does not match its own contents "
                f"({recomputed} vs {manifest['sha256']}). It has been edited by "
                f"hand."
            )
        print("\n--check: split reproduces exactly and the manifest is unedited.")
        return 0

    manifest = build_all(cfg, langs)
    out = Path(args.out)
    if out.exists() and args.langs:
        raise SystemExit(
            f"FATAL: refusing to overwrite {out} with a partial build "
            f"({len(langs)} of {len(known)} languages). Rebuild all languages, "
            f"or write elsewhere with --out."
        )
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")

    print(f"\nwrote {out.relative_to(REPO_ROOT)}")
    print(f"  languages : {len(manifest['languages'])}")
    print(f"  sha256    : {manifest['sha256']}")
    total_train = sum(v["n_train_items"] for v in manifest["languages"].values())
    total_eval = sum(len(v["heldout_eval_item_ids"])
                     for v in manifest["languages"].values())
    print(f"  train items across all languages : {total_train}")
    print(f"  held-out eval items (capped)     : {total_eval}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
