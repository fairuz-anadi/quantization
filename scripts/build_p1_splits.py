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

# ---- the acceptance gates for algorithm_version 2 -------------------------- #
#
# Version 1 shipped an item set that was solvable without reading the passage,
# and nothing in the pipeline noticed, because nothing in the pipeline measured
# it. These are the measurements that would have caught it.

# "Choose the option that appears verbatim in the passage" must be worth no more
# than a guess. v1 scored ~0.96 (English) / ~0.92 (Bangla) here. Under v2 every
# option is present, so the exact value is 0.25 and the tolerance covers nothing
# but arithmetic.
SHORTCUT_ACCURACY_CEILING = 0.30

# The gold and a distractor must be present at the SAME rate. A gap here is the
# shortcut in its raw form, before any tie-breaking.
MAX_PRESENCE_GAP = 0.01

# Drop rates may differ between languages -- Bangla articles tokenise ~3.6x
# longer -- but not so far that the surviving items are a differently selected
# sample per language. Measured at max_seq_tokens=2048: 0.7% English, 6.5%
# Bangla. This bounds the SPREAD across the final-scope languages, which is the
# comparison that a per-language ceiling cannot make.
MAX_DROP_RATE_SPREAD = 0.10

# No option position may carry a usable cue. Uniform is 0.25; this allows real
# sampling wobble while still failing a construction that systematically parks
# the gold first or last, either among the letters or among the option spans as
# they appear in the passage.
MAX_POSITION_SHARE = 0.40


def _rel(p: Path) -> str:
    """Display path relative to the repo when possible; absolute otherwise."""
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(Path(p).resolve())


def assert_construction_is_honest(lang: str, built: dict) -> None:
    """Refuse to freeze an item set that can be solved without reading it.

    Version 1 passed every check the pipeline had -- four options present, gold
    letter consistent, adapter attaching, quantization applying, digests stable
    -- while being ~96% solvable by substring presence alone. Structural checks
    cannot catch that. These are behavioural.
    """
    for partition in ("train", "heldout"):
        d = built["diagnostics"][partition]
        where = f"{lang}/{partition}"
        if d["n_items"] == 0:
            raise SystemExit(f"FATAL: {where}: no items were built.")

        if d["lexical_shortcut_accuracy"] > SHORTCUT_ACCURACY_CEILING:
            raise SystemExit(
                f"FATAL: {where}: 'choose the option that appears in the "
                f"passage' scores {d['lexical_shortcut_accuracy']:.3f}, above "
                f"{SHORTCUT_ACCURACY_CEILING}. The task is solvable without "
                f"reading. This is the exact failure that invalidated "
                f"algorithm_version 1; do not raise the ceiling."
            )
        gap = abs(d["gold_in_context_rate"] - d["distractor_in_context_rate"])
        if gap > MAX_PRESENCE_GAP:
            raise SystemExit(
                f"FATAL: {where}: the gold is present in the passage "
                f"{d['gold_in_context_rate']:.1%} of the time and a distractor "
                f"{d['distractor_in_context_rate']:.1%} -- a {gap:.1%} gap. "
                f"Presence must carry no information about which option is "
                f"correct."
            )
        if d["same_article_distractor_rate"] < 1.0:
            raise SystemExit(
                f"FATAL: {where}: only "
                f"{d['same_article_distractor_rate']:.1%} of items draw every "
                f"distractor from their own article. An answer from another "
                f"article is not in this passage."
            )
        if d["n_items_with_duplicate_options"]:
            raise SystemExit(
                f"FATAL: {where}: {d['n_items_with_duplicate_options']} item(s) "
                f"have duplicate options, so more than one letter is defensible.")
        if d["option_counts"] != [4]:
            raise SystemExit(
                f"FATAL: {where}: option counts {d['option_counts']}, expected "
                f"exactly [4].")

        letter_share = (max(d["gold_letter_distribution"].values())
                        / d["n_items"])
        if letter_share > MAX_POSITION_SHARE:
            raise SystemExit(
                f"FATAL: {where}: one answer letter takes {letter_share:.1%} of "
                f"the golds, above {MAX_POSITION_SHARE:.0%}. Always answering "
                f"that letter would beat reading.")
        rank_share = max(d["gold_position"]["share_by_rank"].values())
        if rank_share > MAX_POSITION_SHARE:
            raise SystemExit(
                f"FATAL: {where}: the gold is the {rank_share:.1%}-most common "
                f"option at one position in the passage, above "
                f"{MAX_POSITION_SHARE:.0%}. A span-covering window must not "
                f"leave the gold at a predictable place in the passage."
            )


def assert_drop_rates_are_comparable(cfg: dict, languages: dict) -> None:
    """The drop rate may vary between languages, but not by enough to matter.

    A per-language ceiling cannot see this. If English keeps 99% of its rows and
    Bangla 75%, the two training sets are differently selected samples and
    "fine-tuning helped English more" stops being separable from "Bangla trained
    on a different kind of article". That is a cross-language comparison, so it
    is made here, once, with every language in hand.
    """
    scope = [l for l in cfg_mod.require(cfg, "finetune.final_scope_languages")
             if l in languages]
    if len(scope) < 2:
        return
    rates = {l: languages[l]["construction_diagnostics"]["train"]["drop_rate"]
             for l in scope}
    spread = max(rates.values()) - min(rates.values())
    print(f"\ndrop rates across the final scope: "
          + ", ".join(f"{l} {r:.1%}" for l, r in sorted(rates.items()))
          + f"  (spread {spread:.1%})")
    if spread > MAX_DROP_RATE_SPREAD:
        raise SystemExit(
            f"FATAL: construction drop rates across {scope} span {spread:.1%}, "
            f"above the {MAX_DROP_RATE_SPREAD:.0%} ceiling: {rates}. The "
            f"surviving items are a differently selected sample per language, "
            f"which is a confound sitting directly on the quantity P1 measures. "
            f"Raise finetune.training.max_seq_tokens, or reduce the scope -- do "
            f"not raise this ceiling."
        )


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
              f"{built['window']['train']['all_options_retained']:.1%} / held out "
              f"{built['window']['heldout']['all_options_retained']:.1%}"
              f"  (items whose passage contains ALL FOUR options)")
        d = built["diagnostics"]["train"]
        print(f"    shortcut acc  : {d['lexical_shortcut_accuracy']:.4f} "
              f"(gold present {d['gold_in_context_rate']:.1%}, distractor "
              f"present {d['distractor_in_context_rate']:.1%})")
        print(f"    gold position : letters "
              f"{max(d['gold_letter_distribution'].values()) / d['n_items']:.1%} max, "
              f"passage order {max(d['gold_position']['share_by_rank'].values()):.1%} max")
        print(f"    train digest  : {built['train_digest'][:16]}...")

        assert_construction_is_honest(lang, built)

        languages[lang] = {
            "construction_diagnostics": built["diagnostics"],
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

    assert_drop_rates_are_comparable(cfg, languages)

    # The common training-set size is a property OF the built languages, so it
    # can only be computed once every one of them exists. It is recorded here
    # and read back by p1data.load_partition, so no training run can pick it up
    # from anywhere else.
    ft = cfg_mod.require(cfg, "finetune")
    scope = [l for l in ft["final_scope_languages"] if l in languages]
    if ft["equalise_train_partition"] and len(scope) == len(
            ft["final_scope_languages"]):
        cap = min(languages[l]["n_train_items"] for l in scope)
        for lang in languages:
            entry = languages[lang]
            entry["n_train_items_equalised"] = (
                min(cap, entry["n_train_items"]) if lang in scope
                else entry["n_train_items"])
        print(f"\nequalised training size across {scope}: {cap} items "
              + ", ".join(f"({l} built {languages[l]['n_train_items']})"
                          for l in scope))

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
        "final_scope_languages": ft["final_scope_languages"],
        "equalise_train_partition": ft["equalise_train_partition"],
        "note": (
            "algorithm_version 2. The split is grouped by article, never by "
            "row. Distractors are the answers to OTHER questions about the "
            "SAME article, and the passage is the token window that covers all "
            "four option spans, expanded outward to context_budget_tokens. "
            "Every option is therefore a verbatim substring of the passage and "
            "'choose the option that appears in the passage' scores exactly "
            "0.25 in every language. Version 1 centred the window on the gold "
            "instead and drew distractors from other articles, which made that "
            "same heuristic worth ~96% (English) and ~92% (Bangla); no v1 item "
            "set or result is admissible. Items are rebuilt deterministically "
            "from (dataset revision, split_seed, tokenizer) and checked against "
            "the digests here; they are not stored inline. `*_choices_sha256` "
            "covers item ids, gold letters and option text only, so it is "
            "invariant to the passage representation and pins the selection "
            "policy on its own."
        ),
        "languages": languages,
    }
    manifest["sha256"] = p1data.manifest_payload_digest(manifest)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=None,
                    help="defaults to finetune.final_scope_languages")
    ap.add_argument("--check", action="store_true",
                    help="rebuild and verify against the frozen manifest; "
                         "writes nothing")
    ap.add_argument("--out", default=str(p1data.SPLIT_MANIFEST_PATH))
    args = ap.parse_args()

    cfg = cfg_mod.load()
    known = cfg_mod.require(cfg, "benchmark.languages")
    scope = cfg_mod.require(cfg, "finetune.final_scope_languages")
    # The P1 CORPUS covers the final-scope languages, not all of
    # benchmark.languages. P0 evaluated five and those results stand; P0 reads
    # configs/item_id_manifest.json and never touches this file, so narrowing
    # the P1 corpus cannot reach it.
    langs = args.langs or scope
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
    if out.exists() and sorted(langs) != sorted(scope):
        raise SystemExit(
            f"FATAL: refusing to overwrite {_rel(out)} with a build of "
            f"{sorted(langs)}, which is not the final scope {sorted(scope)}. "
            f"A manifest covering some other set of languages is not the P1 "
            f"corpus. Write elsewhere with --out."
        )
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")

    print(f"\nwrote {_rel(out)}")
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
