"""Determine and freeze the BELEBELE item_id key.

The project brief assumed `question_number` was the item id. It is not: it
takes only two distinct values across the 900-item split, so it cannot key an
item. Worse, an identity check on it would PASS trivially across languages
({1,2} == {1,2}) and give false confidence while the paired bootstrap resampled
from two ids.

This script establishes the real key empirically, proves the key set is
identical across every language, proves the gold answer agrees per key, and
writes an immutable manifest that all downstream code validates against.

Run:  python scripts/verify_item_ids.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset  # noqa: E402

from quantlang import config  # noqa: E402

# Candidate keys, most-preferred first. Each is a tuple of column names whose
# combination is tested for uniqueness. We do NOT fall back to row position:
# BELEBELE row order differs between languages, so positional pairing silently
# mispairs items.
CANDIDATE_KEYS: list[tuple[str, ...]] = [
    ("question_number",),
    ("link", "question_number"),
]

ITEM_ID_SEP = "#"


def compose_item_id(row: dict, key: tuple[str, ...]) -> str:
    return ITEM_ID_SEP.join(str(row[c]) for c in key)


def main() -> int:
    cfg = config.load()
    bench = cfg["benchmark"]
    langs: list[str] = bench["languages"]
    expected_n: int = bench["n_items_per_lang"]
    dataset: str = bench["hf_dataset"]
    split: str = bench["split"]
    # The manifest is the benchmark contract, so it must be built from the
    # pinned dataset commit, not from whatever `main` happens to be today.
    revision: str = config.require(cfg, "benchmark.hf_revision")

    print(f"dataset={dataset} split={split} revision={revision}")
    print(f"languages={langs}")
    print(f"expected items per language={expected_n}\n")

    loaded: dict[str, list[dict]] = {}
    for lang in langs:
        ds = load_dataset(dataset, lang, split=split, revision=revision)
        rows = [dict(r) for r in ds]
        if len(rows) != expected_n:
            raise SystemExit(
                f"FATAL: {lang} yielded {len(rows)} items, expected {expected_n}. "
                f"Not padding, not resampling, not continuing."
            )
        loaded[lang] = rows
        print(f"  loaded {lang}: {len(rows)} rows")

    # --- choose the key -----------------------------------------------------
    chosen: tuple[str, ...] | None = None
    for key in CANDIDATE_KEYS:
        counts = {
            lang: len({compose_item_id(r, key) for r in rows})
            for lang, rows in loaded.items()
        }
        ok = all(n == expected_n for n in counts.values())
        status = "UNIQUE" if ok else "NOT UNIQUE"
        print(f"\ncandidate key {key}: {status}")
        for lang, n in counts.items():
            print(f"    {lang}: {n}/{expected_n} distinct")
        if ok:
            chosen = key
            break

    if chosen is None:
        raise SystemExit(
            "FATAL: no candidate key uniquely identifies items. Add a candidate "
            "to CANDIDATE_KEYS after inspecting the data; do not proceed with a "
            "non-unique key."
        )
    print(f"\nchosen item_id key: {chosen}")

    # --- prove parity across languages --------------------------------------
    ids = {lang: {compose_item_id(r, chosen) for r in rows} for lang, rows in loaded.items()}
    ref_lang = bench["reference_language"]
    ref_ids = ids[ref_lang]
    for lang, s in ids.items():
        if s != ref_ids:
            missing = sorted(ref_ids - s)[:5]
            extra = sorted(s - ref_ids)[:5]
            raise SystemExit(
                f"FATAL: item_id set for {lang} differs from {ref_lang}.\n"
                f"  missing from {lang}: {missing}\n"
                f"  extra in {lang}:     {extra}\n"
                f"The paired design requires identical item sets."
            )
    print(f"item_id sets identical across all {len(langs)} languages: PASS")

    # --- prove the gold answer agrees per item ------------------------------
    gold_by_lang = {
        lang: {compose_item_id(r, chosen): int(r["correct_answer_num"]) for r in rows}
        for lang, rows in loaded.items()
    }
    ref_gold = gold_by_lang[ref_lang]
    for lang, g in gold_by_lang.items():
        bad = [i for i, v in g.items() if ref_gold[i] != v]
        if bad:
            raise SystemExit(
                f"FATAL: gold answer for {len(bad)} item(s) in {lang} disagrees "
                f"with {ref_lang}, e.g. {bad[:5]}. BELEBELE is parallel; a "
                f"disagreement means the alignment assumption is broken."
            )
    print(f"gold answers agree per item across all languages: PASS")

    # --- row order (informational, but it is why we key explicitly) ---------
    order_ref = [compose_item_id(r, chosen) for r in loaded[ref_lang]]
    shuffled = [
        lang for lang in langs
        if [compose_item_id(r, chosen) for r in loaded[lang]] != order_ref
    ]
    if shuffled:
        print(
            f"\nNOTE: row order differs from {ref_lang} in: {shuffled}\n"
            f"      Positional pairing would mispair these languages. All joins "
            f"downstream key on item_id explicitly."
        )

    # --- freeze the manifest ------------------------------------------------
    sorted_ids = sorted(ref_ids)
    digest = hashlib.sha256("\n".join(sorted_ids).encode("utf-8")).hexdigest()
    manifest = {
        "dataset": dataset,
        "split": split,
        "revision": revision,
        "languages": langs,
        "reference_language": ref_lang,
        "item_id_key": list(chosen),
        "item_id_separator": ITEM_ID_SEP,
        "n_items": len(sorted_ids),
        "sha256": digest,
        "gold_by_item_id": {i: ref_gold[i] for i in sorted_ids},
        "item_ids": sorted_ids,
    }
    out = config.REPO_ROOT / "configs" / "item_id_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out.relative_to(config.REPO_ROOT)}")
    print(f"  n_items = {len(sorted_ids)}")
    print(f"  sha256  = {digest}")

    key_str = ITEM_ID_SEP.join(chosen)
    print(
        f"\nNow set  benchmark.item_id_key: \"{key_str}\"  in configs/experiment.yaml"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
