"""BELEBELE loading, item keying, and prompt construction.

Two rules are enforced here rather than trusted:

1. Items are keyed by `link#question_number`, never by row position. BELEBELE's
   row order is NOT aligned across languages -- on the pinned revision, zero of
   Assamese's 900 rows sit at the same index as their English counterpart, and
   580 of Bangla's differ. Positional pairing would silently compare unrelated
   questions, which is precisely what the cross-language interaction test must
   not do.
2. Every language is checked against the frozen manifest at load time. A count
   that is not 900, an unknown item_id, or a gold answer that disagrees with the
   manifest stops the run. Nothing is padded, dropped, or resampled.
"""

from __future__ import annotations

from typing import Any

from datasets import load_dataset

from . import config as cfg_mod
from .schema import SchemaError, load_manifest


class DataError(RuntimeError):
    """Raised when the benchmark does not match its frozen contract."""


def compose_item_id(row: dict[str, Any], key: list[str], sep: str) -> str:
    return sep.join(str(row[c]) for c in key)


def load_language(cfg: dict[str, Any], lang: str, manifest: dict | None = None) -> list[dict]:
    """Load one BELEBELE language at the pinned revision, keyed and verified."""
    manifest = manifest or load_manifest()
    dataset = cfg_mod.require(cfg, "benchmark.hf_dataset")
    split = cfg_mod.require(cfg, "benchmark.split")
    revision = cfg_mod.require(cfg, "benchmark.hf_revision")
    expected_n = cfg_mod.require(cfg, "benchmark.n_items_per_lang")

    key = manifest["item_id_key"]
    sep = manifest["item_id_separator"]
    gold_by_id = manifest["gold_by_item_id"]

    ds = load_dataset(dataset, lang, split=split, revision=revision)
    if len(ds) != expected_n:
        raise DataError(
            f"{lang}: {len(ds)} items, expected {expected_n}. Refusing to "
            f"continue with a benchmark that is not the pinned one."
        )

    rows: list[dict] = []
    seen: set[str] = set()
    for r in ds:
        item_id = compose_item_id(r, key, sep)
        if item_id in seen:
            raise DataError(f"{lang}: duplicate item_id {item_id!r}")
        seen.add(item_id)
        if item_id not in gold_by_id:
            raise DataError(
                f"{lang}: item_id {item_id!r} is absent from the frozen "
                f"manifest. This is not the pinned benchmark."
            )
        gold = int(r["correct_answer_num"])
        if gold != gold_by_id[item_id]:
            raise DataError(
                f"{lang}: gold for {item_id!r} is {gold}, manifest says "
                f"{gold_by_id[item_id]}. Answer options may have been reordered."
            )
        rows.append({
            "item_id": item_id,
            "gold": gold,                      # 1-indexed, as BELEBELE ships it
            "passage": r["flores_passage"],
            "question": r["question"],
            "options": [r[f"mc_answer{i}"] for i in (1, 2, 3, 4)],
        })

    if seen != set(manifest["item_ids"]):
        raise SchemaError(
            f"{lang}: item_id set differs from the frozen manifest "
            f"({len(seen - set(manifest['item_ids']))} extra, "
            f"{len(set(manifest['item_ids']) - seen)} missing)."
        )
    # Sort by item_id so every language iterates in the SAME order. This is a
    # convenience for reading logs; nothing downstream may rely on it, because
    # correctness comes from the key, not the order.
    rows.sort(key=lambda r: r["item_id"])
    return rows


def build_prompt(cfg: dict[str, Any], row: dict) -> str:
    template = cfg_mod.require(cfg, "scoring.prompt_template")
    a1, a2, a3, a4 = (o.strip() for o in row["options"])
    return template.format(
        passage=row["passage"].strip(),
        question=row["question"].strip(),
        a1=a1, a2=a2, a3=a3, a4=a4,
    ).rstrip("\n")
