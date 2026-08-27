"""The tidy.csv schema and its invariants.

Every check here is fatal. Nothing in this module repairs, coerces, pads or
drops: a violation means the measurement is not what it claims to be, and the
only safe response is to stop.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import REPO_ROOT, VALID_PRECISIONS

# Exact column set, in exact order, as specified by the project brief.
TIDY_COLUMNS: tuple[str, ...] = (
    "model",
    "model_revision",
    "precision",
    "lang",
    "item_id",
    "pred",
    "gold",
    "correct",
)

# BELEBELE answers are 1-indexed (`correct_answer_num` in 1..4).
VALID_ANSWERS = (1, 2, 3, 4)

MANIFEST_PATH = REPO_ROOT / "configs" / "item_id_manifest.json"


class SchemaError(RuntimeError):
    """Raised when data violates the tidy schema or a design invariant."""


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    path = path or MANIFEST_PATH
    if not path.exists():
        raise SchemaError(
            f"Item manifest missing: {path}\n"
            f"Run `python scripts/verify_item_ids.py` first. Downstream code "
            f"must not infer the item set from whatever a run happened to emit."
        )
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def validate_tidy(df: pd.DataFrame, manifest: dict[str, Any] | None = None) -> None:
    """Validate a tidy frame in full. Raises SchemaError on the first problem."""
    manifest = manifest or load_manifest()
    expected_ids = set(manifest["item_ids"])
    gold_by_id = manifest["gold_by_item_id"]
    n_expected = int(manifest["n_items"])

    # --- columns ------------------------------------------------------------
    if tuple(df.columns) != TIDY_COLUMNS:
        raise SchemaError(
            f"Column mismatch.\n  expected: {list(TIDY_COLUMNS)}\n"
            f"  actual:   {list(df.columns)}"
        )

    if df.isnull().any().any():
        bad = df.columns[df.isnull().any()].tolist()
        raise SchemaError(f"Null values present in column(s): {bad}")

    # --- value domains ------------------------------------------------------
    bad_prec = sorted(set(df["precision"]) - set(VALID_PRECISIONS))
    if bad_prec:
        raise SchemaError(
            f"Invalid precision value(s) {bad_prec}. Allowed: "
            f"{list(VALID_PRECISIONS)}. Bare 'int4'/'int8' are never valid."
        )

    for col in ("pred", "gold"):
        bad = sorted(set(df[col]) - set(VALID_ANSWERS))
        if bad:
            raise SchemaError(f"Column '{col}' has out-of-range value(s): {bad}")

    bad_correct = sorted(set(df["correct"]) - {0, 1})
    if bad_correct:
        raise SchemaError(f"Column 'correct' must be 0/1, found: {bad_correct}")

    # --- correct must be derived, not asserted ------------------------------
    derived = (df["pred"] == df["gold"]).astype(int)
    mismatch = int((derived != df["correct"]).sum())
    if mismatch:
        raise SchemaError(
            f"{mismatch} row(s) where correct != (pred == gold). The scored "
            f"outcome disagrees with the recorded prediction."
        )

    # --- gold must match the frozen manifest --------------------------------
    mapped = df["item_id"].map(gold_by_id)
    if mapped.isnull().any():
        unknown = sorted(set(df.loc[mapped.isnull(), "item_id"]))[:5]
        raise SchemaError(
            f"item_id(s) not present in the frozen manifest, e.g. {unknown}. "
            f"These items are not part of the pinned benchmark."
        )
    bad_gold = int((mapped != df["gold"]).sum())
    if bad_gold:
        raise SchemaError(
            f"{bad_gold} row(s) whose gold disagrees with the frozen manifest. "
            f"This usually means answer options were reordered during scoring."
        )

    # --- per-cell completeness ---------------------------------------------
    for (model, prec, lang), grp in df.groupby(["model", "precision", "lang"], sort=True):
        cell = f"({model}, {prec}, {lang})"
        if grp["item_id"].duplicated().any():
            dups = sorted(grp.loc[grp["item_id"].duplicated(), "item_id"])[:5]
            raise SchemaError(f"Duplicate item_id(s) in cell {cell}: {dups}")
        got = set(grp["item_id"])
        if got != expected_ids:
            raise SchemaError(
                f"Cell {cell} has {len(got)} items, expected {n_expected} "
                f"matching the manifest exactly.\n"
                f"  missing: {len(expected_ids - got)}  extra: {len(got - expected_ids)}\n"
                f"An incomplete cell is never padded or partially reported."
            )
        if len(grp["model_revision"].unique()) != 1:
            raise SchemaError(
                f"Cell {cell} mixes model revisions: "
                f"{sorted(grp['model_revision'].unique())}"
            )

    # --- cross-language parity (the reason the paired design works) ---------
    for (model, prec), grp in df.groupby(["model", "precision"], sort=True):
        per_lang = {lang: set(g["item_id"]) for lang, g in grp.groupby("lang")}
        langs = sorted(per_lang)
        for lang in langs[1:]:
            if per_lang[lang] != per_lang[langs[0]]:
                raise SchemaError(
                    f"({model}, {prec}): item sets differ between {langs[0]} and "
                    f"{lang}. The paired bootstrap requires identical item sets."
                )
