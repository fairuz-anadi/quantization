"""Schema validator must reject every way a cell can be quietly wrong.

NOTE ON FIXTURES: the frames below are STRUCTURAL fixtures. Predictions are
assigned by a fixed rule so that rows exist at all; no value here is an
accuracy, a degradation, or any other measured quantity, and nothing derived
from these frames may ever appear in the paper. Real numbers come only from
results/raw/.
"""

import pandas as pd
import pytest

from quantlang import schema

MODEL = "qwen2.5-7b-instruct"
REV = "0000000000000000000000000000000000000000"  # structural placeholder revision


def _cell(manifest, lang, precision, all_correct=True):
    ids = manifest["item_ids"]
    gold = [manifest["gold_by_item_id"][i] for i in ids]
    pred = list(gold) if all_correct else [(g % 4) + 1 for g in gold]
    return pd.DataFrame(
        {
            "model": MODEL,
            "model_revision": REV,
            "precision": precision,
            "lang": lang,
            "item_id": ids,
            "pred": pred,
            "gold": gold,
            "correct": [int(p == g) for p, g in zip(pred, gold)],
        }
    )[list(schema.TIDY_COLUMNS)]


@pytest.fixture(scope="module")
def manifest():
    return schema.load_manifest()


@pytest.fixture
def frame(manifest):
    return pd.concat(
        [_cell(manifest, lang, "fp16") for lang in ("eng_Latn", "ben_Beng")],
        ignore_index=True,
    )


def test_valid_frame_passes(frame, manifest):
    schema.validate_tidy(frame, manifest)


def test_rejects_wrong_column_order(frame, manifest):
    shuffled = frame[list(reversed(schema.TIDY_COLUMNS))]
    with pytest.raises(schema.SchemaError, match="Column mismatch"):
        schema.validate_tidy(shuffled, manifest)


def test_rejects_short_cell(frame, manifest):
    """A cell with fewer items than the manifest must crash, never be reported."""
    truncated = frame.iloc[:-1]
    with pytest.raises(schema.SchemaError, match="expected 900"):
        schema.validate_tidy(truncated, manifest)


def test_rejects_duplicate_item(frame, manifest):
    dup = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(schema.SchemaError, match="Duplicate item_id"):
        schema.validate_tidy(dup, manifest)


def test_rejects_correct_inconsistent_with_pred_gold(frame, manifest):
    tampered = frame.copy()
    tampered.loc[0, "correct"] = 1 - tampered.loc[0, "correct"]
    with pytest.raises(schema.SchemaError, match="correct != "):
        schema.validate_tidy(tampered, manifest)


def test_rejects_gold_disagreeing_with_manifest(frame, manifest):
    """Catches answer-option reordering during scoring."""
    tampered = frame.copy()
    row = tampered.index[0]
    tampered.loc[row, "gold"] = (tampered.loc[row, "gold"] % 4) + 1
    tampered.loc[row, "correct"] = int(
        tampered.loc[row, "pred"] == tampered.loc[row, "gold"]
    )
    with pytest.raises(schema.SchemaError, match="disagrees with the frozen manifest"):
        schema.validate_tidy(tampered, manifest)


def test_rejects_unknown_item_id(frame, manifest):
    tampered = frame.copy()
    tampered.loc[0, "item_id"] = "https://example.invalid/not-a-real-passage#1"
    with pytest.raises(schema.SchemaError, match="not present in the frozen manifest"):
        schema.validate_tidy(tampered, manifest)


@pytest.mark.parametrize("bad", ["int8", "int4", "fp32"])
def test_rejects_bare_int_precision_naming(frame, manifest, bad):
    tampered = frame.copy()
    tampered["precision"] = bad
    with pytest.raises(schema.SchemaError, match="Invalid precision"):
        schema.validate_tidy(tampered, manifest)


def test_rejects_mismatched_item_sets_across_languages(manifest):
    """The paired bootstrap is invalid unless both languages cover the same items."""
    eng = _cell(manifest, "eng_Latn", "fp16")
    ben = _cell(manifest, "ben_Beng", "fp16").iloc[:-1]
    combined = pd.concat([eng, ben], ignore_index=True)
    with pytest.raises(schema.SchemaError):
        schema.validate_tidy(combined, manifest)


def test_rejects_mixed_model_revisions_in_a_cell(frame, manifest):
    tampered = frame.copy()
    tampered.loc[0, "model_revision"] = "f" * 40
    with pytest.raises(schema.SchemaError, match="mixes model revisions"):
        schema.validate_tidy(tampered, manifest)


def test_rejects_nulls(frame, manifest):
    tampered = frame.copy()
    tampered.loc[0, "pred"] = None
    with pytest.raises(schema.SchemaError, match="Null values"):
        schema.validate_tidy(tampered, manifest)
