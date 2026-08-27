"""The frozen item manifest is the contract every run is checked against."""

import hashlib

from quantlang import config, schema


def test_manifest_exists_and_has_900_items():
    m = schema.load_manifest()
    cfg = config.load()
    assert m["n_items"] == cfg["benchmark"]["n_items_per_lang"] == 900
    assert len(m["item_ids"]) == 900


def test_item_ids_are_unique():
    m = schema.load_manifest()
    assert len(set(m["item_ids"])) == len(m["item_ids"])


def test_question_number_alone_would_not_have_keyed_items():
    """Regression guard for the bug the brief would have shipped."""
    m = schema.load_manifest()
    tails = {i.rsplit(m["item_id_separator"], 1)[-1] for i in m["item_ids"]}
    assert len(tails) < 10, "sanity: question_number is low-cardinality"
    assert m["item_id_key"] == ["link", "question_number"]


def test_sha256_matches_contents():
    """Detects hand-editing of the manifest."""
    m = schema.load_manifest()
    digest = hashlib.sha256("\n".join(sorted(m["item_ids"])).encode()).hexdigest()
    assert digest == m["sha256"]


def test_every_item_has_gold_in_range():
    m = schema.load_manifest()
    gold = m["gold_by_item_id"]
    assert set(gold) == set(m["item_ids"])
    assert set(gold.values()) <= set(schema.VALID_ANSWERS)
