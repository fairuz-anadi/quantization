"""The P1 80/20 split must be a real partition, grouped by article.

Two layers here, both offline:

  * the FROZEN manifest is checked for internal consistency, so a hand-edited
    or half-rebuilt split is caught without touching the network;
  * the split FUNCTIONS are checked on synthetic articles, so the properties
    that matter -- determinism, whole-article assignment, disjointness -- are
    tested directly rather than inferred from one dataset's output.

The substantive corpus checks (that multi-wiki-qa really has these row counts,
and that its text does not overlap BELEBELE) need the network and are opt-in;
see tests/test_p1_corpus_network.py.
"""

import pytest

from quantlang import config as cfg_mod
from quantlang import p1data
from quantlang.p1data import P1DataError

BAND = (0.70, 0.90)


@pytest.fixture(scope="module")
def cfg():
    return cfg_mod.load()


@pytest.fixture(scope="module")
def manifest():
    return p1data.load_split_manifest()


# --------------------------------------------------------------------------- #
# the frozen manifest
# --------------------------------------------------------------------------- #

def test_manifest_sha256_matches_its_own_contents(manifest):
    """Detects hand-editing, exactly as the BELEBELE manifest's digest does."""
    assert p1data.manifest_payload_digest(manifest) == manifest["sha256"]


def test_manifest_pins_the_dataset_revision_from_the_config(cfg, manifest):
    assert manifest["dataset"] == cfg_mod.require(cfg, "finetune.train_dataset")
    assert manifest["revision"] == cfg_mod.require(cfg, "finetune.hf_revision")
    assert len(manifest["revision"]) == 40, "a revision must be a full commit SHA"


def test_manifest_covers_exactly_the_final_scope(cfg, manifest):
    """The P1 CORPUS is the final scope, not all of benchmark.languages.

    P0 evaluated five languages and those results stand. P0 reads
    configs/item_id_manifest.json and never touches this manifest, so narrowing
    the P1 corpus cannot reach it. Sinhala was additionally measured as
    unbuildable under algorithm_version 2 -- 23.5% of its rows -- and Assamese
    and Nepali were already out of scope.
    """
    scope = cfg_mod.require(cfg, "finetune.final_scope_languages")
    assert set(manifest["languages"]) == set(scope)
    assert manifest["final_scope_languages"] == scope
    known = cfg_mod.require(cfg, "benchmark.languages")
    assert set(scope) <= set(known), (
        "a P1 language must still be one of the frozen P0 languages")


def test_split_is_grouped_by_article_not_by_row(cfg, manifest):
    assert manifest["group_key"] == cfg_mod.require(cfg, "finetune.group_key")
    assert manifest["group_key"], "a grouped split needs a group key"


SCOPE = cfg_mod.load()["finetune"]["final_scope_languages"]


@pytest.mark.parametrize("lang", SCOPE)
class TestPerLanguage:
    """The invariant the whole design rests on: no article on both sides."""

    def test_train_and_heldout_articles_are_disjoint(self, manifest, lang):
        e = manifest["languages"][lang]
        overlap = set(e["train_articles"]) & set(e["heldout_articles"])
        assert not overlap, (
            f"{lang}: {len(overlap)} article(s) in both partitions. Every "
            f"question about an article shares one context, so this leaks the "
            f"context across the train/held-out boundary."
        )

    def test_article_counts_add_up(self, manifest, lang):
        e = manifest["languages"][lang]
        assert len(e["train_articles"]) == e["n_train_articles"]
        assert len(e["heldout_articles"]) == e["n_heldout_articles"]
        assert e["n_train_articles"] + e["n_heldout_articles"] == e["n_articles"]

    def test_every_source_row_lands_in_exactly_one_partition(self, manifest, lang):
        """Nothing is lost silently: every row is an item or a counted drop.

        Two filters run before the corpus is frozen -- rows missing a field or
        an answer span, and rows whose question and four options alone exceed
        max_seq_tokens. Both are counted per language in the manifest, so the
        arithmetic has to close exactly.
        """
        e = manifest["languages"][lang]
        dropped = (e["n_dropped_too_long"]["train"]
                   + e["n_dropped_too_long"]["heldout"])
        assert (e["n_train_items"] + e["n_heldout_items"] + dropped
                == e["n_source_rows_used"])
        assert (e["n_source_rows_used"]
                + e["n_source_rows_dropped_empty"]
                + e["n_source_rows_dropped_no_answer_span"]
                == e["n_source_rows_total"])

    def test_articles_are_unique_within_each_partition(self, manifest, lang):
        e = manifest["languages"][lang]
        for part in ("train_articles", "heldout_articles"):
            assert len(set(e[part])) == len(e[part]), f"{lang}: duplicate in {part}"

    def test_row_fraction_is_near_the_target(self, manifest, lang):
        e = manifest["languages"][lang]
        assert BAND[0] <= e["train_row_fraction"] <= BAND[1], (
            f"{lang}: {e['train_row_fraction']:.1%} of rows in train. The split "
            f"is 80/20 by ARTICLE, so the row fraction only lands near 0.8 if "
            f"questions-per-article is roughly uniform."
        )

    def test_heldout_eval_items_come_only_from_heldout_articles(self, manifest, lang):
        """The secondary surface must not contain a trained-on article.

        item_id is `{article}{sep}{ordinal}`, so the article is recoverable and
        this is checkable directly rather than taken on trust.
        """
        e = manifest["languages"][lang]
        sep = manifest["item_id_separator"]
        heldout = set(e["heldout_articles"])
        train = set(e["train_articles"])
        for item_id in e["heldout_eval_item_ids"]:
            article = item_id.rsplit(sep, 1)[0]
            assert article in heldout, (
                f"{lang}: eval item {item_id!r} belongs to {article!r}, which is "
                f"not a held-out article")
            assert article not in train

    def test_heldout_eval_is_capped_and_unique(self, cfg, manifest, lang):
        e = manifest["languages"][lang]
        cap = cfg_mod.require(cfg, "finetune.heldout_eval_cap")
        ids = e["heldout_eval_item_ids"]
        assert len(set(ids)) == len(ids), f"{lang}: duplicate eval item ids"
        assert len(ids) == min(cap, e["n_heldout_items"])

    def test_heldout_eval_gold_is_complete_and_in_range(self, manifest, lang):
        e = manifest["languages"][lang]
        gold = e["heldout_eval_gold"]
        assert set(gold) == set(e["heldout_eval_item_ids"])
        assert set(gold.values()) <= {1, 2, 3, 4}


def test_heldout_eval_matches_belebele_n(manifest):
    """Both surfaces carry n=900 so their Wilson intervals are comparable."""
    for lang, e in manifest["languages"].items():
        assert len(e["heldout_eval_item_ids"]) == 900, lang


# --------------------------------------------------------------------------- #
# the split functions, on synthetic articles
# --------------------------------------------------------------------------- #

def _rows(n_articles=40, per_article=5):
    """Synthetic corpus: whole articles, several questions each."""
    out = []
    for a in range(n_articles):
        for q in range(per_article):
            out.append({
                "group_id": f"Article {a:03d}",
                "context": f"Article {a:03d} discusses topic{a} and subject{a}.",
                "question": f"question {q} about article {a}?",
                "answer": f"answer {a}-{q}",
                "source_id": f"http://example.invalid/{a}",
                "ordinal": q,
                "item_id": f"Article {a:03d}#{q}",
            })
    return out


def test_grouped_split_is_deterministic(cfg):
    rows = _rows()
    first = p1data.grouped_split(cfg, "eng_Latn", rows)
    second = p1data.grouped_split(cfg, "eng_Latn", rows)
    assert first == second


def test_grouped_split_partitions_every_article(cfg):
    rows = _rows()
    train, heldout = p1data.grouped_split(cfg, "eng_Latn", rows)
    groups = {r["group_id"] for r in rows}
    assert set(train) | set(heldout) == groups
    assert not set(train) & set(heldout)


def test_grouped_split_never_splits_an_article(cfg):
    """Every row of an article follows the article, which is the whole point."""
    rows = _rows()
    train, _ = p1data.grouped_split(cfg, "eng_Latn", rows)
    train_set = set(train)
    for group in train_set:
        members = [r for r in rows if r["group_id"] == group]
        assert members, group
        assert all(r["group_id"] in train_set for r in members)


def test_grouped_split_respects_the_configured_fraction(cfg):
    rows = _rows(n_articles=100)
    train, heldout = p1data.grouped_split(cfg, "eng_Latn", rows)
    fraction = cfg_mod.require(cfg, "finetune.train_fraction")
    assert abs(len(train) / (len(train) + len(heldout)) - fraction) < 0.02


def test_grouped_split_differs_between_languages(cfg):
    """Seeded per language, so two languages do not get the same assignment."""
    rows = _rows(n_articles=100)
    eng = p1data.grouped_split(cfg, "eng_Latn", rows)[0]
    ben = p1data.grouped_split(cfg, "ben_Beng", rows)[0]
    assert eng != ben


def test_grouped_split_refuses_a_corpus_with_one_article(cfg):
    rows = _rows(n_articles=1, per_article=10)
    with pytest.raises(P1DataError, match="grouped split is"):
        p1data.grouped_split(cfg, "eng_Latn", rows)


def test_both_partitions_stay_non_empty_for_a_tiny_corpus(cfg):
    rows = _rows(n_articles=2, per_article=3)
    train, heldout = p1data.grouped_split(cfg, "eng_Latn", rows)
    assert train and heldout


# --------------------------------------------------------------------------- #
# item ids
# --------------------------------------------------------------------------- #

def test_ordinals_do_not_depend_on_dataset_row_order():
    """A reshuffled source must produce byte-identical item ids.

    The dataset hands rows back in its own order; keying on that order would
    make the split silently depend on it.
    """
    raw = [{k: v for k, v in r.items() if k not in ("ordinal", "item_id")}
           for r in _rows(n_articles=6, per_article=4)]
    forward = p1data.assign_ordinals(raw, "#")
    backward = p1data.assign_ordinals(list(reversed(raw)), "#")
    assert [r["item_id"] for r in forward] == [r["item_id"] for r in backward]


def test_item_ids_are_unique():
    raw = [{k: v for k, v in r.items() if k not in ("ordinal", "item_id")}
           for r in _rows()]
    rows = p1data.assign_ordinals(raw, "#")
    ids = [r["item_id"] for r in rows]
    assert len(set(ids)) == len(ids)


def test_context_spanning_two_articles_is_rejected():
    """Grouping by a label only prevents leakage if the label partitions text."""
    rows = _rows(n_articles=3, per_article=2)
    rows[0]["context"] = rows[-1]["context"]
    with pytest.raises(P1DataError, match="more than one article"):
        p1data.assert_context_grouping("eng_Latn", rows)


def test_context_grouping_accepts_a_clean_corpus():
    p1data.assert_context_grouping("eng_Latn", _rows())
