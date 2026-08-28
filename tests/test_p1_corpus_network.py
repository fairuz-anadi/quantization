"""Corpus checks that need the network. Opt-in.

These download multi-wiki-qa and BELEBELE, so they are skipped by default and
the everyday suite stays offline and fast. They exist because two of P1's
load-bearing claims are empirical and would otherwise live only in a comment:

  1. the pinned dataset really has the five configs and row counts the frozen
     split manifest was built from;
  2. multi-wiki-qa text does not appear inside BELEBELE evaluation passages --
     the check that disqualified SIB-200, whose English training sentences
     overlap BELEBELE passages 85.6%.

    QUANTLANG_NETWORK_TESTS=1 python -m pytest tests/test_p1_corpus_network.py
"""

import os
import re

import pytest

from quantlang import config as cfg_mod
from quantlang import p1data

pytestmark = pytest.mark.skipif(
    os.environ.get("QUANTLANG_NETWORK_TESTS") != "1",
    reason="network test; set QUANTLANG_NETWORK_TESTS=1 to run",
)

LANGS = ("eng_Latn", "ben_Beng", "sin_Sinh", "asm_Beng", "npi_Deva")

# Sentences shorter than this are common phrases and match by coincidence.
MIN_SENTENCE_CHARS = 40


@pytest.fixture(scope="module")
def cfg():
    return cfg_mod.load()


@pytest.mark.parametrize("lang", LANGS)
def test_pinned_corpus_matches_the_frozen_manifest(cfg, lang):
    """The split reproduces exactly from the pinned revision."""
    manifest = p1data.load_split_manifest()
    built = p1data.build_language(cfg, lang)
    p1data.verify_against_manifest(built, manifest)

    entry = manifest["languages"][lang]
    assert built["report"]["n_source_rows_total"] == entry["n_source_rows_total"]
    assert len(built["train_items"]) == entry["n_train_items"]
    assert len(built["heldout_items"]) == entry["n_heldout_items"]


@pytest.mark.parametrize("lang", LANGS)
def test_every_language_config_exists_with_the_expected_schema(cfg, lang):
    rows, report = p1data.load_source_rows(cfg, lang)
    assert report["config"] == cfg_mod.require(cfg, "finetune.lang_configs")[lang]
    assert rows, f"{lang} yielded no usable rows"
    for field in ("group_id", "context", "question", "answer", "item_id"):
        assert rows[0][field], f"{lang}: row is missing {field}"


def test_multi_wiki_qa_does_not_overlap_belebele_passages(cfg):
    """The check that chose this corpus over SIB-200.

    SIB-200 fails this badly -- it is built from the same FLORES sentences as
    BELEBELE. If multi-wiki-qa ever started overlapping, fine-tuning would be
    training on the evaluation text and the P1 result would be meaningless.
    """
    from datasets import load_dataset

    belebele = load_dataset(
        cfg_mod.require(cfg, "benchmark.hf_dataset"), "eng_Latn",
        split=cfg_mod.require(cfg, "benchmark.split"),
        revision=cfg_mod.require(cfg, "benchmark.hf_revision"),
    )
    passages = sorted({r["flores_passage"] for r in belebele})
    blob = "\n".join(passages)

    rows, _ = p1data.load_source_rows(cfg, "eng_Latn")
    contexts = sorted({r["context"] for r in rows})

    sentences = [
        s.strip()
        for ctx in contexts
        for s in re.split(r"(?<=[.!?])\s+", ctx)
        if len(s.strip()) >= MIN_SENTENCE_CHARS
    ]
    assert sentences, "no sentences long enough to test"

    hits = [s for s in sentences if s in blob]
    assert not hits, (
        f"{len(hits)} of {len(sentences)} multi-wiki-qa context sentences appear "
        f"verbatim in BELEBELE passages, e.g. {hits[:2]}. Fine-tuning on this "
        f"corpus would train on evaluation text."
    )

    reverse = [p for p in passages if p and p in "\n".join(contexts)]
    assert not reverse, (
        f"{len(reverse)} BELEBELE passage(s) appear inside multi-wiki-qa contexts.")


def test_articles_carry_several_questions_each(cfg):
    """The fact that makes a row-level split invalid, asserted rather than
    remembered."""
    rows, _ = p1data.load_source_rows(cfg, "eng_Latn")
    per_article: dict[str, int] = {}
    for r in rows:
        per_article[r["group_id"]] = per_article.get(r["group_id"], 0) + 1
    mean = len(rows) / len(per_article)
    multi = sum(1 for v in per_article.values() if v > 1)
    assert mean > 2.0, f"mean questions per article is {mean:.1f}"
    assert multi / len(per_article) > 0.8, (
        f"only {multi}/{len(per_article)} articles carry more than one question; "
        f"if this ever stopped holding, row-level splitting would be less "
        f"dangerous -- but the grouped split stays correct either way."
    )
