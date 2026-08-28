"""MCQ construction: four options, one correct, no leakage, reproducible.

An item that is answerable without reading the passage teaches a shortcut, and
an item with two defensible answers teaches noise. Both are silent failures --
training completes, loss falls, and the resulting model is measuring something
other than reading comprehension in the target language. The checks here are
what stand between that and the GPU.

All offline: items are built from synthetic articles, so the properties are
tested directly rather than sampled from one corpus.
"""

import pytest

from quantlang import config as cfg_mod
from quantlang import p1data
from quantlang.p1data import P1DataError


@pytest.fixture(scope="module")
def cfg():
    return cfg_mod.load()


def _rows(n_articles=60, per_article=5):
    """Synthetic corpus with distinct per-article vocabulary.

    Contexts differ in wording so the tf-idf neighbour ranking has something
    real to rank; answers vary in shape so surface matching is exercised.
    """
    shapes = [
        "answer text {a}-{q}",
        "{a}{q} March 19{a:02d}",
        "Name Surname {a}-{q}",
        "a rather longer phrase describing outcome {a} number {q}",
        "{a}.{q}",
    ]
    out = []
    for a in range(n_articles):
        for q in range(per_article):
            out.append({
                "group_id": f"Article {a:03d}",
                "context": (f"Topic{a % 7} matters here. Article {a:03d} covers "
                            f"subject{a} with detail{a}{q} and term{a % 11}."),
                "question": f"question {q} about article {a}?",
                "answer": shapes[q % len(shapes)].format(a=a, q=q),
                "source_id": f"http://example.invalid/{a}",
                "ordinal": q,
                "item_id": f"Article {a:03d}#{q}",
            })
    # The answer must be locatable in the context: the window is centred on it.
    for rec in out:
        rec["context"] = f"{rec['context']} Fact: {rec['answer']} ends it."
        rec["answer_start"] = rec["context"].index(rec["answer"])
    return out


@pytest.fixture(scope="module")
def built(cfg, word_tokenizer):
    """Built with the offline stub tokenizer, so this file needs no network."""
    rows = _rows()
    train_groups, heldout_groups = p1data.grouped_split(cfg, "eng_Latn", rows)
    train, train_dropped = p1data.build_mcq_items(
        cfg, "eng_Latn", rows, train_groups, "train", word_tokenizer)
    heldout, heldout_dropped = p1data.build_mcq_items(
        cfg, "eng_Latn", rows, heldout_groups, "heldout", word_tokenizer)
    assert not train_dropped and not heldout_dropped, (
        "the synthetic corpus is short enough that nothing should be dropped")
    return {
        "rows": rows,
        "train_groups": train_groups,
        "heldout_groups": heldout_groups,
        "train": train,
        "heldout": heldout,
    }


# --------------------------------------------------------------------------- #
# shape
# --------------------------------------------------------------------------- #

def test_every_item_has_exactly_four_options(cfg, built):
    n = cfg_mod.require(cfg, "finetune.n_options")
    assert n == 4
    for it in built["train"]:
        assert len(it["options"]) == n


def test_every_item_has_exactly_one_correct_option(built):
    for it in built["train"]:
        gold_norm = p1data.normalise_answer(it["gold_text"])
        matches = [o for o in it["options"]
                   if p1data.normalise_answer(o) == gold_norm]
        assert len(matches) == 1, f"{it['item_id']}: {len(matches)} correct options"


def test_gold_index_points_at_the_source_answer(built):
    for it in built["train"]:
        chosen = it["options"][it["gold"] - 1]
        assert p1data.normalise_answer(chosen) == p1data.normalise_answer(
            it["gold_text"])


def test_gold_is_one_indexed_like_belebele(built):
    for it in built["train"]:
        assert it["gold"] in (1, 2, 3, 4)


def test_options_within_an_item_are_distinct(built):
    for it in built["train"]:
        norms = [p1data.normalise_answer(o) for o in it["options"]]
        assert len(set(norms)) == len(norms), f"{it['item_id']}: duplicate options"


def test_gold_position_is_not_constant(built):
    """A fixed gold position would let the model learn 'always answer A'."""
    positions = {it["gold"] for it in built["train"]}
    assert positions == {1, 2, 3, 4}
    counts = {p: sum(1 for it in built["train"] if it["gold"] == p)
              for p in (1, 2, 3, 4)}
    n = len(built["train"])
    for p, c in counts.items():
        assert 0.15 < c / n < 0.35, f"gold position {p} appears {c / n:.1%} of the time"


# --------------------------------------------------------------------------- #
# leakage
# --------------------------------------------------------------------------- #

def test_no_distractor_comes_from_the_items_own_article(built):
    """A same-article answer can be a second defensible answer to the context."""
    for it in built["train"]:
        for did in it["distractor_item_ids"]:
            assert did.rsplit("#", 1)[0] != it["group_id"], (
                f"{it['item_id']}: distractor {did!r} is from the same article")


def test_training_distractors_never_come_from_held_out_articles(built):
    """Held-out data must not enter training, not even as an option string."""
    heldout = set(built["heldout_groups"])
    for it in built["train"]:
        for did in it["distractor_item_ids"]:
            assert did.rsplit("#", 1)[0] not in heldout, (
                f"{it['item_id']}: distractor {did!r} came from a held-out article")


def test_heldout_distractors_never_come_from_training_articles(built):
    """And the reverse: a held-out item built from trained-on strings is not
    clean held-out data."""
    train = set(built["train_groups"])
    for it in built["heldout"]:
        for did in it["distractor_item_ids"]:
            assert did.rsplit("#", 1)[0] not in train, (
                f"{it['item_id']}: distractor {did!r} came from a train article")


def test_no_belebele_item_id_appears_in_the_p1_corpus():
    """P1 and P0 keep disjoint item namespaces, so a mix-up cannot go unnoticed.

    This is a namespace check, not a text-overlap check. The substantive
    verification -- that multi-wiki-qa text does not appear in BELEBELE
    passages -- needs the corpus and lives in tests/test_p1_corpus_network.py.
    """
    from quantlang import schema
    belebele = set(schema.load_manifest()["item_ids"])
    manifest = p1data.load_split_manifest()
    for lang, e in manifest["languages"].items():
        assert not belebele & set(e["heldout_eval_item_ids"]), lang


def test_p1_does_not_train_on_belebele(cfg):
    """The training corpus is not the benchmark, and cannot be set to it."""
    train_ds = cfg_mod.require(cfg, "finetune.train_dataset")
    bench_ds = cfg_mod.require(cfg, "benchmark.hf_dataset")
    assert train_ds != bench_ds
    assert "belebele" not in train_ds.lower()


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #

def test_items_are_reproducible(cfg, built, word_tokenizer):
    """Same inputs, same items -- option text and order included."""
    again, _ = p1data.build_mcq_items(cfg, "eng_Latn", built["rows"],
                                      built["train_groups"], "train",
                                      word_tokenizer)
    assert p1data.items_digest(again) == p1data.items_digest(built["train"])
    for a, b in zip(again, built["train"]):
        assert a["options"] == b["options"]
        assert a["gold"] == b["gold"]


def test_digest_changes_when_an_option_changes(built):
    """The manifest digest has to be sensitive, or it guarantees nothing."""
    baseline = p1data.items_digest(built["train"])
    tampered = [dict(it) for it in built["train"]]
    tampered[0]["options"] = list(tampered[0]["options"])
    tampered[0]["options"][0] = tampered[0]["options"][0] + " (edited)"
    assert p1data.items_digest(tampered) != baseline


def test_digest_changes_when_gold_moves(built):
    baseline = p1data.items_digest(built["train"])
    tampered = [dict(it) for it in built["train"]]
    tampered[0]["gold"] = 1 + (tampered[0]["gold"] % 4)
    assert p1data.items_digest(tampered) != baseline


def test_different_partitions_seed_differently(cfg, built, word_tokenizer):
    """Otherwise a train and a held-out item on the same row would coincide."""
    as_heldout, _ = p1data.build_mcq_items(cfg, "eng_Latn", built["rows"],
                                           built["train_groups"], "heldout",
                                           word_tokenizer)
    assert p1data.items_digest(as_heldout) != p1data.items_digest(built["train"])


# --------------------------------------------------------------------------- #
# the validator itself
# --------------------------------------------------------------------------- #

def test_validator_accepts_wellformed_items(cfg, built):
    p1data.assert_items_wellformed(cfg, built["train"])


def test_validator_rejects_duplicate_options(cfg, built):
    bad = dict(built["train"][0])
    bad["options"] = list(bad["options"])
    bad["options"][(bad["gold"] % 4)] = bad["options"][bad["gold"] - 1]
    with pytest.raises(P1DataError, match="duplicate options"):
        p1data.assert_items_wellformed(cfg, [bad])


def test_validator_rejects_wrong_option_count(cfg, built):
    bad = dict(built["train"][0])
    bad["options"] = list(bad["options"])[:3]
    bad["gold"] = 1
    bad["gold_text"] = bad["options"][0]
    with pytest.raises(P1DataError, match="options, expected 4"):
        p1data.assert_items_wellformed(cfg, [bad])


def test_validator_rejects_gold_letter_pointing_elsewhere(cfg, built):
    bad = dict(built["train"][0])
    bad["gold"] = 1 + (bad["gold"] % 4)
    with pytest.raises(P1DataError, match="not the source answer"):
        p1data.assert_items_wellformed(cfg, [bad])


def test_validator_rejects_out_of_range_gold(cfg, built):
    bad = dict(built["train"][0])
    bad["gold"] = 5
    with pytest.raises(P1DataError, match="out of range"):
        p1data.assert_items_wellformed(cfg, [bad])


# --------------------------------------------------------------------------- #
# rendering through P0's frozen template
# --------------------------------------------------------------------------- #

def test_prompt_is_rendered_by_p0s_frozen_template(cfg, built):
    """A P1 item and a P0 item must be the same object to the scorer."""
    from quantlang import data as p0_data
    it = built["train"][0]
    assert p1data.build_p1_prompt(cfg, it) == p0_data.build_prompt(cfg, it)


def test_prompt_ends_with_the_answer_cue_and_no_trailing_space(cfg, built):
    """letter_logit reads the logit of ' A'; a trailing space would change the
    token the model is actually about to emit."""
    prompt = p1data.build_p1_prompt(cfg, built["train"][0])
    assert prompt.endswith("Answer:")
    assert not prompt.endswith("Answer: ")


def test_prompt_contains_all_four_lettered_options(cfg, built):
    it = built["train"][0]
    prompt = p1data.build_p1_prompt(cfg, it)
    for letter, option in zip("ABCD", it["options"]):
        assert f"{letter}. {option}" in prompt


def test_surface_homogeneity_is_reported(built):
    """Recorded so the give-away rate is auditable rather than assumed."""
    value = p1data.surface_homogeneity(built["train"])
    assert 0.0 <= value <= 1.0
