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
    """Synthetic corpus shaped like multi-wiki-qa: one context per article.

    Every question about an article shares that article's context, and every
    answer appears verbatim inside it. That is what version 2 requires: the
    distractors are the answers to the article's OTHER questions, so they have
    to be locatable in the same passage. Answers vary in shape so the surface
    preference is exercised.
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
        answers = [shapes[q % len(shapes)].format(a=a, q=q)
                   for q in range(per_article)]
        # One shared context per article, carrying every answer, with filler
        # between the facts so the window has something to expand into and the
        # covering span is not trivially the whole article.
        parts = [f"Topic{a % 7} matters here. Article {a:03d} covers "
                 f"subject{a} and term{a % 11}."]
        for q, ans in enumerate(answers):
            parts.append(f"filler{a} word{q} padding{q} more{q} text{q}.")
            parts.append(f"Fact {q}: {ans} ends it.")
        context = " ".join(parts)
        for q, ans in enumerate(answers):
            out.append({
                "group_id": f"Article {a:03d}",
                "context": context,
                "question": f"question {q} about article {a}?",
                "answer": ans,
                "source_id": f"http://example.invalid/{a}",
                "ordinal": q,
                "item_id": f"Article {a:03d}#{q}",
                "answer_start": context.index(ans),
            })
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

def test_every_distractor_comes_from_the_items_own_article(built):
    """The v1 rule INVERTED, and the inversion is the whole repair.

    v1 excluded same-article answers, on the grounds that a fact from the same
    context might be a second defensible answer. The cost of that rule was far
    worse than the risk it avoided: a distractor from another article is not in
    this passage, so the gold became the only option present and the item was
    solvable by substring presence alone.
    """
    for partition in ("train", "heldout"):
        for it in built[partition]:
            for did in it["distractor_item_ids"]:
                assert did.rsplit("#", 1)[0] == it["group_id"], (
                    f"{it['item_id']}: distractor {did!r} is from another "
                    f"article, so its text is not in this passage")


def test_every_option_appears_verbatim_in_the_passage(built):
    """THE anti-shortcut invariant, checked on built items rather than assumed."""
    for partition in ("train", "heldout"):
        for it in built[partition]:
            for i, opt in enumerate(it["options"], start=1):
                assert opt in it["passage"], (
                    f"{it['item_id']}: option {i} ({opt!r}) is not in the "
                    f"passage, so presence identifies the gold without reading")


def test_the_substring_heuristic_is_worth_exactly_a_guess(built):
    """v1 scored ~0.96 here. Anything above chance is a shortcut."""
    for partition in ("train", "heldout"):
        acc = p1data.lexical_shortcut_accuracy(built[partition])
        assert abs(acc - 0.25) < 1e-9, (
            f"{partition}: 'choose the option that appears in the passage' "
            f"scores {acc:.4f}, not chance")


def test_gold_and_distractors_are_present_at_the_same_rate(built):
    """The asymmetry, stated directly: v1 was 100% against ~10%."""
    for partition in ("train", "heldout"):
        d = p1data.construction_diagnostics(built[partition], [],
                                            len(built[partition]))
        assert d["gold_in_context_rate"] == 1.0
        assert d["distractor_in_context_rate"] == 1.0


def test_an_absent_option_is_rejected_by_the_validator(cfg, built):
    """The invariant is enforced, not merely satisfied by luck."""
    it = dict(built["train"][0])
    it["passage"] = it["passage"].replace(it["options"][0], "REDACTED", 1)
    with pytest.raises(P1DataError, match="not verbatim in the passage"):
        p1data.assert_items_wellformed(cfg, [it])


def test_a_cross_article_distractor_is_rejected_by_the_validator(cfg, built):
    it = dict(built["train"][0])
    it["distractor_item_ids"] = ["Some Other Article#0"] + \
        list(it["distractor_item_ids"][1:])
    with pytest.raises(P1DataError, match="rather than from the item's own"):
        p1data.assert_items_wellformed(cfg, [it])


def test_training_distractors_never_come_from_held_out_articles(built):
    """Held-out data must not enter training, not even as an option string.

    Under v2 this holds by construction -- distractors are same-article and the
    split is grouped by article -- but it is the property that actually matters,
    so it is still asserted rather than argued.
    """
    heldout = set(built["heldout_groups"])
    for it in built["train"]:
        for did in it["distractor_item_ids"]:
            assert did.rsplit("#", 1)[0] not in heldout, (
                f"{it['item_id']}: distractor {did!r} came from a held-out article")


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
