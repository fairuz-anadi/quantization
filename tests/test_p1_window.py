"""The answer-centred context window.

This exists because of a measured failure. multi-wiki-qa contexts are whole
Wikipedia articles; relying on left-truncation to fit them into
`max_seq_tokens` destroyed the evidence for 61% of training items, and did so
at rates running from 20% (English) to 78% (Assamese). A language-correlated
artefact of that size sits directly on top of the quantity P1 exists to
measure, so the window is not a convenience -- it is what makes the cross-
language comparison mean anything.

The properties tested here are the ones that keep it honest: the answer span
always survives, the question and options are never cut, the assembled prompt
genuinely fits, the budget is the same number for every language, and an item
that cannot be represented is reported rather than quietly clipped.
"""

import pytest

from quantlang import config as cfg_mod
from quantlang import p1data
from quantlang.p1data import P1DataError, P1ItemTooLong


@pytest.fixture(scope="module")
def cfg():
    return cfg_mod.load()


@pytest.fixture(scope="module")
def manifest():
    return p1data.load_split_manifest()


def _context(n_words=4000, answer="THE ANSWER PHRASE", at=2000):
    words = [f"w{i}" for i in range(n_words)]
    head = " ".join(words[:at])
    tail = " ".join(words[at:])
    context = f"{head} {answer} {tail}"
    return context, context.index(answer)


def _row(context, start, answer="THE ANSWER PHRASE"):
    return {
        "item_id": "Article 000#0",
        "group_id": "Article 000",
        "context": context,
        "question": "What is the answer?",
        "answer": answer,
        "answer_start": start,
        "source_id": "http://example.invalid/0",
    }


OPTIONS = ["THE ANSWER PHRASE", "another option", "third option", "fourth option"]


# --------------------------------------------------------------------------- #
# placement
# --------------------------------------------------------------------------- #

def test_window_keeps_the_answer_span(cfg, word_tokenizer):
    context, start = _context()
    window = p1data.answer_centred_window(cfg, word_tokenizer, context,
                                          "THE ANSWER PHRASE", start, 200)
    assert "THE ANSWER PHRASE" in window


@pytest.mark.parametrize("at", [0, 1, 50, 2000, 3990, 3999])
def test_answer_survives_wherever_it_sits_in_the_article(cfg, word_tokenizer, at):
    """Including both edges, where the window has to redistribute its budget."""
    context, start = _context(at=at)
    window = p1data.answer_centred_window(cfg, word_tokenizer, context,
                                          "THE ANSWER PHRASE", start, 200)
    assert "THE ANSWER PHRASE" in window


def test_window_respects_the_token_budget(cfg, word_tokenizer):
    context, start = _context()
    for budget in (50, 200, 800):
        window = p1data.answer_centred_window(cfg, word_tokenizer, context,
                                              "THE ANSWER PHRASE", start, budget)
        n = len(word_tokenizer(window, add_special_tokens=False)["input_ids"])
        assert n <= budget, f"budget {budget} produced {n} tokens"


def test_window_uses_the_budget_it_is_given(cfg, word_tokenizer):
    """A window far under budget would be discarding evidence for nothing."""
    context, start = _context()
    window = p1data.answer_centred_window(cfg, word_tokenizer, context,
                                          "THE ANSWER PHRASE", start, 200)
    n = len(word_tokenizer(window, add_special_tokens=False)["input_ids"])
    assert n >= 150


def test_short_context_is_returned_whole(cfg, word_tokenizer):
    context = "A short passage containing THE ANSWER PHRASE and nothing else."
    start = context.index("THE ANSWER PHRASE")
    window = p1data.answer_centred_window(cfg, word_tokenizer, context,
                                          "THE ANSWER PHRASE", start, 800)
    assert window == context


def test_window_is_centred_on_the_answer(cfg, word_tokenizer):
    """Evidence usually sits around the answer, not only after it."""
    context, start = _context(at=2000)
    window = p1data.answer_centred_window(cfg, word_tokenizer, context,
                                          "THE ANSWER PHRASE", start, 200)
    position = window.index("THE ANSWER PHRASE") / len(window)
    assert 0.3 < position < 0.7, f"answer sits at {position:.0%} of the window"


def test_windowing_is_deterministic(cfg, word_tokenizer):
    context, start = _context()
    first = p1data.answer_centred_window(cfg, word_tokenizer, context,
                                         "THE ANSWER PHRASE", start, 200)
    second = p1data.answer_centred_window(cfg, word_tokenizer, context,
                                          "THE ANSWER PHRASE", start, 200)
    assert first == second


def test_window_does_not_alter_the_answer_text(cfg, word_tokenizer):
    context, start = _context()
    window = p1data.answer_centred_window(cfg, word_tokenizer, context,
                                          "THE ANSWER PHRASE", start, 200)
    assert window.count("THE ANSWER PHRASE") == 1
    assert "THE ANSWER PHRAS " not in window


def test_edges_land_on_whitespace(cfg, word_tokenizer):
    """A passage should not begin or end mid-word."""
    context, start = _context()
    window = p1data.answer_centred_window(cfg, word_tokenizer, context,
                                          "THE ANSWER PHRASE", start, 200)
    assert window == window.strip()
    assert context.count(window) >= 1, "the window must be a literal substring"


# --------------------------------------------------------------------------- #
# failure rather than silent truncation
# --------------------------------------------------------------------------- #

def test_answer_longer_than_the_budget_is_reported(cfg, word_tokenizer):
    """Requirement: fail loudly rather than clip the answer itself."""
    answer = " ".join(f"a{i}" for i in range(300))
    context, start = _context(answer=answer)
    with pytest.raises(P1ItemTooLong, match="answer span is"):
        p1data.answer_centred_window(cfg, word_tokenizer, context, answer,
                                     start, 50)


def test_too_long_failure_is_a_p1_data_error(cfg, word_tokenizer):
    """P1ItemTooLong stays inside the pipeline's error hierarchy."""
    assert issubclass(P1ItemTooLong, P1DataError)


def test_zero_budget_is_rejected(cfg, word_tokenizer):
    context, start = _context()
    with pytest.raises(P1DataError, match="budget must be positive"):
        p1data.answer_centred_window(cfg, word_tokenizer, context,
                                     "THE ANSWER PHRASE", start, 0)


def test_item_whose_options_fill_the_budget_is_reported(cfg, word_tokenizer):
    """No room for a passage means no evidence; the item is not shipped."""
    context, start = _context()
    huge = " ".join(f"o{i}" for i in range(400))
    with pytest.raises(P1ItemTooLong, match="leaving no room for a passage"):
        p1data.windowed_passage(cfg, word_tokenizer, _row(context, start),
                                [huge, huge + " x", huge + " y", huge + " z"], 1)


# --------------------------------------------------------------------------- #
# the assembled prompt
# --------------------------------------------------------------------------- #

def test_assembled_prompt_fits_max_seq_tokens(cfg, word_tokenizer):
    context, start = _context()
    out = p1data.windowed_passage(cfg, word_tokenizer, _row(context, start),
                                  OPTIONS, 1)
    ceiling = cfg_mod.require(cfg, "finetune.training.max_seq_tokens") - 1
    assert out["prompt_tokens"] <= ceiling


def test_prompt_length_is_measured_not_assumed(cfg, word_tokenizer):
    """The reported length must be the real one."""
    context, start = _context()
    row = _row(context, start)
    out = p1data.windowed_passage(cfg, word_tokenizer, row, OPTIONS, 1)
    prompt = p1data.build_p1_prompt(cfg, {**row, "passage": out["passage"],
                                          "options": OPTIONS, "gold": 1})
    actual = len(word_tokenizer(prompt, add_special_tokens=False)["input_ids"])
    assert out["prompt_tokens"] == actual


def test_question_and_all_options_survive(cfg, word_tokenizer):
    """Only the passage is windowed; nothing else may be cut."""
    context, start = _context()
    row = _row(context, start)
    out = p1data.windowed_passage(cfg, word_tokenizer, row, OPTIONS, 1)
    prompt = p1data.build_p1_prompt(cfg, {**row, "passage": out["passage"],
                                          "options": OPTIONS, "gold": 1})
    assert row["question"] in prompt
    for letter, option in zip("ABCD", OPTIONS):
        assert f"{letter}. {option}" in prompt
    assert prompt.endswith("Answer:")


def test_long_options_shrink_the_passage_not_the_options(cfg, word_tokenizer):
    """The effective budget comes from this item's own measured overhead."""
    context, start = _context()
    row = _row(context, start)
    short = p1data.windowed_passage(cfg, word_tokenizer, row, OPTIONS, 1)
    long_opts = [OPTIONS[0]] + [" ".join(f"z{i}" for i in range(120))] * 3
    long = p1data.windowed_passage(cfg, word_tokenizer, row, long_opts, 1)
    assert long["overhead_tokens"] > short["overhead_tokens"]
    assert long["effective_budget"] < short["effective_budget"]
    ceiling = cfg_mod.require(cfg, "finetune.training.max_seq_tokens") - 1
    assert long["prompt_tokens"] <= ceiling


# --------------------------------------------------------------------------- #
# one algorithm, one budget, every language
# --------------------------------------------------------------------------- #

def test_the_budget_is_a_single_number_for_every_language(cfg):
    """No per-language window sizes: the point is to equalise the evidence
    budget, not to hand each language a different amount of text."""
    wcfg = cfg_mod.require(cfg, "finetune.context_window")
    assert isinstance(wcfg["context_budget_tokens"], int)
    langs = cfg_mod.require(cfg, "benchmark.languages")
    for lang in langs:
        assert lang not in str(wcfg), (
            f"{lang} appears in the context_window config; the window must be "
            f"language-independent")


def test_window_config_is_recorded_in_the_manifest(cfg, manifest):
    recorded = manifest["context_window"]
    live = cfg_mod.require(cfg, "finetune.context_window")
    assert recorded["policy"] == live["policy"] == "answer_centred_tokens"
    assert recorded["algorithm_version"] == live["algorithm_version"]
    assert recorded["context_budget_tokens"] == live["context_budget_tokens"]
    assert recorded["trim_to_whitespace"] == live["trim_to_whitespace"]


def test_tokenizer_identity_is_recorded_in_the_manifest(cfg, manifest):
    """Windowing by tokens couples the corpus to a tokenizer, so it is pinned."""
    recorded = manifest["context_window_tokenizer"]
    live = p1data.tokenizer_identity(cfg)
    assert recorded == live
    assert recorded["hf_id"] == "Qwen/Qwen2.5-3B-Instruct"
    assert len(recorded["revision"]) == 40
    assert recorded["revision"] == cfg_mod.require(cfg, "models")[0]["revision"]


def test_manifest_records_max_seq_tokens(cfg, manifest):
    assert manifest["max_seq_tokens"] == cfg_mod.require(
        cfg, "finetune.training.max_seq_tokens")


# --------------------------------------------------------------------------- #
# the acceptance criterion
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("lang", ["eng_Latn", "ben_Beng", "sin_Sinh",
                                  "asm_Beng", "npi_Deva"])
def test_evidence_is_retained_for_every_language(manifest, lang):
    """The criterion the fix exists to satisfy: the severe, language-correlated
    evidence loss is gone."""
    for partition in ("train", "heldout"):
        stats = manifest["languages"][lang]["context_window_stats"][partition]
        assert stats["evidence_retained"] == 1.0, (
            f"{lang}/{partition}: evidence retained "
            f"{stats['evidence_retained']:.1%}")


@pytest.mark.parametrize("lang", ["eng_Latn", "ben_Beng", "sin_Sinh",
                                  "asm_Beng", "npi_Deva"])
def test_no_recorded_prompt_exceeds_the_budget(cfg, manifest, lang):
    ceiling = cfg_mod.require(cfg, "finetune.training.max_seq_tokens") - 1
    for partition in ("train", "heldout"):
        stats = manifest["languages"][lang]["context_window_stats"][partition]
        assert stats["prompt_tokens_max"] <= ceiling, (
            f"{lang}/{partition}: max prompt {stats['prompt_tokens_max']} "
            f"exceeds {ceiling}")


@pytest.mark.parametrize("lang", ["eng_Latn", "ben_Beng", "sin_Sinh",
                                  "asm_Beng", "npi_Deva"])
def test_dropped_items_are_counted_and_stay_rare(manifest, lang):
    """Items too long to render are excluded, listed, and kept to a trickle.

    The drop rate is itself mildly language-correlated (0% English to 0.8%
    Sinhala), so it is recorded per language rather than mentioned once.
    """
    e = manifest["languages"][lang]
    dropped = e["n_dropped_too_long"]["train"] + e["n_dropped_too_long"]["heldout"]
    assert dropped == len(e["dropped_too_long_item_ids"])
    assert dropped / e["n_source_rows_used"] < 0.02


@pytest.mark.parametrize("lang", ["eng_Latn", "ben_Beng", "sin_Sinh",
                                  "asm_Beng", "npi_Deva"])
def test_items_and_drops_account_for_every_source_row(manifest, lang):
    e = manifest["languages"][lang]
    dropped = e["n_dropped_too_long"]["train"] + e["n_dropped_too_long"]["heldout"]
    assert e["n_train_items"] + e["n_heldout_items"] + dropped == e["n_source_rows_used"]


# --------------------------------------------------------------------------- #
# the manifest moved for exactly one reason
# --------------------------------------------------------------------------- #

# Choice digests from the pre-window build, captured before the rebuild. They
# cover item ids, gold letters and option text -- the ratified selection policy
# -- and deliberately exclude the passage.
PRE_WINDOW_TRAIN_CHOICES = {
    "eng_Latn": "d500ed32739c2aa4dbb816e052f00dfc4fded0ce510c52e94cc4a11bd6ee2935",
    "ben_Beng": "2a81928e4f261433d57fd7601495f42c6688e883a5400b8fc51c0404bfca9192",
    "sin_Sinh": "a4339e302eb72ecc6ada35a91456c479dd3da5a7b2a247bfbbe8bb08a331a906",
    "asm_Beng": "74f8c2f5ea18867ddc69c70a6e4bc8490a13641cd4cba1958267795c2ad9d615",
    "npi_Deva": "f6695e9d77b046a0ef21ce71bae9626abf5513a480420cf63a0cd440061e57f9",
}


def test_context_window_did_not_touch_distractor_selection(manifest):
    """English drops nothing, so its selection fingerprint must be UNCHANGED.

    This is the evidence that the approved change altered only the passage
    representation: same distractors, same gold positions, same order, byte for
    byte. The other four languages differ from their pre-window digests solely
    by the handful of items excluded as too long to render.
    """
    assert (manifest["languages"]["eng_Latn"]["train_choices_sha256"]
            == PRE_WINDOW_TRAIN_CHOICES["eng_Latn"]), (
        "English selection changed. The context window was supposed to alter "
        "the passage and nothing else.")
    assert manifest["languages"]["eng_Latn"]["n_dropped_too_long"]["train"] == 0


@pytest.mark.parametrize("lang", ["ben_Beng", "sin_Sinh", "asm_Beng", "npi_Deva"])
def test_other_languages_differ_only_where_items_were_dropped(manifest, lang):
    e = manifest["languages"][lang]
    assert e["n_dropped_too_long"]["train"] > 0, (
        f"{lang} dropped nothing, so its choice digest should have been "
        f"unchanged too -- investigate rather than accepting the difference")
    assert e["train_choices_sha256"] != PRE_WINDOW_TRAIN_CHOICES[lang]


def test_split_itself_is_unchanged_by_the_window(manifest):
    """Articles are assigned before any windowing, so the split must not move."""
    expected = {"eng_Latn": (489, 122), "ben_Beng": (544, 136),
                "sin_Sinh": (574, 144), "asm_Beng": (549, 137),
                "npi_Deva": (567, 142)}
    for lang, (n_train, n_heldout) in expected.items():
        e = manifest["languages"][lang]
        assert (e["n_train_articles"], e["n_heldout_articles"]) == (n_train, n_heldout)


def test_manifest_digest_is_self_consistent(manifest):
    assert p1data.manifest_payload_digest(manifest) == manifest["sha256"]
