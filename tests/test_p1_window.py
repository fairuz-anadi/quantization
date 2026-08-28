"""The span-covering context window.  (algorithm_version 2)

Windowing exists at all because of a measured failure: multi-wiki-qa contexts
are whole Wikipedia articles, and relying on left-truncation to fit them into
`max_seq_tokens` destroyed the evidence for most training items at severely
language-correlated rates. That reason is unchanged.

WHERE the window goes changed, and this file is mostly about why. Version 1
centred it on the gold answer span, which guaranteed the gold was in the passage
while distractors drawn from other articles were not. The item was then solvable
by "which option appears verbatim in the passage?" -- worth ~0.96 in English and
~0.92 in Bangla -- and nothing in the pipeline measured it, because every check
was structural. Version 2 places the window over the span covering ALL FOUR
option strings, so presence is constant across options and the heuristic is
worth exactly a guess.

The properties tested here are the ones that keep that honest: every required
span survives, the window expands around them rather than clipping to them, the
question and options are never cut, the assembled prompt genuinely fits, the
budget is the same number for every language, an item that cannot show all four
options is reported rather than quietly clipped -- and, at the corpus level, the
substring heuristic is worth 0.25 in BOTH languages, with drop rates that do not
diverge between them.
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


SPAN_TEXTS = ["THE GOLD PHRASE", "FIRST DISTRACTOR", "SECOND DISTRACTOR",
              "THIRD DISTRACTOR"]


def _context(n_words=4000, at=2000, spread=6):
    """An article with the four option strings planted `spread` words apart.

    The gold is planted first, at `at`; the three distractors follow. `spread`
    controls how far the covering window has to reach, which is the quantity the
    whole construction is bounded by.
    """
    words = [f"w{i}" for i in range(n_words)]
    positions = [at + i * spread for i in range(len(SPAN_TEXTS))]
    for text, pos in zip(SPAN_TEXTS, positions):
        words.insert(min(pos, len(words)), text)
    context = " ".join(words)
    spans = sorted((context.index(t), context.index(t) + len(t))
                   for t in SPAN_TEXTS)
    return context, spans


def _span_texts(context, spans):
    return [context[lo:hi] for lo, hi in sorted(spans)]


def _row(context, spans, answer=SPAN_TEXTS[0]):
    return {
        "item_id": "Article 000#0",
        "group_id": "Article 000",
        "context": context,
        "question": "What is the answer?",
        "answer": answer,
        "answer_start": context.index(answer),
        "source_id": "http://example.invalid/0",
    }


OPTIONS = list(SPAN_TEXTS)


# --------------------------------------------------------------------------- #
# placement
# --------------------------------------------------------------------------- #

def test_window_keeps_every_required_span(cfg, word_tokenizer):
    context, spans = _context()
    window = p1data.span_covering_window(cfg, word_tokenizer, context, spans, 200)
    for text in _span_texts(context, spans):
        assert text in window


@pytest.mark.parametrize("at", [0, 1, 50, 2000, 3900, 3980])
def test_spans_survive_wherever_they_sit_in_the_article(cfg, word_tokenizer, at):
    """Including both edges, where the window has to redistribute its budget."""
    context, spans = _context(at=at)
    window = p1data.span_covering_window(cfg, word_tokenizer, context, spans, 200)
    for text in _span_texts(context, spans):
        assert text in window


def test_window_respects_the_token_budget(cfg, word_tokenizer):
    context, spans = _context()
    for budget in (50, 200, 800):
        window = p1data.span_covering_window(cfg, word_tokenizer, context,
                                             spans, budget)
        n = len(word_tokenizer(window, add_special_tokens=False)["input_ids"])
        assert n <= budget, f"budget {budget} produced {n} tokens"


def test_window_uses_the_budget_it_is_given(cfg, word_tokenizer):
    """A window far under budget would be discarding evidence for nothing."""
    context, spans = _context()
    window = p1data.span_covering_window(cfg, word_tokenizer, context, spans, 200)
    n = len(word_tokenizer(window, add_special_tokens=False)["input_ids"])
    assert n >= 150


def test_short_context_is_returned_whole(cfg, word_tokenizer):
    context = "A short passage with GOLD SPAN and D1 SPAN and nothing else."
    spans = [(context.index("GOLD SPAN"), context.index("GOLD SPAN") + 9),
             (context.index("D1 SPAN"), context.index("D1 SPAN") + 7)]
    window = p1data.span_covering_window(cfg, word_tokenizer, context, spans, 800)
    assert window == context


def test_window_expands_around_the_covered_interval(cfg, word_tokenizer):
    """The window is not clipped to the four spans.

    A window that stopped at the outermost option would put option text at both
    edges of the passage, which is a positional cue in its own right, and would
    leave the model no surrounding prose to read.
    """
    context, spans = _context(at=2000)
    window = p1data.span_covering_window(cfg, word_tokenizer, context, spans, 300)
    first, last = _span_texts(context, spans)[0], _span_texts(context, spans)[-1]
    assert not window.startswith(first)
    assert not window.endswith(last)


def test_windowing_is_deterministic(cfg, word_tokenizer):
    context, spans = _context()
    first = p1data.span_covering_window(cfg, word_tokenizer, context, spans, 200)
    second = p1data.span_covering_window(cfg, word_tokenizer, context, spans, 200)
    assert first == second


def test_window_does_not_alter_the_span_text(cfg, word_tokenizer):
    context, spans = _context()
    window = p1data.span_covering_window(cfg, word_tokenizer, context, spans, 200)
    for text in _span_texts(context, spans):
        assert window.count(text) >= 1


def test_edges_land_on_whitespace(cfg, word_tokenizer):
    """A passage should not begin or end mid-word."""
    context, spans = _context()
    window = p1data.span_covering_window(cfg, word_tokenizer, context, spans, 200)
    assert window == window.strip()
    assert context.count(window) >= 1, "the window must be a literal substring"


# --------------------------------------------------------------------------- #
# failure rather than silent truncation
# --------------------------------------------------------------------------- #

def test_spans_wider_than_the_budget_are_reported(cfg, word_tokenizer):
    """Requirement: fail loudly rather than drop an option out of the passage.

    Clipping here would be the version 1 failure re-entering through the back
    door: an option that is not in the passage makes presence informative again.
    """
    context, spans = _context(at=100, spread=3000)
    with pytest.raises(P1ItemTooLong, match="option spans cover"):
        p1data.span_covering_window(cfg, word_tokenizer, context, spans, 50)


def test_too_long_failure_is_a_p1_data_error(cfg, word_tokenizer):
    """P1ItemTooLong stays inside the pipeline's error hierarchy."""
    assert issubclass(P1ItemTooLong, P1DataError)


def test_zero_budget_is_rejected(cfg, word_tokenizer):
    context, spans = _context()
    with pytest.raises(P1DataError, match="budget must be positive"):
        p1data.span_covering_window(cfg, word_tokenizer, context, spans, 0)


def test_item_whose_options_fill_the_budget_is_reported(cfg, word_tokenizer):
    """No room for a passage means no evidence; the item is not shipped."""
    context, spans = _context()
    huge = " ".join(f"o{i}" for i in range(4000))
    with pytest.raises(P1ItemTooLong, match="leaving no room for a passage"):
        p1data.windowed_passage(cfg, word_tokenizer, _row(context, spans),
                                [huge, huge + " x", huge + " y", huge + " z"],
                                1, spans)


# --------------------------------------------------------------------------- #
# the assembled prompt
# --------------------------------------------------------------------------- #

def test_assembled_prompt_fits_max_seq_tokens(cfg, word_tokenizer):
    context, spans = _context()
    out = p1data.windowed_passage(cfg, word_tokenizer, _row(context, spans),
                                  OPTIONS, 1, spans)
    ceiling = cfg_mod.require(cfg, "finetune.training.max_seq_tokens") - 1
    assert out["prompt_tokens"] <= ceiling


def test_prompt_length_is_measured_not_assumed(cfg, word_tokenizer):
    """The reported length must be the real one."""
    context, spans = _context()
    row = _row(context, spans)
    out = p1data.windowed_passage(cfg, word_tokenizer, row, OPTIONS, 1, spans)
    prompt = p1data.build_p1_prompt(cfg, {**row, "passage": out["passage"],
                                          "options": OPTIONS, "gold": 1})
    actual = len(word_tokenizer(prompt, add_special_tokens=False)["input_ids"])
    assert out["prompt_tokens"] == actual


def test_question_and_all_options_survive(cfg, word_tokenizer):
    """Only the passage is windowed; nothing else may be cut."""
    context, spans = _context()
    row = _row(context, spans)
    out = p1data.windowed_passage(cfg, word_tokenizer, row, OPTIONS, 1, spans)
    prompt = p1data.build_p1_prompt(cfg, {**row, "passage": out["passage"],
                                          "options": OPTIONS, "gold": 1})
    assert row["question"] in prompt
    for letter, option in zip("ABCD", OPTIONS):
        assert f"{letter}. {option}" in prompt
    assert prompt.endswith("Answer:")


def test_long_options_shrink_the_passage_not_the_options(cfg, word_tokenizer):
    """The effective budget comes from this item's own measured overhead."""
    context, spans = _context()
    row = _row(context, spans)
    short = p1data.windowed_passage(cfg, word_tokenizer, row, OPTIONS, 1, spans)
    long_opts = [OPTIONS[0]] + [" ".join(f"z{i}" for i in range(500))] * 3
    long = p1data.windowed_passage(cfg, word_tokenizer, row, long_opts, 1, spans)
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
    assert recorded["policy"] == live["policy"] == "span_covering_tokens"
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

# The P1 CORPUS covers the final-scope languages only. P0 evaluated five and
# those results stand; P0 reads configs/item_id_manifest.json and never touches
# the P1 split manifest, so narrowing the P1 corpus cannot reach it. Sinhala was
# additionally measured as unbuildable under algorithm_version 2 -- 23.5% of its
# rows -- which is recorded in configs/experiment.yaml.
SCOPE = cfg_mod.load()["finetune"]["final_scope_languages"]


@pytest.mark.parametrize("lang", SCOPE)
def test_every_option_is_retained_for_every_language(manifest, lang):
    """The criterion the fix exists to satisfy.

    v1's criterion was that the GOLD survived the window, which it did 100% of
    the time -- and that was the bug, because the distractors did not. The
    criterion that matters is that ALL FOUR options survive.
    """
    for partition in ("train", "heldout"):
        stats = manifest["languages"][lang]["context_window_stats"][partition]
        assert stats["all_options_retained"] == 1.0, (
            f"{lang}/{partition}: all four options retained in only "
            f"{stats['all_options_retained']:.1%} of items")
        assert stats["evidence_retained"] == 1.0


@pytest.mark.parametrize("lang", SCOPE)
def test_the_substring_heuristic_is_worth_a_guess_in_every_language(manifest, lang):
    """v1 scored 0.96 (English) and 0.92 (Bangla) here, and nothing measured it.

    This is the single check whose absence let an invalid corpus through a green
    pipeline, so it is asserted on the FROZEN manifest, not just at build time.
    """
    for partition in ("train", "heldout"):
        d = manifest["languages"][lang]["construction_diagnostics"][partition]
        assert abs(d["lexical_shortcut_accuracy"] - 0.25) < 1e-9, (
            f"{lang}/{partition}: substring heuristic scores "
            f"{d['lexical_shortcut_accuracy']:.4f}")
        assert d["gold_in_context_rate"] == d["distractor_in_context_rate"] == 1.0
        assert d["same_article_distractor_rate"] == 1.0


def test_the_shortcut_is_not_language_dependent(manifest):
    """The gap between English and Bangla is what sat on P1's estimand.

    v1's 96%/92% split was itself a language effect, indistinguishable in the
    results from the quantization-by-language interaction P1 exists to measure.
    """
    values = [manifest["languages"][l]["construction_diagnostics"]["train"]
              ["lexical_shortcut_accuracy"] for l in SCOPE]
    assert max(values) - min(values) < 1e-9, (
        f"shortcut accuracy differs across languages: "
        f"{dict(zip(SCOPE, values))}")


@pytest.mark.parametrize("lang", SCOPE)
def test_no_option_position_carries_a_cue(manifest, lang):
    """A span-covering window could leave the gold at a predictable place."""
    for partition in ("train", "heldout"):
        d = manifest["languages"][lang]["construction_diagnostics"][partition]
        letters = d["gold_letter_distribution"]
        assert max(letters.values()) / d["n_items"] < 0.40, (
            f"{lang}/{partition}: gold letters {letters}")
        ranks = d["gold_position"]["share_by_rank"]
        assert max(ranks.values()) < 0.40, (
            f"{lang}/{partition}: gold position in passage {ranks}")


@pytest.mark.parametrize("lang", SCOPE)
def test_no_recorded_prompt_exceeds_the_budget(cfg, manifest, lang):
    ceiling = cfg_mod.require(cfg, "finetune.training.max_seq_tokens") - 1
    for partition in ("train", "heldout"):
        stats = manifest["languages"][lang]["context_window_stats"][partition]
        assert stats["prompt_tokens_max"] <= ceiling, (
            f"{lang}/{partition}: max prompt {stats['prompt_tokens_max']} "
            f"exceeds {ceiling}")


@pytest.mark.parametrize("lang", SCOPE)
def test_dropped_items_are_counted_and_stay_under_the_ceiling(manifest, lang):
    """Items that cannot show all four options are excluded, listed, and capped.

    The rate is genuinely language-correlated -- Bangla articles tokenise ~3.6x
    longer than English ones -- so it is recorded per language and separately
    bounded ACROSS languages by test_drop_rates_do_not_diverge below.
    """
    e = manifest["languages"][lang]
    dropped = e["n_dropped_too_long"]["train"] + e["n_dropped_too_long"]["heldout"]
    assert dropped == len(e["dropped_too_long_item_ids"])
    assert dropped / e["n_source_rows_used"] < p1data.MAX_CONSTRUCTION_DROP_FRACTION


def test_drop_rates_do_not_diverge_between_languages(manifest):
    """A per-language ceiling cannot see this, and it is the thing that matters.

    If one language keeps 99% of its rows and another 75%, the two training sets
    are differently selected samples and "fine-tuning helped English more" is no
    longer separable from "Bangla trained on a different kind of article".
    """
    rates = {}
    for lang in SCOPE:
        e = manifest["languages"][lang]
        dropped = (e["n_dropped_too_long"]["train"]
                   + e["n_dropped_too_long"]["heldout"])
        rates[lang] = dropped / e["n_source_rows_used"]
    assert max(rates.values()) - min(rates.values()) < 0.10, rates


@pytest.mark.parametrize("lang", SCOPE)
def test_items_and_drops_account_for_every_source_row(manifest, lang):
    e = manifest["languages"][lang]
    dropped = e["n_dropped_too_long"]["train"] + e["n_dropped_too_long"]["heldout"]
    assert e["n_train_items"] + e["n_heldout_items"] + dropped == e["n_source_rows_used"]


@pytest.mark.parametrize("lang", SCOPE)
def test_heldout_still_fills_the_cap(cfg, manifest, lang):
    """Held-out n must match BELEBELE's 900 so the Wilson intervals compare.

    At max_seq_tokens=1024 Bangla yielded only 722 constructible held-out items
    and this failed outright, which is one of the two reasons the ceiling was
    raised to 2048.
    """
    cap = cfg_mod.require(cfg, "finetune.heldout_eval_cap")
    e = manifest["languages"][lang]
    assert e["n_heldout_items"] >= cap, (
        f"{lang}: {e['n_heldout_items']} held-out items, below the {cap} cap")
    assert len(e["heldout_eval_item_ids"]) == cap


# --------------------------------------------------------------------------- #
# the manifest moved for exactly one reason
# --------------------------------------------------------------------------- #

# Choice digests from the algorithm_version 1 build, captured before the
# rebuild. They cover item ids, gold letters and option text -- the SELECTION
# policy -- and deliberately exclude the passage.
V1_TRAIN_CHOICES = {
    "eng_Latn": "d500ed32739c2aa4dbb816e052f00dfc4fded0ce510c52e94cc4a11bd6ee2935",
    "ben_Beng": "2a81928e4f261433d57fd7601495f42c6688e883a5400b8fc51c0404bfca9192",
}

# Article-level split, which is decided BEFORE any option is chosen or any
# window placed. Unchanged from v1, and that is the point of recording it.
V1_ARTICLE_SPLIT = {"eng_Latn": (489, 122), "ben_Beng": (544, 136)}


@pytest.mark.parametrize("lang", SCOPE)
def test_selection_actually_changed(manifest, lang):
    """The repair had to move the options, and here is the proof that it did.

    A rebuild that left the choice digest where v1 put it would mean the
    same-article distractor rule never took effect.
    """
    assert manifest["languages"][lang]["train_choices_sha256"] != \
        V1_TRAIN_CHOICES[lang], (
        f"{lang}: option selection is byte-identical to algorithm_version 1, so "
        f"the same-article rule did not take effect")


@pytest.mark.parametrize("lang", SCOPE)
def test_the_split_itself_did_not_move(manifest, lang):
    """Articles are assigned before any option or window exists.

    The repair changed which options an item gets and where its passage is cut.
    It must not have changed which articles are held out -- that would make the
    v1 and v2 corpora incomparable for reasons unrelated to the repair.
    """
    e = manifest["languages"][lang]
    assert (e["n_train_articles"], e["n_heldout_articles"]) == \
        V1_ARTICLE_SPLIT[lang]


def test_the_manifest_declares_algorithm_version_2(manifest):
    assert manifest["context_window"]["algorithm_version"] == 2
    assert manifest["context_window"]["policy"] == "span_covering_tokens"
    assert manifest["distractors"]["source"] == "same_article"
    assert manifest["distractors"]["exclude_same_article"] is False


def test_manifest_digest_is_self_consistent(manifest):
    assert p1data.manifest_payload_digest(manifest) == manifest["sha256"]
