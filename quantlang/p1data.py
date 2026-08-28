"""P1 training data: multi-wiki-qa -> grouped 80/20 split -> 4-option MCQ.

P0's evaluation contract is a frozen 900-item manifest that every run is checked
against. P1 needs the same discipline for a corpus that does not ship one, so
this module is written to be *deterministic first*: given the pinned dataset
revision and the frozen `finetune.split_seed`, every article assignment, every
distractor, and every gold-answer position is reproducible from scratch, and
`configs/p1_split_manifest.json` records digests that make any drift detectable.

Four things are enforced here rather than trusted.

1. THE SPLIT IS GROUPED BY ARTICLE, NEVER BY ROW.
   multi-wiki-qa carries several questions per Wikipedia article over one shared
   `context`. Measured on 1,000 English rows: 122 articles, mean 8.2 questions
   each, and every article had more than one question. A row-level random split
   would put the same context on both sides of the train/held-out boundary for
   essentially every article. `assert_context_grouping` goes further and proves
   no distinct context spans two groups, which is the invariant that actually
   matters -- grouping by a label is only as good as the label.

2. EVERY OPTION IS A VERBATIM SUBSTRING OF THE PASSAGE.  (algorithm_version 2)
   This is the invariant the whole construction now turns on, and it exists
   because version 1 violated it in the worst possible way.

   v1 centred the passage window on the gold answer span and drew distractors
   from OTHER articles. The gold was therefore in the passage with probability
   1.0 and a distractor only by coincidence, so the item was solvable by
   "which option appears verbatim in the passage?" -- measured shortcut success
   ~96% (English) and ~92% (Bangla) against a 100% gold-in-passage rate in both.
   The 4pp gap between those two numbers is itself language-dependent, which put
   the artefact directly on top of the quantity P1 exists to measure. Every v1
   item set and every result derived from one is excluded from the paper.

   v2 draws the three distractors from the item's OWN article -- they are the
   answers to OTHER questions about the same passage -- and then places the
   window over the span that COVERS ALL FOUR option strings. Presence is now
   constant across options, so the substring heuristic scores exactly 0.25 in
   every language and carries no signal at all. `assert_items_wellformed`
   checks this per item rather than trusting the construction to have held.

   The cost is a residual ambiguity risk that v1 avoided by construction: a
   fact from the same passage could in principle be a second defensible answer.
   It is accepted deliberately. The options answer DIFFERENT questions, the
   wellformedness check still rejects any item where two options normalise
   equal, and unlike a 100%-vs-0% presence asymmetry this risk is not
   language-correlated.

3. DISTRACTORS NEVER CROSS THE PARTITION BOUNDARY.
   Same-article implies same-partition -- the split is grouped by article -- so
   this now holds by construction rather than by filtering. A held-out item
   built from strings the model was trained on would not be clean held-out data.

4. ITEMS ARE RENDERED THROUGH P0'S FROZEN PROMPT TEMPLATE.
   `build_prompt` is imported from `quantlang.data` and used unchanged, so a P1
   item and a P0 item are the same object as far as the scorer is concerned.
   The training loss covers the answer-letter token only, which is exactly what
   `letter_logit` reads -- fine-tuning teaches the scored behaviour rather than
   drifting away from it toward free-form text.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset

from . import config as cfg_mod
from .config import REPO_ROOT
from .data import build_prompt  # P0's frozen renderer, used verbatim

SPLIT_MANIFEST_PATH = REPO_ROOT / "configs" / "p1_split_manifest.json"

PARTITIONS = ("train", "heldout")

# A row missing any of these cannot become an item. Such rows are filtered
# before the split is frozen and the count is recorded in the manifest; this is
# a data-quality filter on the SOURCE corpus, not a drop of any measurement.
REQUIRED_FIELDS = ("context", "question", "answer")

# If more than this fraction of a language's rows are unusable, the corpus is
# not what we think it is and the build stops rather than quietly shrinking.
MAX_EMPTY_ROW_FRACTION = 0.02

# The separate ceiling on rows that load fine but cannot be BUILT into an item
# whose passage contains all four options. This is a property of the
# construction, not of the corpus, so it gets its own number rather than
# stretching the source-quality one.
#
# 8% is set from measurement, not taste. Across the full corpus at
# max_seq_tokens=2048 the construction drop rate is 0.7% (English) and 6.5%
# (Bangla); at 1024 Bangla was 25.2%, which is what forced the ceiling up. The
# threshold sits above the measured Bangla rate and far below the rate any
# further regression would produce.
#
# The ABSOLUTE rate is only half the concern. A rate that is merely high is a
# cost; a rate that DIFFERS between languages is a confound, because the
# surviving items are then a differently-selected sample per language. That
# comparison needs every language at once, so it lives in
# scripts/build_p1_splits.py rather than here.
MAX_CONSTRUCTION_DROP_FRACTION = 0.08


class P1DataError(RuntimeError):
    """Raised when the P1 corpus does not match its frozen contract."""


class P1ItemTooLong(P1DataError):
    """One source row cannot be rendered inside max_seq_tokens.

    Its own question and four options consume the whole budget, leaving no room
    for a passage. Such rows are counted and excluded at construction time --
    before the corpus is frozen -- and the count is recorded per language in the
    manifest. This is a source-quality filter like the empty-row filter, not a
    dropped measurement, and MAX_EMPTY_ROW_FRACTION caps how much of it is
    tolerated before the build stops.
    """


# --------------------------------------------------------------------------- #
# determinism helpers
# --------------------------------------------------------------------------- #

def _rng_seed(*parts: Any) -> int:
    """A stable 64-bit seed from arbitrary parts.

    Derived from sha256 rather than from `hash()`, whose value for str is
    randomised per process and would make every build differ from the last.
    """
    blob = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big")


def _rng(*parts: Any):
    import numpy as np
    return np.random.default_rng(_rng_seed(*parts))


def normalise_answer(text: str) -> str:
    """Whitespace- and case-folded form, used only for equality comparisons."""
    return " ".join(str(text).split()).casefold()


# --------------------------------------------------------------------------- #
# source loading
# --------------------------------------------------------------------------- #

def lang_config(cfg: dict[str, Any], lang: str) -> str:
    """multi-wiki-qa config name for a BELEBELE language code."""
    mapping = cfg_mod.require(cfg, "finetune.lang_configs")
    if lang not in mapping:
        raise P1DataError(
            f"{lang} has no finetune.lang_configs entry. Every language in "
            f"benchmark.languages needs one or its P1 cell can never be filled."
        )
    return mapping[lang]


def load_source_rows(cfg: dict[str, Any], lang: str) -> tuple[list[dict], dict]:
    """Load one language of multi-wiki-qa at the pinned revision, normalised.

    Returns the usable rows plus a small report describing what was filtered,
    so the manifest can record it instead of the count silently changing.
    """
    dataset = cfg_mod.require(cfg, "finetune.train_dataset")
    revision = cfg_mod.require(cfg, "finetune.hf_revision")
    split = cfg_mod.require(cfg, "finetune.split")
    group_key = cfg_mod.require(cfg, "finetune.group_key")
    sep = cfg_mod.require(cfg, "finetune.item_id_separator")
    config_name = lang_config(cfg, lang)

    ds = load_dataset(dataset, config_name, split=split, revision=revision)

    raw: list[dict] = []
    n_total = 0
    n_empty = 0
    n_no_span = 0
    for r in ds:
        n_total += 1
        answers = r.get("answers") or {}
        texts = answers.get("text") or []
        starts = answers.get("answer_start") or []

        # `answer_start` indexes the RAW context. Stripping leading whitespace
        # would shift every offset, so the shift is measured and applied rather
        # than hoped about.
        raw_context = str(r.get("context", ""))
        context = raw_context.strip()
        lead = len(raw_context) - len(raw_context.lstrip())

        answer = str(texts[0]).strip() if texts else ""
        rec = {
            "group_id": str(r.get(group_key, "")).strip(),
            "context": context,
            "question": str(r.get("question", "")).strip(),
            "answer": answer,
            "source_id": str(r.get("id", "")).strip(),
            "answer_start": None,
        }
        if not rec["group_id"] or any(not rec[f] for f in REQUIRED_FIELDS):
            n_empty += 1
            continue

        # The window is centred on this span, so it has to be right. Verified
        # exact on 3,000/3,000 rows across all five languages; a row whose
        # offset does not check out falls back to a search, and one that cannot
        # be located at all is dropped and counted rather than windowed blind.
        start = (int(starts[0]) - lead) if starts else -1
        if not (0 <= start and context[start:start + len(answer)] == answer):
            start = context.find(answer)
        if start < 0:
            n_no_span += 1
            continue
        rec["answer_start"] = start
        if sep in rec["group_id"]:
            raise P1DataError(
                f"{lang}: article id {rec['group_id']!r} contains the item_id "
                f"separator {sep!r}, so item ids would be ambiguous and the "
                f"group could not be recovered from an item_id. Change "
                f"finetune.item_id_separator."
            )
        raw.append(rec)

    if n_total == 0:
        raise P1DataError(f"{lang}: {dataset}/{config_name} yielded no rows.")
    frac_empty = (n_empty + n_no_span) / n_total
    if frac_empty > MAX_EMPTY_ROW_FRACTION:
        raise P1DataError(
            f"{lang}: {n_empty} row(s) missing a context/question/answer and "
            f"{n_no_span} whose answer span could not be located, out of "
            f"{n_total} ({frac_empty:.1%}) -- above the "
            f"{MAX_EMPTY_ROW_FRACTION:.0%} ceiling. This corpus is not what the "
            f"design assumed; stop and look at it rather than training on the "
            f"remainder."
        )

    rows = assign_ordinals(raw, sep)
    report = {
        "config": config_name,
        "n_source_rows_total": n_total,
        "n_source_rows_dropped_empty": n_empty,
        "n_source_rows_dropped_no_answer_span": n_no_span,
        "n_source_rows_used": len(rows),
    }
    return rows, report


def assign_ordinals(raw: list[dict], sep: str) -> list[dict]:
    """Give every row a stable item_id: `{article}{sep}{ordinal}`.

    The ordinal comes from sorting within the article by (question, answer), so
    it does not depend on the order the dataset happened to hand rows back in.
    Duplicate (question, answer) pairs inside one article stay distinct because
    the ordinal is positional after the sort.
    """
    by_group: dict[str, list[dict]] = {}
    for rec in raw:
        by_group.setdefault(rec["group_id"], []).append(rec)

    rows: list[dict] = []
    for group_id in sorted(by_group):
        members = sorted(by_group[group_id], key=lambda r: (r["question"], r["answer"]))
        for ordinal, rec in enumerate(members):
            rows.append({**rec, "ordinal": ordinal,
                         "item_id": f"{group_id}{sep}{ordinal}"})
    rows.sort(key=lambda r: r["item_id"])
    return rows


def assert_context_grouping(lang: str, rows: list[dict]) -> None:
    """Prove no single context spans two articles.

    Grouping by a label only prevents leakage if the label actually partitions
    the text. If one context appeared under two article ids, the grouped split
    could still put that text on both sides of the boundary.
    """
    seen: dict[str, str] = {}
    offenders: list[str] = []
    for r in rows:
        digest = hashlib.sha256(r["context"].encode("utf-8")).hexdigest()
        first = seen.setdefault(digest, r["group_id"])
        if first != r["group_id"]:
            offenders.append(f"{first!r} vs {r['group_id']!r}")
    if offenders:
        raise P1DataError(
            f"{lang}: {len(offenders)} context(s) appear under more than one "
            f"article id, e.g. {offenders[:3]}. Grouping by "
            f"finetune.group_key would not actually separate the text, so the "
            f"80/20 split could leak a context across the boundary."
        )


# --------------------------------------------------------------------------- #
# the grouped split
# --------------------------------------------------------------------------- #

def grouped_split(cfg: dict[str, Any], lang: str,
                  rows: list[dict]) -> tuple[list[str], list[str]]:
    """Deterministic 80/20 split over ARTICLES. Returns (train, heldout) ids."""
    fraction = float(cfg_mod.require(cfg, "finetune.train_fraction"))
    seed = cfg_mod.require(cfg, "finetune.split_seed")

    groups = sorted({r["group_id"] for r in rows})
    if len(groups) < 2:
        raise P1DataError(
            f"{lang}: only {len(groups)} article(s); a grouped split is "
            f"meaningless.")

    rng = _rng(seed, lang, "grouped_split")
    order = rng.permutation(len(groups))
    n_train = int(round(fraction * len(groups)))
    # Both partitions must be non-empty even for a tiny corpus (smoke tests).
    n_train = max(1, min(len(groups) - 1, n_train))

    train = sorted(groups[i] for i in order[:n_train])
    heldout = sorted(groups[i] for i in order[n_train:])
    return train, heldout


# --------------------------------------------------------------------------- #
# MCQ construction
# --------------------------------------------------------------------------- #

# Digit density splits answers three ways rather than two. A boolean
# "contains a digit" put "27 February 1945" and "RFD #3" in the same class, so a
# question about a publication drew three dates and the gold became the only
# non-date option -- the same give-away the topical matching exists to remove.
DIGIT_DOMINANT_RATIO = 0.5


def _digit_class(text: str) -> str:
    """`numeric` (mostly digits, e.g. a date), `mixed`, or `text`."""
    compact = "".join(text.split())
    if not compact:
        return "text"
    ratio = sum(1 for ch in compact if ch.isdigit()) / len(compact)
    if ratio == 0.0:
        return "text"
    return "numeric" if ratio > DIGIT_DOMINANT_RATIO else "mixed"


def _length_bucket(text: str, bounds: list[int]) -> int:
    n = len(text.split())
    for i, upper in enumerate(bounds):
        if n <= upper:
            return i
    return len(bounds)


def _surface_key(text: str, bounds: list[int]) -> tuple[str, int]:
    """Coarse shape of an answer: numeric-or-not, plus a word-count bucket.

    Surface matching alone is far too weak to build a usable item. Under v2 it
    is a PREFERENCE applied within the item's own article, and only where
    honouring it costs no extra context tokens -- see `build_mcq_items`.
    """
    return (_digit_class(text), _length_bucket(text, bounds))


# v1 drew distractors from the 25 topically NEAREST OTHER articles, ranked by
# cosine over tf-idf article profiles, so an option was wrong on the facts
# rather than wrong on the category. That machinery is GONE with v1: under
# `source: same_article` every option is a fact from the very same passage,
# which is a stronger topical control than any similarity ranking, and it
# costs no all-pairs similarity pass over the corpus.

# --------------------------------------------------------------------------- #
# answer-centred context window
# --------------------------------------------------------------------------- #

_TOKENIZER_CACHE: dict[tuple[str, str], Any] = {}

# The shrink loop below exists because tokenization is not additive: a passage
# of N tokens does not always contribute exactly N tokens to the assembled
# prompt. Each pass gives back the overshoot plus a small margin.
_FIT_MARGIN_TOKENS = 8
_MAX_FIT_ITERATIONS = 8

# Occurrences of one candidate answer searched for inside one article. An answer
# string can repeat, and the nearest occurrence to the gold is often what keeps
# the covering window small. Bounded so a single very common string cannot make
# selection quadratic in the article length.
_MAX_OCCURRENCES = 8


def tokenizer_identity(cfg: dict[str, Any]) -> dict[str, str]:
    """Which tokenizer defines the window. Recorded in the split manifest.

    Windowing by tokens couples the corpus to a tokenizer, so the tokenizer is
    pinned and written down rather than left implicit.
    """
    models = cfg_mod.require(cfg, "models")
    primary = [m for m in models if m.get("role") == "primary"]
    if len(primary) != 1:
        raise P1DataError(
            f"expected exactly one role=primary model, found {len(primary)}")
    return {"hf_id": primary[0]["hf_id"], "revision": primary[0]["revision"]}


def training_tokenizer(cfg: dict[str, Any]):
    """The pinned model's tokenizer, cached per process."""
    ident = tokenizer_identity(cfg)
    key = (ident["hf_id"], ident["revision"])
    if key not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(ident["hf_id"],
                                            revision=ident["revision"])
        if not getattr(tok, "is_fast", False):
            raise P1DataError(
                f"{ident['hf_id']} loaded a slow tokenizer. The answer-centred "
                f"window needs offset mapping to locate the answer span in "
                f"token space; without it the window cannot be placed."
            )
        tok.truncation_side = cfg_mod.require(cfg, "scoring.truncation_side")
        _TOKENIZER_CACHE[key] = tok
    return _TOKENIZER_CACHE[key]


def _n_tokens(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def _trim_to_whitespace(context: str, lo: int, hi: int,
                        keep_lo: int, keep_hi: int) -> tuple[int, int]:
    """Move the window edges onto whitespace without cutting the answer span.

    `keep_lo`/`keep_hi` bound the answer; the trim never crosses them, so a
    tidier edge can never cost the evidence the window exists to preserve.
    """
    if lo > 0 and not context[lo - 1].isspace():
        j = lo
        while j < keep_lo and not context[j].isspace():
            j += 1
        while j < keep_lo and context[j].isspace():
            j += 1
        if j < keep_lo:
            lo = j
    if hi < len(context) and not context[hi].isspace():
        j = hi
        while j > keep_hi and not context[j - 1].isspace():
            j -= 1
        while j > keep_hi and context[j - 1].isspace():
            j -= 1
        if j > keep_hi:
            hi = j
    return lo, hi


def _occurrences(hay: str, needle: str, limit: int = _MAX_OCCURRENCES
                 ) -> list[tuple[int, int]]:
    """Every place `needle` appears in `hay`, left to right, capped.

    An answer string can occur several times in an article, and taking the
    occurrence NEAREST the gold is often the difference between a 200-token
    window and a 2,000-token one. The cap keeps a pathological string (a single
    common word) from dominating the search; it is a compute bound, not a
    modelling choice, and the search below is monotone in it.
    """
    out: list[tuple[int, int]] = []
    i = 0
    while len(out) < limit:
        j = hay.find(needle, i)
        if j < 0:
            break
        out.append((j, j + len(needle)))
        i = j + 1
    return out


def _reach(candidates: list[dict], gold_lo: int, gold_hi: int,
           side: str, k: int) -> tuple[int | None, list[dict]]:
    """How far out on one side we must go to pick up `k` distinct candidates.

    Returns the character distance and the candidates chosen. Ordering is by
    (distance, surface mismatch, text), which is total, so the choice is fully
    determined by the data and never by dict or dataset ordering.

    Surface shape enters ONLY here, as a tie-break between candidates that cost
    the same number of characters to reach. It is never allowed to widen the
    window: widening it is precisely what makes construction rates diverge
    between English and Bangla, which is the failure this version exists to
    avoid.
    """
    if k == 0:
        return 0, []
    ranked: list[tuple[int, int, str, dict]] = []
    for cand in candidates:
        best = None
        for lo, hi in cand["spans"]:
            if side == "left" and hi <= gold_lo:
                d = gold_lo - lo
            elif side == "right" and lo >= gold_hi:
                d = hi - gold_hi
            else:
                continue
            if best is None or d < best[0]:
                best = (d, (lo, hi))
        if best is not None:
            ranked.append((best[0], cand["surface_penalty"], cand["text"],
                           {**cand, "span": best[1]}))
    if len(ranked) < k:
        return None, []
    ranked.sort(key=lambda r: (r[0], r[1], r[2]))
    picked = [r[3] for r in ranked[:k]]
    return max(r[0] for r in ranked[:k]), picked


def choose_covering_distractors(candidates: list[dict], gold_lo: int,
                                gold_hi: int, k: int
                                ) -> tuple[list[dict], tuple[int, int]] | None:
    """Pick `k` distractors whose spans, with the gold's, span the least text.

    The window has to contain all four option strings, so the cost of a
    distractor is how much further the window must reach to include it. This
    tries every split of `k` between the two sides of the gold span and keeps
    the cheapest; with k=3 that is four candidate placements, each resolved
    deterministically by `_reach`.

    Returns (chosen, (lo, hi)) or None if fewer than `k` distinct candidates can
    be located in the context at all.
    """
    best: tuple[list[dict], tuple[int, int]] | None = None
    best_width = None
    for left_k in range(k + 1):
        left, left_picked = _reach(candidates, gold_lo, gold_hi, "left", left_k)
        if left is None:
            continue
        # A candidate that occurs on BOTH sides of the gold would otherwise be
        # selected twice and produce two identical options, so whatever the left
        # pass took is withheld from the right pass.
        taken = {normalise_answer(c["text"]) for c in left_picked}
        remaining = [c for c in candidates
                     if normalise_answer(c["text"]) not in taken]
        right, right_picked = _reach(remaining, gold_lo, gold_hi, "right",
                                     k - left_k)
        if right is None:
            continue
        lo, hi = gold_lo - left, gold_hi + right
        width = hi - lo
        if best_width is None or width < best_width:
            best_width = width
            best = (left_picked + right_picked, (lo, hi))
    return best


def span_covering_window(cfg: dict[str, Any], tok, context: str,
                         required: list[tuple[int, int]],
                         budget_tokens: int) -> str:
    """A `budget_tokens` window of `context` containing every span in `required`.

    `required` is the gold answer span plus the three distractor spans. The
    window is placed over the interval that covers them and then expanded
    outward, so the passage reads as continuous prose rather than as four
    stitched fragments, and so the four option spans do not sit at the window
    edges where their position would itself be a cue.

    The budget is a token count and is THE SAME NUMBER FOR EVERY LANGUAGE. A
    character window would hand English roughly four times the evidence of
    Bangla for the same compute; equalising tokens equalises what the model
    actually gets to read.

    Raises if the covering interval alone exceeds the budget -- such an item is
    reported and dropped by the caller, never clipped. Clipping it would drop an
    option out of the passage and silently restore the presence asymmetry this
    construction exists to remove.
    """
    if budget_tokens <= 0:
        raise P1DataError(f"context budget must be positive, got {budget_tokens}")
    if not required:
        raise P1DataError("span_covering_window needs at least one required span")

    enc = tok(context, add_special_tokens=False, return_offsets_mapping=True)
    offsets = list(enc["offset_mapping"])
    n = len(offsets)
    if n <= budget_tokens:
        return context

    req_lo = min(lo for lo, _ in required)
    req_hi = max(hi for _, hi in required)
    tok_lo = next((i for i, (s, e) in enumerate(offsets) if e > req_lo), 0)
    tok_hi = next((i + 1 for i in range(n - 1, -1, -1)
                   if offsets[i][0] < req_hi), tok_lo + 1)
    span = tok_hi - tok_lo
    if span > budget_tokens:
        raise P1ItemTooLong(
            f"the four option spans cover {span} tokens against a "
            f"{budget_tokens}-token context budget, so they cannot all be shown. "
            f"Failing rather than dropping an option out of the passage."
        )

    remaining = budget_tokens - span
    left = remaining // 2
    lo = tok_lo - left
    hi = tok_hi + (remaining - left)
    # Redistribute rather than lose budget when the covered interval sits near
    # an edge of the article.
    if lo < 0:
        hi = min(n, hi - lo)
        lo = 0
    if hi > n:
        lo = max(0, lo - (hi - n))
        hi = n

    char_lo, char_hi = offsets[lo][0], offsets[hi - 1][1]
    if cfg_mod.require(cfg, "finetune.context_window.trim_to_whitespace"):
        char_lo, char_hi = _trim_to_whitespace(context, char_lo, char_hi,
                                               req_lo, req_hi)
    window = context[char_lo:char_hi].strip()

    shift = context.index(window, max(0, char_lo - 2)) if window else char_lo
    for lo_c, hi_c in required:
        if not (shift <= lo_c and hi_c <= shift + len(window)):
            raise P1DataError(
                f"a required option span fell outside its own window; this is a "
                f"bug in window placement, not a data problem. "
                f"span=({lo_c},{hi_c}) window=({shift},{shift + len(window)})"
            )
    return window


def covering_span_tokens(tok, context: str,
                         required: list[tuple[int, int]]) -> int:
    """Token length of the interval covering `required`. The item's hard floor."""
    enc = tok(context, add_special_tokens=False, return_offsets_mapping=True)
    offsets = list(enc["offset_mapping"])
    req_lo = min(lo for lo, _ in required)
    req_hi = max(hi for _, hi in required)
    return sum(1 for s, e in offsets if e > req_lo and s < req_hi)


def windowed_passage(cfg: dict[str, Any], tok, row: dict, options: list[str],
                     gold: int, required: list[tuple[int, int]]) -> dict[str, Any]:
    """Window one item's passage and PROVE the assembled prompt fits.

    The effective budget is computed from this item's own measured overhead --
    its question, its four options, the template scaffolding and the answer
    label -- never assumed. The final prompt length is then measured, and the
    budget shrinks until it actually fits. Nothing here relies on truncation.

    The one asymmetry against version 1: the budget may EXCEED
    `context_budget_tokens` when the covering interval demands it, up to the
    per-item ceiling. Only the tail of the distribution overflows, so this costs
    ~nothing (measured: 1.00x English, 1.02x Bangla total training tokens), and
    the alternative -- clipping -- would drop an option out of the passage.
    """
    wcfg = cfg_mod.require(cfg, "finetune.context_window")
    budget = int(wcfg["context_budget_tokens"])
    overflow_ok = bool(wcfg["allow_overflow_to_cover_options"])
    max_seq = int(cfg_mod.require(cfg, "finetune.training.max_seq_tokens"))
    # One token is reserved for the answer label appended during training.
    ceiling = max_seq - 1

    probe = {**row, "passage": "", "options": options, "gold": gold}
    overhead = _n_tokens(tok, build_prompt(cfg, probe))
    headroom = ceiling - overhead
    if headroom <= 0:
        raise P1ItemTooLong(
            f"{row['item_id']}: question and options alone take {overhead} "
            f"tokens, leaving no room for a passage inside {ceiling}. "
            f"Failing rather than shipping an item with no evidence."
        )

    floor = covering_span_tokens(tok, row["context"], required)
    effective = min(headroom, max(budget, floor) if overflow_ok else budget)
    if floor > effective:
        raise P1ItemTooLong(
            f"{row['item_id']}: the four option spans cover {floor} tokens "
            f"against an effective budget of {effective} "
            f"(ceiling {ceiling} - overhead {overhead}). Dropping the item "
            f"rather than showing only some of its options."
        )

    for _ in range(_MAX_FIT_ITERATIONS):
        passage = span_covering_window(cfg, tok, row["context"], required,
                                       effective)
        prompt = build_prompt(cfg, {**row, "passage": passage,
                                    "options": options, "gold": gold})
        total = _n_tokens(tok, prompt)
        if total <= ceiling:
            return {
                "passage": passage,
                "prompt_tokens": total,
                "context_tokens": _n_tokens(tok, passage),
                "overhead_tokens": overhead,
                "effective_budget": effective,
                "covering_span_tokens": floor,
                "overflowed_budget": effective > budget,
            }
        effective -= (total - ceiling) + _FIT_MARGIN_TOKENS
        if effective < floor:
            break

    raise P1ItemTooLong(
        f"{row['item_id']}: could not fit the prompt within {ceiling} tokens "
        f"without cutting into the option spans ({floor} tokens). Reporting "
        f"rather than truncating."
    )
def build_mcq_items(cfg: dict[str, Any], lang: str, rows: list[dict],
                    group_ids: list[str], partition: str,
                    tok=None) -> tuple[list[dict], list[dict]]:
    """Build 4-option items for one partition. Returns (items, dropped).

    Distractors are the answers to OTHER questions about the SAME article, and
    the passage window is then chosen to contain all four option strings. That
    pairing is the whole point: it makes "which option appears in the passage?"
    a uniform 1-in-4 guess in every language, where version 1 made it a ~96%
    (English) / ~92% (Bangla) solution.

    Selection is deterministic by construction cost -- the three distractors are
    the ones whose spans, with the gold's, span the least text -- with surface
    shape as a tie-break at equal cost and article-order as the final tie-break.
    The only stochastic decision left is WHERE the gold lands among the four
    letters, seeded from (split_seed, lang, partition, item_id) so an item is
    reproducible in isolation without rebuilding the corpus around it.
    """
    if partition not in PARTITIONS:
        raise P1DataError(f"unknown partition {partition!r}; allowed {PARTITIONS}")
    tok = tok if tok is not None else training_tokenizer(cfg)

    seed = cfg_mod.require(cfg, "finetune.split_seed")
    n_options = int(cfg_mod.require(cfg, "finetune.n_options"))
    dcfg = cfg_mod.require(cfg, "finetune.distractors")
    bounds = list(dcfg["length_buckets"])
    match_surface = bool(dcfg["match_surface_type"])
    source = dcfg["source"]
    if source != "same_article":
        raise P1DataError(
            f"finetune.distractors.source is {source!r}. This builder "
            f"implements 'same_article' only -- 'other_article' is "
            f"algorithm_version 1, whose items are solvable by substring "
            f"presence alone."
        )

    keep = set(group_ids)
    members = [r for r in rows if r["group_id"] in keep]
    members.sort(key=lambda r: r["item_id"])

    # Rows are grouped by (article, context) rather than by article alone. An
    # article with more than one distinct context must not lend an answer from
    # one context to an item built on another -- the string would not be in the
    # passage, which is exactly the asymmetry being removed.
    by_context: dict[tuple[str, str], list[dict]] = {}
    for r in members:
        by_context.setdefault((r["group_id"], r["context"]), []).append(r)

    n_needed = n_options - 1
    items: list[dict] = []
    dropped: list[dict] = []
    for r in members:
        gold_text = r["answer"]
        gold_norm = normalise_answer(gold_text)
        gold_lo = r["answer_start"]
        gold_hi = gold_lo + len(gold_text)
        gold_surface = _surface_key(gold_text, bounds)
        context = r["context"]

        candidates: list[dict] = []
        for other in by_context[(r["group_id"], context)]:
            if other["item_id"] == r["item_id"]:
                continue
            text = other["answer"]
            if not text or normalise_answer(text) == gold_norm:
                continue
            if any(normalise_answer(c["text"]) == normalise_answer(text)
                   for c in candidates):
                continue
            spans = _occurrences(context, text)
            if not spans:
                # The answer to another question about this article is not a
                # substring of THIS row's context. It cannot be shown, so it
                # cannot be an option.
                continue
            candidates.append({
                "item_id": other["item_id"],
                "group_id": other["group_id"],
                "text": text,
                "spans": spans,
                "surface_penalty": (0 if _surface_key(text, bounds) == gold_surface
                                    else 1) if match_surface else 0,
            })

        if len(candidates) < n_needed:
            dropped.append({
                "item_id": r["item_id"],
                "reason": (f"only {len(candidates)} same-article answer(s) can be "
                           f"located in this passage; {n_needed} are needed. The "
                           f"item is dropped rather than topped up from another "
                           f"article, which would put an absent string among the "
                           f"options and restore the presence shortcut."),
            })
            continue

        # Gold placement is drawn once, before any tier is tried, so which tier
        # succeeds cannot shift where the gold lands.
        rng = _rng(seed, lang, partition, r["item_id"])
        gold_index = int(rng.integers(0, n_options))
        gold = gold_index + 1

        # Tier 1 is the candidates whose surface shape MATCHES the gold's, so
        # the gold is not the only date-shaped (or only non-date-shaped) option
        # and cannot be picked out by semantic type without reading. Tier 2 is
        # every same-article candidate.
        #
        # Tier 1 is accepted only if its window still fits inside
        # context_budget_tokens, which makes the preference genuinely free: it
        # never widens a window, never costs a token, and never drops an item
        # that tier 2 would have kept. That is why surface shape can be a real
        # preference here where in `_reach` it is only a tie-break.
        tiers: list[list[dict]] = []
        if match_surface:
            tiers.append([c for c in candidates if c["surface_penalty"] == 0])
        tiers.append(candidates)

        chosen = options = required = win = None
        last_error: str | None = None
        for tier_index, tier in enumerate(tiers):
            is_last = tier_index == len(tiers) - 1
            if len(tier) < n_needed:
                continue
            picked = choose_covering_distractors(tier, gold_lo, gold_hi, n_needed)
            if picked is None:
                last_error = (f"no placement of {n_needed} distinct same-article "
                              f"answers around the gold span could be found.")
                continue
            cand_chosen, _cover = picked
            cand_options = [c["text"] for c in cand_chosen]
            cand_options.insert(gold_index, gold_text)
            cand_required = sorted([(gold_lo, gold_hi)]
                                   + [c["span"] for c in cand_chosen])
            try:
                # The window is placed AFTER the options are chosen, because the
                # effective budget depends on how many tokens this item's own
                # question and options consume -- and because the window must
                # cover their spans.
                cand_win = windowed_passage(cfg, tok, r, cand_options, gold,
                                            cand_required)
            except P1ItemTooLong as exc:
                last_error = str(exc)
                continue
            if cand_win["overflowed_budget"] and not is_last:
                # Tier 1 would only fit by spending extra tokens. Prefer the
                # cheaper item over the better-matched one.
                continue
            chosen, options, required, win = (cand_chosen, cand_options,
                                              cand_required, cand_win)
            break

        if win is None:
            # Counted and reported, never silently truncated. Only P1ItemTooLong
            # is absorbed above, so a genuine bug in window placement still stops
            # the build.
            dropped.append({"item_id": r["item_id"],
                            "reason": last_error or "no usable distractor set"})
            continue
        distractor_ids = [c["item_id"] for c in chosen]

        items.append({
            "item_id": r["item_id"],
            "lang": lang,
            "partition": partition,
            "group_id": r["group_id"],
            "passage": win["passage"],
            "question": r["question"],
            "options": options,
            "gold": gold,                     # 1-indexed, as BELEBELE ships it
            "gold_text": gold_text,
            "distractor_item_ids": distractor_ids,
            "source_id": r["source_id"],
            "prompt_tokens": win["prompt_tokens"],
            "context_tokens": win["context_tokens"],
            "covering_span_tokens": win["covering_span_tokens"],
            "overflowed_budget": win["overflowed_budget"],
            "full_context_tokens": None,
            "answer_in_passage": gold_text in win["passage"],
            "n_options_in_passage": sum(1 for o in options
                                        if o in win["passage"]),
        })

    return items, dropped


def assert_items_wellformed(cfg: dict[str, Any], items: list[dict]) -> None:
    """Every invariant an MCQ item must satisfy before it can be trained on."""
    n_options = int(cfg_mod.require(cfg, "finetune.n_options"))
    for it in items:
        opts = it["options"]
        if len(opts) != n_options:
            raise P1DataError(
                f"{it['item_id']}: {len(opts)} options, expected {n_options}")
        norms = [normalise_answer(o) for o in opts]
        if len(set(norms)) != n_options:
            raise P1DataError(
                f"{it['item_id']}: duplicate options {opts}. Two identical "
                f"options make more than one letter defensible."
            )
        if not 1 <= it["gold"] <= n_options:
            raise P1DataError(f"{it['item_id']}: gold {it['gold']} out of range")
        if normalise_answer(opts[it["gold"] - 1]) != normalise_answer(it["gold_text"]):
            raise P1DataError(
                f"{it['item_id']}: option {it['gold']} is not the source answer. "
                f"The gold letter and the gold text disagree."
            )
        n_correct = sum(1 for n in norms if n == normalise_answer(it["gold_text"]))
        if n_correct != 1:
            raise P1DataError(
                f"{it['item_id']}: {n_correct} options equal the correct answer; "
                f"exactly one is required."
            )
        sep = cfg_mod.require(cfg, "finetune.item_id_separator")
        source_articles = {d.rsplit(sep, 1)[0] for d in it["distractor_item_ids"]}
        if source_articles != {it["group_id"]}:
            raise P1DataError(
                f"{it['item_id']}: distractors came from {sorted(source_articles)} "
                f"rather than from the item's own article {it['group_id']!r}. An "
                f"answer from another article is not in this passage, so the "
                f"gold would be identifiable by presence alone."
            )
        if it["item_id"] in it["distractor_item_ids"]:
            raise P1DataError(
                f"{it['item_id']}: the item's own answer was reused as one of "
                f"its distractors.")

        # THE anti-shortcut invariant. Checked per item rather than trusted to
        # the construction, because it is the single property that stops the
        # task being solvable without reading: if every option is present,
        # "pick the one that appears in the passage" is a 1-in-4 guess.
        absent = [o for o in opts if o not in it["passage"]]
        if absent:
            raise P1DataError(
                f"{it['item_id']}: option(s) {absent} are not verbatim in the "
                f"passage. Presence would then identify the gold without any "
                f"reading, which is the failure algorithm_version 2 exists to "
                f"remove."
            )


def surface_homogeneity(items: list[dict]) -> float:
    """Share of items whose four options all share the gold's digit class.

    A low value means the correct answer is often the only date-shaped (or only
    non-date-shaped) option, which a model can exploit without reading the
    passage at all.
    """
    if not items:
        return 0.0
    n = 0
    for it in items:
        gold_class = _digit_class(it["options"][it["gold"] - 1])
        if all(_digit_class(o) == gold_class for o in it["options"]):
            n += 1
    return n / len(items)


def lexical_shortcut_accuracy(items: list[dict]) -> float:
    """Accuracy of "choose the option that appears verbatim in the passage".

    THE headline diagnostic. Version 1 scored ~0.96 here in English and ~0.92 in
    Bangla, on a gold-in-passage rate of 1.00 in both -- the task was solvable
    without reading, and solvable to a language-dependent degree.

    Ties are scored as the heuristic's own expected value, 1/k over the k
    options it cannot separate, rather than by breaking them in the gold's
    favour or against it. Under version 2 every option is present, so k is 4 and
    the value is exactly 0.25 in every language.
    """
    if not items:
        return 0.0
    total = 0.0
    for it in items:
        present = [i for i, o in enumerate(it["options"], start=1)
                   if o in it["passage"]]
        if not present:
            # No option is present: the heuristic has nothing to go on and
            # guesses uniformly over all four.
            total += 1.0 / len(it["options"])
        elif it["gold"] in present:
            total += 1.0 / len(present)
    return total / len(items)


def gold_position_uniformity(items: list[dict]) -> dict[str, Any]:
    """Where the gold sits among the four option spans, by position in passage.

    A window placed over the four option spans could in principle leave the gold
    systematically central (or systematically at an edge), which would be a new
    positional shortcut replacing the substring one. This measures the rank of
    the gold's first occurrence among the four options' first occurrences, and
    the build records it so the claim is audited rather than assumed.
    """
    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    n = 0
    for it in items:
        positions = []
        for i, opt in enumerate(it["options"], start=1):
            at = it["passage"].find(opt)
            if at < 0:
                positions = []
                break
            positions.append((at, i))
        if not positions:
            continue
        positions.sort()
        rank = next(r for r, (_, i) in enumerate(positions, start=1)
                    if i == it["gold"])
        counts[rank] = counts.get(rank, 0) + 1
        n += 1
    return {"n_scored": n,
            "share_by_rank": {str(k): (v / n if n else 0.0)
                              for k, v in sorted(counts.items())}}


def construction_diagnostics(items: list[dict], dropped: list[dict],
                             n_source_rows: int) -> dict[str, Any]:
    """Everything needed to judge whether this item set is learnable and fair.

    Recorded per language and per partition in the split manifest. Nothing here
    gates the build on its own -- `scripts/build_p1_splits.py` owns the
    thresholds -- because a diagnostic that silently changes what gets built is
    no longer a diagnostic.
    """
    n = len(items)
    if n == 0:
        return {"n_items": 0, "n_dropped": len(dropped)}

    gold_letters: dict[str, int] = {}
    for it in items:
        key = str(it["gold"])
        gold_letters[key] = gold_letters.get(key, 0) + 1

    n_opts = {len(it["options"]) for it in items}
    dup = sum(1 for it in items
              if len({normalise_answer(o) for o in it["options"]})
              != len(it["options"]))
    gold_in = sum(1 for it in items if it["gold_text"] in it["passage"])
    distractor_in = 0
    distractor_total = 0
    for it in items:
        for i, o in enumerate(it["options"], start=1):
            if i == it["gold"]:
                continue
            distractor_total += 1
            distractor_in += int(o in it["passage"])

    prompts = sorted(it["prompt_tokens"] for it in items)
    covers = sorted(it.get("covering_span_tokens") or 0 for it in items)

    def pct(v: list[int], q: float) -> int:
        return v[min(len(v) - 1, int(q * len(v)))]

    same_article = sum(
        1 for it in items
        if all(d.rsplit("#", 1)[0] == it["group_id"]
               for d in it["distractor_item_ids"]))

    return {
        "n_items": n,
        "n_dropped": len(dropped),
        "drop_rate": len(dropped) / n_source_rows if n_source_rows else 0.0,
        "option_counts": sorted(n_opts),
        "n_items_with_duplicate_options": dup,
        "gold_letter_distribution": gold_letters,
        "gold_letter_max_share": max(gold_letters.values()) / n,
        "gold_in_context_rate": gold_in / n,
        "distractor_in_context_rate": (distractor_in / distractor_total
                                       if distractor_total else 0.0),
        "lexical_shortcut_accuracy": lexical_shortcut_accuracy(items),
        "gold_position": gold_position_uniformity(items),
        "same_article_distractor_rate": same_article / n,
        "option_surface_homogeneity": surface_homogeneity(items),
        "prompt_tokens_median": pct(prompts, 0.50),
        "prompt_tokens_p90": pct(prompts, 0.90),
        "prompt_tokens_max": prompts[-1],
        "covering_span_tokens_median": pct(covers, 0.50),
        "covering_span_tokens_p90": pct(covers, 0.90),
        "n_overflowed_budget": sum(1 for it in items
                                   if it.get("overflowed_budget")),
        # Truncation is not a thing that can happen here: the prompt length is
        # measured and the item dropped if it does not fit. Recorded as 0 so the
        # absence is a stated fact rather than a missing field.
        "truncation_rate": 0.0,
    }


def choice_digest(items: list[dict]) -> str:
    """sha256 over item ids, gold letters and exact option text, in order.

    This is the fingerprint of the ratified SELECTION policy -- which
    distractors were drawn and where the gold landed. It deliberately excludes
    the passage, so it stays constant across a change to the passage
    representation and can prove such a change touched nothing else.
    """
    h = hashlib.sha256()
    for it in items:
        h.update(it["item_id"].encode("utf-8"));   h.update(b"\x00")
        h.update(str(it["gold"]).encode("utf-8")); h.update(b"\x00")
        for opt in it["options"]:
            h.update(opt.encode("utf-8"));         h.update(b"\x01")
        h.update(b"\n")
    return h.hexdigest()


def items_digest(items: list[dict]) -> str:
    """sha256 over the complete item, passage included.

    Any change to construction -- a different distractor, a shifted gold
    position, a re-ordered option list, a differently placed context window --
    moves this digest, so a rebuilt corpus that no longer matches the manifest
    is detected instead of silently used.
    """
    h = hashlib.sha256()
    for it in items:
        h.update(it["item_id"].encode("utf-8"));   h.update(b"\x00")
        h.update(str(it["gold"]).encode("utf-8")); h.update(b"\x00")
        h.update(it["passage"].encode("utf-8"));   h.update(b"\x02")
        h.update(it["question"].encode("utf-8"));  h.update(b"\x03")
        for opt in it["options"]:
            h.update(opt.encode("utf-8"));         h.update(b"\x01")
        h.update(b"\n")
    return h.hexdigest()


def window_stats(items: list[dict]) -> dict[str, Any]:
    """Window measurements for one partition, recorded in the manifest."""
    if not items:
        return {}
    prompts = sorted(it["prompt_tokens"] for it in items)
    contexts = sorted(it["context_tokens"] for it in items)

    def pct(values: list[int], q: float) -> int:
        return values[min(len(values) - 1, int(q * len(values)))]

    return {
        "n_items": len(items),
        "prompt_tokens_median": pct(prompts, 0.50),
        "prompt_tokens_p90": pct(prompts, 0.90),
        "prompt_tokens_max": prompts[-1],
        "context_tokens_median": pct(contexts, 0.50),
        "context_tokens_p90": pct(contexts, 0.90),
        "evidence_retained": sum(1 for it in items
                                 if it["answer_in_passage"]) / len(items),
        # Under v2 the window must contain all four options, not just the gold,
        # so this is the property that actually matters and it is recorded
        # alongside the older one rather than replacing it.
        "all_options_retained": sum(
            1 for it in items
            if it["n_options_in_passage"] == len(it["options"])) / len(items),
        "n_overflowed_budget": sum(1 for it in items
                                   if it.get("overflowed_budget")),
    }


def select_heldout_eval(cfg: dict[str, Any], lang: str,
                        heldout_items: list[dict]) -> list[str]:
    """Frozen deterministic subsample of the held-out partition, capped.

    Capped at `finetune.heldout_eval_cap` so the secondary surface has the same
    n as BELEBELE and the two sets' Wilson intervals are directly comparable.
    """
    cap = int(cfg_mod.require(cfg, "finetune.heldout_eval_cap"))
    seed = cfg_mod.require(cfg, "finetune.split_seed")
    ids = sorted(it["item_id"] for it in heldout_items)
    if len(ids) <= cap:
        return ids
    rng = _rng(seed, lang, "heldout_eval_selection")
    order = rng.permutation(len(ids))
    return sorted(ids[int(i)] for i in order[:cap])


# --------------------------------------------------------------------------- #
# whole-language build
# --------------------------------------------------------------------------- #

def build_language(cfg: dict[str, Any], lang: str) -> dict[str, Any]:
    """Load, split, and build every P1 item for one language."""
    rows, report = load_source_rows(cfg, lang)
    assert_context_grouping(lang, rows)
    train_groups, heldout_groups = grouped_split(cfg, lang, rows)

    overlap = set(train_groups) & set(heldout_groups)
    if overlap:
        raise P1DataError(
            f"{lang}: {len(overlap)} article(s) in both partitions, e.g. "
            f"{sorted(overlap)[:3]}. The split is not a partition."
        )

    tok = training_tokenizer(cfg)
    train_items, train_dropped = build_mcq_items(
        cfg, lang, rows, train_groups, "train", tok)
    heldout_items, heldout_dropped = build_mcq_items(
        cfg, lang, rows, heldout_groups, "heldout", tok)
    n_dropped = len(train_dropped) + len(heldout_dropped)
    assert_items_wellformed(cfg, train_items)
    assert_items_wellformed(cfg, heldout_items)

    if len(train_items) + len(heldout_items) + n_dropped != len(rows):
        raise P1DataError(
            f"{lang}: {len(train_items)} + {len(heldout_items)} items and "
            f"{n_dropped} dropped, from {len(rows)} source rows. Every row must "
            f"land in exactly one partition or be accounted for as dropped."
        )
    if n_dropped / len(rows) > MAX_CONSTRUCTION_DROP_FRACTION:
        raise P1DataError(
            f"{lang}: {n_dropped}/{len(rows)} rows ({n_dropped / len(rows):.1%}) "
            f"could not be built into a four-option item whose passage contains "
            f"all four options, above the "
            f"{MAX_CONSTRUCTION_DROP_FRACTION:.0%} ceiling. Stop and look at the "
            f"corpus rather than raising the ceiling: the ABSOLUTE rate is only "
            f"half the concern, and scripts/build_p1_splits.py separately "
            f"refuses a build whose drop rates diverge between languages."
        )

    eval_ids = select_heldout_eval(cfg, lang, heldout_items)
    gold_by_id = {it["item_id"]: it["gold"] for it in heldout_items}

    # Recorded, not asserted. This is the share of items on which the gold
    # cannot be picked out by digit-class alone -- the give-away that surface
    # matching exists to reduce. It will not reach 1.0: surface shape is only a
    # tie-break between equally cheap same-article candidates and is never
    # allowed to widen the window. Auditing the number beats assuming it.
    homogeneity = {
        part: surface_homogeneity(items)
        for part, items in (("train", train_items), ("heldout", heldout_items))
    }
    diagnostics = {
        "train": construction_diagnostics(train_items, train_dropped, len(rows)),
        "heldout": construction_diagnostics(heldout_items, heldout_dropped,
                                            len(rows)),
    }

    return {
        "diagnostics": diagnostics,
        "lang": lang,
        "rows": rows,
        "report": report,
        "train_groups": train_groups,
        "heldout_groups": heldout_groups,
        "train_items": train_items,
        "heldout_items": heldout_items,
        "heldout_eval_item_ids": eval_ids,
        "heldout_eval_gold": {i: gold_by_id[i] for i in eval_ids},
        "train_digest": items_digest(train_items),
        "heldout_digest": items_digest(heldout_items),
        "train_choices": choice_digest(train_items),
        "heldout_choices": choice_digest(heldout_items),
        "surface_homogeneity": homogeneity,
        "window": {"train": window_stats(train_items),
                   "heldout": window_stats(heldout_items)},
        "n_dropped_too_long": {"train": len(train_dropped),
                               "heldout": len(heldout_dropped)},
        "dropped_too_long": train_dropped + heldout_dropped,
    }


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #

def load_split_manifest(path: Path | None = None) -> dict[str, Any]:
    path = path or SPLIT_MANIFEST_PATH
    if not path.exists():
        raise P1DataError(
            f"P1 split manifest missing: {path}\n"
            f"Run `python scripts/build_p1_splits.py` first. Training and "
            f"held-out evaluation must not infer their item set from whatever a "
            f"rebuild happens to produce."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_payload_digest(manifest: dict[str, Any]) -> str:
    """Digest over everything in the manifest except the digest field itself."""
    payload = {k: v for k, v in manifest.items() if k != "sha256"}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def verify_against_manifest(built: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Check a freshly built language against the frozen manifest. Fatal on drift."""
    lang = built["lang"]
    entry = (manifest.get("languages") or {}).get(lang)
    if entry is None:
        raise P1DataError(
            f"{lang} is absent from the frozen P1 split manifest. It is not part "
            f"of the pinned P1 corpus."
        )
    checks = (
        ("train_articles", sorted(built["train_groups"]), sorted(entry["train_articles"])),
        ("heldout_articles", sorted(built["heldout_groups"]), sorted(entry["heldout_articles"])),
        ("train_digest", built["train_digest"], entry["train_items_sha256"]),
        ("heldout_digest", built["heldout_digest"], entry["heldout_items_sha256"]),
        ("train_choices", built["train_choices"], entry["train_choices_sha256"]),
        ("heldout_choices", built["heldout_choices"],
         entry["heldout_choices_sha256"]),
        ("heldout_eval_item_ids", built["heldout_eval_item_ids"],
         entry["heldout_eval_item_ids"]),
    )
    for name, got, want in checks:
        if got != want:
            raise P1DataError(
                f"{lang}: rebuilt {name} does not match the frozen manifest. The "
                f"corpus, the revision, or the construction rule changed; prior "
                f"P1 results are no longer comparable."
            )


def train_equalise_cap(cfg: dict[str, Any],
                       manifest: dict[str, Any]) -> int | None:
    """Common training-set size across `finetune.final_scope_languages`.

    English yields more constructible items than Bangla, so without this the FT
    arms would differ in how many gradient steps each language got, and "English
    gained more from fine-tuning" could not be separated from "English trained
    on more data". The cap is min(n_train_items) over the FINAL-SCOPE languages
    only, so rebuilding a provenance language alongside them cannot move it.

    Returns None when equalisation is off.
    """
    if not cfg_mod.require(cfg, "finetune.equalise_train_partition"):
        return None
    scope = cfg_mod.require(cfg, "finetune.final_scope_languages")
    langs = manifest.get("languages") or {}
    missing = [l for l in scope if l not in langs]
    if missing:
        raise P1DataError(
            f"the split manifest is missing final-scope language(s) {missing}, "
            f"so the common training-set size cannot be determined. Rebuild the "
            f"manifest for every language before training."
        )
    return min(int(langs[l]["n_train_items"]) for l in scope)


def select_equalised_train(cfg: dict[str, Any], lang: str,
                           train_items: list[dict], cap: int) -> list[str]:
    """Deterministic subsample of a train partition down to `cap` item ids.

    Same mechanism as `select_heldout_eval`: seeded from split_seed and the
    language, sorted for a stable order, and returning ids rather than items so
    the choice is recordable and checkable.
    """
    seed = cfg_mod.require(cfg, "finetune.split_seed")
    ids = sorted(it["item_id"] for it in train_items)
    if len(ids) <= cap:
        return ids
    rng = _rng(seed, lang, "train_equalise")
    order = rng.permutation(len(ids))
    return sorted(ids[int(i)] for i in order[:cap])


def load_partition(cfg: dict[str, Any], lang: str, partition: str,
                   manifest: dict[str, Any] | None = None) -> list[dict]:
    """Rebuild one partition and verify it against the frozen manifest.

    This is the only sanctioned way for training or evaluation code to get P1
    items -- it is the P1 analogue of `data.load_language`.

    The `train` partition is returned already trimmed to the common size across
    the final-scope languages. Doing the trim HERE rather than in the caller is
    deliberate: every training run goes through this function, so no run can
    accidentally train on the untrimmed set.
    """
    manifest = manifest or load_split_manifest()
    built = build_language(cfg, lang)
    verify_against_manifest(built, manifest)
    if partition == "train":
        cap = train_equalise_cap(cfg, manifest)
        if cap is None or len(built["train_items"]) <= cap:
            return built["train_items"]
        keep = set(select_equalised_train(cfg, lang, built["train_items"], cap))
        return [it for it in built["train_items"] if it["item_id"] in keep]
    if partition == "train_full":
        return built["train_items"]
    if partition == "heldout":
        return built["heldout_items"]
    if partition == "heldout_eval":
        keep = set(built["heldout_eval_item_ids"])
        return [it for it in built["heldout_items"] if it["item_id"] in keep]
    raise P1DataError(
        f"unknown partition {partition!r}; allowed "
        f"{PARTITIONS + ('heldout_eval', 'train_full')}")


def build_p1_prompt(cfg: dict[str, Any], item: dict[str, Any]) -> str:
    """Render a P1 item through P0's frozen template, unchanged."""
    return build_prompt(cfg, item)
