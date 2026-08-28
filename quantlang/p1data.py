"""P1 training data: multi-wiki-qa -> grouped 80/20 split -> 4-option MCQ.

P0's evaluation contract is a frozen 900-item manifest that every run is checked
against. P1 needs the same discipline for a corpus that does not ship one, so
this module is written to be *deterministic first*: given the pinned dataset
revision and the frozen `finetune.split_seed`, every article assignment, every
distractor, and every gold-answer position is reproducible from scratch, and
`configs/p1_split_manifest.json` records digests that make any drift detectable.

Three things are enforced here rather than trusted.

1. THE SPLIT IS GROUPED BY ARTICLE, NEVER BY ROW.
   multi-wiki-qa carries several questions per Wikipedia article over one shared
   `context`. Measured on 1,000 English rows: 122 articles, mean 8.2 questions
   each, and every article had more than one question. A row-level random split
   would put the same context on both sides of the train/held-out boundary for
   essentially every article. `assert_context_grouping` goes further and proves
   no distinct context spans two groups, which is the invariant that actually
   matters -- grouping by a label is only as good as the label.

2. DISTRACTORS COME FROM OTHER ARTICLES, AND ONLY FROM THE SAME PARTITION.
   Same-article answers are excluded because a fact stated in the same context
   can be a second defensible answer, and an ambiguous item teaches noise.
   Cross-partition answers are excluded because a held-out item built from
   strings the model was trained on is no longer clean held-out data.

3. ITEMS ARE RENDERED THROUGH P0'S FROZEN PROMPT TEMPLATE.
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


def _has_digit(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


def _length_bucket(text: str, bounds: list[int]) -> int:
    n = len(text.split())
    for i, upper in enumerate(bounds):
        if n <= upper:
            return i
    return len(bounds)


def _surface_key(text: str, bounds: list[int]) -> tuple[str, int]:
    """Coarse shape of an answer: numeric-or-not, plus a word-count bucket.

    Surface matching alone is far too weak to build a usable item -- it is a
    tie-breaker applied *within* a topically related candidate set, never the
    primary filter. See `_neighbours` for why.
    """
    return (_digit_class(text), _length_bucket(text, bounds))


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Terms kept per article profile, highest tf-idf first. Pruning to the most
# distinctive terms is what keeps the all-pairs similarity cheap; the tail of
# common words contributes noise and cost in equal measure.
_PROFILE_TERMS = 120


def _tokenise(text: str) -> list[str]:
    """Whitespace/word tokens, case-folded. Works for all five scripts, which
    all use spaces; no language-specific analysis is involved."""
    return [t for t in _TOKEN_RE.findall(text.casefold()) if len(t) > 1]


def _article_profiles(members: list[dict]) -> dict[str, dict[str, float]]:
    """tf-idf profile per article, L2-normalised, pruned to _PROFILE_TERMS."""
    texts: dict[str, set[str]] = {}
    for r in members:
        texts.setdefault(r["group_id"], set()).add(r["context"])

    docs = {g: _tokenise(" ".join(sorted(s))) for g, s in texts.items()}
    n_docs = len(docs)
    df: dict[str, int] = {}
    for toks in docs.values():
        for t in set(toks):
            df[t] = df.get(t, 0) + 1

    profiles: dict[str, dict[str, float]] = {}
    for g in sorted(docs):
        tf: dict[str, int] = {}
        for t in docs[g]:
            tf[t] = tf.get(t, 0) + 1
        vec = {t: (1.0 + math.log(c)) * math.log(n_docs / df[t])
               for t, c in tf.items() if df[t] < n_docs}
        top = sorted(vec.items(), key=lambda kv: (-kv[1], kv[0]))[:_PROFILE_TERMS]
        norm = math.sqrt(sum(w * w for _, w in top)) or 1.0
        profiles[g] = {t: w / norm for t, w in top}
    return profiles


def _neighbours(profiles: dict[str, dict[str, float]], k: int) -> dict[str, list[str]]:
    """For each article, the k most topically similar OTHER articles.

    This is the fix for the failure mode that surface matching alone produces.
    Drawing distractors uniformly from the whole language gives options like
    "University of Alabama Press" against a question about troop movements: the
    correct answer is identifiable from semantic type without reading the
    passage at all, so training on such items teaches a shortcut rather than
    reading comprehension in the target language. Distractors taken from
    topically adjacent articles are wrong on the facts rather than wrong on the
    category, which is what makes the item require the passage.

    Cosine over the pruned tf-idf profiles, via an inverted index. Ties break on
    article id so the ordering is total and reproducible.
    """
    groups = sorted(profiles)
    inverted: dict[str, list[tuple[str, float]]] = {}
    for g in groups:
        for term, weight in profiles[g].items():
            inverted.setdefault(term, []).append((g, weight))

    out: dict[str, list[str]] = {}
    for g in groups:
        scores: dict[str, float] = {}
        for term, weight in profiles[g].items():
            for other, other_weight in inverted.get(term, ()):
                if other == g:
                    continue
                scores[other] = scores.get(other, 0.0) + weight * other_weight
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        out[g] = [other for other, _ in ranked]
    return out


# --------------------------------------------------------------------------- #
# answer-centred context window
# --------------------------------------------------------------------------- #

_TOKENIZER_CACHE: dict[tuple[str, str], Any] = {}

# The shrink loop below exists because tokenization is not additive: a passage
# of N tokens does not always contribute exactly N tokens to the assembled
# prompt. Each pass gives back the overshoot plus a small margin.
_FIT_MARGIN_TOKENS = 8
_MAX_FIT_ITERATIONS = 8


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


def answer_centred_window(cfg: dict[str, Any], tok, context: str,
                          answer: str, answer_start: int,
                          budget_tokens: int) -> str:
    """A `budget_tokens`-long window of `context` centred on the answer span.

    The budget is a token count and is the SAME NUMBER FOR EVERY LANGUAGE. A
    character window would hand English roughly four times the evidence of
    Sinhala for the same compute; equalising tokens equalises what the model
    actually gets to read.

    Raises if the answer span alone exceeds the budget -- an item whose answer
    cannot fit is reported, never silently clipped.
    """
    if budget_tokens <= 0:
        raise P1DataError(f"context budget must be positive, got {budget_tokens}")

    enc = tok(context, add_special_tokens=False, return_offsets_mapping=True)
    offsets = list(enc["offset_mapping"])
    n = len(offsets)
    if n <= budget_tokens:
        return context

    a_lo, a_hi = answer_start, answer_start + len(answer)
    tok_lo = next((i for i, (s, e) in enumerate(offsets) if e > a_lo), 0)
    tok_hi = next((i + 1 for i in range(n - 1, -1, -1)
                   if offsets[i][0] < a_hi), tok_lo + 1)
    span = tok_hi - tok_lo
    if span > budget_tokens:
        # Loud, per-item, and counted by the caller -- never a clipped answer.
        raise P1ItemTooLong(
            f"answer span is {span} tokens against a {budget_tokens}-token "
            f"context budget, so it cannot be shown in full: {answer[:80]!r}. "
            f"Failing rather than truncating the answer itself."
        )

    remaining = budget_tokens - span
    left = remaining // 2
    lo = tok_lo - left
    hi = tok_hi + (remaining - left)
    # Redistribute rather than lose budget when the span sits near an edge.
    if lo < 0:
        hi = min(n, hi - lo)
        lo = 0
    if hi > n:
        lo = max(0, lo - (hi - n))
        hi = n

    char_lo, char_hi = offsets[lo][0], offsets[hi - 1][1]
    if cfg_mod.require(cfg, "finetune.context_window.trim_to_whitespace"):
        char_lo, char_hi = _trim_to_whitespace(context, char_lo, char_hi,
                                               a_lo, a_hi)
    window = context[char_lo:char_hi].strip()
    if answer not in window:
        raise P1DataError(
            f"the answer span fell outside its own window; this is a bug in "
            f"window placement, not a data problem. answer={answer[:60]!r}"
        )
    return window


def windowed_passage(cfg: dict[str, Any], tok, row: dict, options: list[str],
                     gold: int) -> dict[str, Any]:
    """Window one item's passage and PROVE the assembled prompt fits.

    The effective budget is computed from this item's own measured overhead --
    its question, its four options, the template scaffolding and the answer
    label -- never assumed. The final prompt length is then measured, and the
    budget shrinks until it actually fits. Nothing here relies on truncation.
    """
    wcfg = cfg_mod.require(cfg, "finetune.context_window")
    budget = int(wcfg["context_budget_tokens"])
    max_seq = int(cfg_mod.require(cfg, "finetune.training.max_seq_tokens"))
    # One token is reserved for the answer label appended during training.
    ceiling = max_seq - 1

    probe = {**row, "passage": "", "options": options, "gold": gold}
    overhead = _n_tokens(tok, build_prompt(cfg, probe))
    effective = min(budget, ceiling - overhead)
    if effective <= 0:
        raise P1ItemTooLong(
            f"{row['item_id']}: question and options alone take {overhead} "
            f"tokens, leaving no room for a passage inside {ceiling}. "
            f"Failing rather than shipping an item with no evidence."
        )

    for _ in range(_MAX_FIT_ITERATIONS):
        passage = answer_centred_window(cfg, tok, row["context"], row["answer"],
                                        row["answer_start"], effective)
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
            }
        effective -= (total - ceiling) + _FIT_MARGIN_TOKENS
        if effective <= 0:
            break

    raise P1ItemTooLong(
        f"{row['item_id']}: could not fit the prompt within {ceiling} tokens "
        f"after {_MAX_FIT_ITERATIONS} attempts. Reporting rather than "
        f"truncating."
    )


def build_mcq_items(cfg: dict[str, Any], lang: str, rows: list[dict],
                    group_ids: list[str], partition: str,
                    tok=None) -> tuple[list[dict], list[dict]]:
    """Build 4-option items for one partition. Returns (items, dropped).

    The answer pool is drawn from THIS partition only, and never from the item's
    own article. Everything is seeded from (split_seed, lang, partition,
    item_id), so an item is reproducible in isolation without rebuilding the
    corpus around it.
    """
    if partition not in PARTITIONS:
        raise P1DataError(f"unknown partition {partition!r}; allowed {PARTITIONS}")
    tok = tok if tok is not None else training_tokenizer(cfg)

    seed = cfg_mod.require(cfg, "finetune.split_seed")
    n_options = int(cfg_mod.require(cfg, "finetune.n_options"))
    dcfg = cfg_mod.require(cfg, "finetune.distractors")
    bounds = list(dcfg["length_buckets"])
    exclude_same_article = bool(dcfg["exclude_same_article"])
    match_surface = bool(dcfg["match_surface_type"])
    n_neighbours = int(dcfg["neighbour_articles"])

    keep = set(group_ids)
    members = [r for r in rows if r["group_id"] in keep]
    members.sort(key=lambda r: r["item_id"])

    # Answer pool for this partition, in a fixed order so sampling is stable.
    pool = [{"item_id": r["item_id"], "group_id": r["group_id"],
             "text": r["answer"],
             "surface": _surface_key(r["answer"], bounds)} for r in members]

    by_surface: dict[tuple, list[dict]] = {}
    by_digit: dict[str, list[dict]] = {}
    by_group: dict[str, list[dict]] = {}
    for entry in pool:
        by_surface.setdefault(entry["surface"], []).append(entry)
        by_digit.setdefault(entry["surface"][0], []).append(entry)
        by_group.setdefault(entry["group_id"], []).append(entry)

    neighbours = _neighbours(_article_profiles(members), n_neighbours)

    items: list[dict] = []
    dropped: list[dict] = []
    for r in members:
        gold_text = r["answer"]
        gold_norm = normalise_answer(gold_text)
        rng = _rng(seed, lang, partition, r["item_id"])

        def usable(entry: dict) -> bool:
            if exclude_same_article and entry["group_id"] == r["group_id"]:
                return False
            return normalise_answer(entry["text"]) != gold_norm

        # Candidates from topically adjacent articles first, so a distractor is
        # wrong on the facts rather than wrong on the category. Surface matching
        # is a tie-breaker inside that set, never the primary filter -- on its
        # own it produced items whose gold was identifiable from semantic type
        # without reading the passage.
        near = [e for g in neighbours.get(r["group_id"], ())
                for e in by_group.get(g, ())]
        gold_surface = _surface_key(gold_text, bounds)
        tiers = []
        if match_surface:
            tiers.append([e for e in near if e["surface"] == gold_surface])
            tiers.append(near)
            tiers.append(by_surface.get(gold_surface, []))
            tiers.append(by_digit.get(_digit_class(gold_text), []))
        else:
            tiers.append(near)
        tiers.append(pool)

        chosen: list[dict] = []
        chosen_norm = {gold_norm}
        for tier in tiers:
            candidates = [e for e in tier if usable(e)
                          and normalise_answer(e["text"]) not in chosen_norm]
            if not candidates:
                continue
            for idx in rng.permutation(len(candidates)):
                entry = candidates[int(idx)]
                norm = normalise_answer(entry["text"])
                if norm in chosen_norm:
                    continue
                chosen.append(entry)
                chosen_norm.add(norm)
                if len(chosen) == n_options - 1:
                    break
            if len(chosen) == n_options - 1:
                break

        if len(chosen) != n_options - 1:
            raise P1DataError(
                f"{lang}/{partition}: item {r['item_id']!r} could not be given "
                f"{n_options - 1} distinct distractors (found {len(chosen)}). "
                f"The answer pool for this partition is too small or too "
                f"repetitive. Stop and inspect it -- do not build a short item."
            )

        gold_index = int(rng.integers(0, n_options))
        options = [e["text"] for e in chosen]
        options.insert(gold_index, gold_text)
        distractor_ids = [e["item_id"] for e in chosen]

        # The window is placed AFTER the options are chosen, because the
        # effective budget depends on how many tokens this item's own question
        # and options consume. Distractor selection above is untouched by it.
        gold = gold_index + 1
        try:
            win = windowed_passage(cfg, tok, r, options, gold)
        except P1ItemTooLong as exc:
            # Counted and reported, never silently truncated. Only this narrow
            # exception is caught, so a genuine bug in window placement still
            # stops the build.
            dropped.append({"item_id": r["item_id"], "reason": str(exc)})
            continue

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
            "full_context_tokens": None,
            "answer_in_passage": r["answer"] in win["passage"],
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
        if it["group_id"] in {d.rsplit(
                cfg_mod.require(cfg, "finetune.item_id_separator"), 1)[0]
                for d in it["distractor_item_ids"]}:
            raise P1DataError(
                f"{it['item_id']}: a distractor came from the item's own "
                f"article, which can make it a second correct answer."
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
    if n_dropped / len(rows) > MAX_EMPTY_ROW_FRACTION:
        raise P1DataError(
            f"{lang}: {n_dropped}/{len(rows)} rows ({n_dropped / len(rows):.1%}) "
            f"have a question and four options that alone exceed "
            f"max_seq_tokens, above the {MAX_EMPTY_ROW_FRACTION:.0%} ceiling. "
            f"A drop rate this high would itself be a language-correlated "
            f"artefact; stop and look at the corpus."
        )

    eval_ids = select_heldout_eval(cfg, lang, heldout_items)
    gold_by_id = {it["item_id"]: it["gold"] for it in heldout_items}

    # Recorded, not asserted. This is the share of items on which the gold
    # cannot be picked out by digit-class alone -- the give-away that the
    # topical + surface matching exists to remove. It will not reach 1.0: an
    # oddly shaped answer sometimes has no same-class neighbour, and the builder
    # widens rather than fabricating one. Auditing the number beats assuming it.
    homogeneity = {
        part: surface_homogeneity(items)
        for part, items in (("train", train_items), ("heldout", heldout_items))
    }

    return {
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


def load_partition(cfg: dict[str, Any], lang: str, partition: str,
                   manifest: dict[str, Any] | None = None) -> list[dict]:
    """Rebuild one partition and verify it against the frozen manifest.

    This is the only sanctioned way for training or evaluation code to get P1
    items -- it is the P1 analogue of `data.load_language`.
    """
    manifest = manifest or load_split_manifest()
    built = build_language(cfg, lang)
    verify_against_manifest(built, manifest)
    if partition == "train":
        return built["train_items"]
    if partition == "heldout":
        return built["heldout_items"]
    if partition == "heldout_eval":
        keep = set(built["heldout_eval_item_ids"])
        return [it for it in built["heldout_items"] if it["item_id"] in keep]
    raise P1DataError(
        f"unknown partition {partition!r}; allowed "
        f"{PARTITIONS + ('heldout_eval',)}")


def build_p1_prompt(cfg: dict[str, Any], item: dict[str, Any]) -> str:
    """Render a P1 item through P0's frozen template, unchanged."""
    return build_prompt(cfg, item)
