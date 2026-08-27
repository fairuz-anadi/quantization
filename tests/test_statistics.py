"""Invariants for the statistics layer.

The first test in this file is the important one. An earlier draft of the
analysis paired items across languages by sorted row index. BELEBELE's row
order is not aligned across languages, so that silently compared unrelated
questions while every intermediate number looked entirely plausible. These
tests fail if anything reintroduces positional pairing.
"""

import hashlib

import numpy as np
import pandas as pd
import pytest

from quantlang.schema import TIDY_COLUMNS, SchemaError, load_manifest
from quantlang.statistics import (
    Aligned, holm, mcnemar_exact, paired_bootstrap_interaction, wilson_ci,
)


def _stable_correct(item_id: str) -> bool:
    """Deterministic across processes. Python's str hash is salted per run."""
    return int(hashlib.md5(item_id.encode()).hexdigest(), 16) % 3 == 0


@pytest.fixture(scope="module")
def manifest():
    return load_manifest()


def _cell(manifest, lang, precision, correct_of, order=None, model="m"):
    """One cell as tidy rows. `order` controls ROW order only, never content."""
    gold_by_id = manifest["gold_by_item_id"]
    ids = list(order or manifest["item_ids"])
    rows = []
    for item_id in ids:
        gold = gold_by_id[item_id]
        correct = correct_of(item_id)
        rows.append({
            "model": model, "model_revision": "rev0", "precision": precision,
            "lang": lang, "item_id": item_id,
            "pred": gold if correct else (gold % 4) + 1,
            "gold": gold, "correct": int(correct),
        })
    return rows


# --------------------------------------------------------------------------- #
# the regression that matters
# --------------------------------------------------------------------------- #
def test_alignment_ignores_row_order(manifest):
    """A cell's aligned vector must not depend on the order rows arrived in."""
    ids = manifest["item_ids"]
    correct_of = _stable_correct

    natural = _cell(manifest, "eng_Latn", "fp16", correct_of, order=ids)
    shuffled_ids = list(reversed(ids))
    shuffled = _cell(manifest, "eng_Latn", "fp16", correct_of, order=shuffled_ids)

    a = Aligned(pd.DataFrame(natural)[list(TIDY_COLUMNS)])
    b = Aligned(pd.DataFrame(shuffled)[list(TIDY_COLUMNS)])
    np.testing.assert_array_equal(
        a.get("m", "eng_Latn", "fp16"), b.get("m", "eng_Latn", "fp16")
    )


def test_positional_pairing_would_have_given_a_different_answer(manifest):
    """Guard rail: prove the ordering hazard is real, not theoretical.

    If this ever starts failing because the two pairings agree, the fixture has
    stopped exercising the hazard and the test above is no longer protective.
    """
    ids = manifest["item_ids"]
    correct_of = _stable_correct

    eng = pd.DataFrame(_cell(manifest, "eng_Latn", "fp16", correct_of, order=ids))
    ben = pd.DataFrame(
        _cell(manifest, "ben_Beng", "fp16", correct_of, order=list(reversed(ids)))
    )
    aligned = Aligned(pd.concat([eng, ben])[list(TIDY_COLUMNS)])

    def discordant(a, b):
        return int(np.sum(np.asarray(a) != np.asarray(b)))

    keyed = discordant(aligned.get("m", "eng_Latn", "fp16"),
                       aligned.get("m", "ben_Beng", "fp16"))
    positional = discordant(eng["correct"].to_numpy(), ben["correct"].to_numpy())

    # Per-item outcomes are identical by construction, so item-keyed pairing
    # must see zero disagreement; positional pairing manufactures hundreds.
    assert keyed == 0
    assert positional > 100, (
        "positional pairing agreed with item-keyed pairing; the fixture no "
        "longer exercises the row-order hazard"
    )


def test_incomplete_cell_is_rejected_not_trimmed(manifest):
    rows = _cell(manifest, "eng_Latn", "fp16", lambda i: True)[:-1]
    with pytest.raises(SchemaError, match="is missing 1 of 900"):
        Aligned(pd.DataFrame(rows)[list(TIDY_COLUMNS)])


def test_missing_cell_raises_rather_than_returning_a_default(manifest):
    rows = _cell(manifest, "eng_Latn", "fp16", lambda i: True)
    aligned = Aligned(pd.DataFrame(rows)[list(TIDY_COLUMNS)])
    with pytest.raises(SchemaError, match="never measured"):
        aligned.get("m", "ben_Beng", "nf4")


# --------------------------------------------------------------------------- #
# estimator behaviour
# --------------------------------------------------------------------------- #
def test_wilson_ci_brackets_the_point_estimate():
    for k, n in [(0, 900), (225, 900), (650, 900), (900, 900)]:
        lo, hi = wilson_ci(k, n)
        assert 0.0 <= lo <= k / n <= hi <= 1.0


def test_wilson_ci_stays_inside_zero_one_at_the_boundaries():
    assert wilson_ci(0, 900)[0] == 0.0
    assert wilson_ci(900, 900)[1] == 1.0


def test_mcnemar_is_one_when_predictions_agree():
    v = np.array([1, 0, 1, 1, 0])
    assert mcnemar_exact(v, v) == 1.0


def test_mcnemar_is_small_for_a_large_one_sided_disagreement():
    a = np.ones(100, dtype=int)
    b = np.concatenate([np.zeros(30, dtype=int), np.ones(70, dtype=int)])
    assert mcnemar_exact(a, b) < 1e-6


def test_mcnemar_rejects_unpaired_vectors():
    with pytest.raises(ValueError):
        mcnemar_exact(np.ones(5), np.ones(6))


def test_holm_never_reduces_a_pvalue_and_is_bounded():
    raw = {"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.9}
    adj = holm(raw)
    assert set(adj) == set(raw)
    for k, p in raw.items():
        assert adj[k] >= p - 1e-12
        assert 0.0 <= adj[k] <= 1.0


def test_interaction_recovers_a_planted_effect():
    """English loses 10pp under quantization, Bangla loses 20pp: d_int = +0.10."""
    n = 900
    fp16_eng = np.ones(n, dtype=int)
    quant_eng = np.ones(n, dtype=int)
    quant_eng[:90] = 0                       # -10pp
    fp16_ben = np.ones(n, dtype=int)
    quant_ben = np.ones(n, dtype=int)
    quant_ben[:180] = 0                      # -20pp

    res = paired_bootstrap_interaction(fp16_ben, quant_ben, fp16_eng, quant_eng,
                                       n_boot=2000, seed=1)
    assert res["delta_interaction"] == pytest.approx(0.10, abs=1e-9)
    assert res["ci_low"] > 0, "a planted 10pp interaction should exclude zero"
    assert res["ci_low"] <= 0.10 <= res["ci_high"]


def test_interaction_finds_nothing_when_degradation_is_equal():
    n = 900
    fp16 = np.ones(n, dtype=int)
    quant = np.ones(n, dtype=int)
    quant[:90] = 0
    res = paired_bootstrap_interaction(fp16, quant.copy(), fp16, quant.copy(),
                                       n_boot=2000, seed=1)
    assert res["delta_interaction"] == pytest.approx(0.0, abs=1e-12)
    assert res["ci_low"] <= 0 <= res["ci_high"]


def test_interaction_is_deterministic_for_a_fixed_seed():
    rng = np.random.default_rng(0)
    vecs = [rng.integers(0, 2, 900) for _ in range(4)]
    a = paired_bootstrap_interaction(*vecs, n_boot=500, seed=42)
    b = paired_bootstrap_interaction(*vecs, n_boot=500, seed=42)
    assert a == b


def test_interaction_rejects_length_mismatch():
    with pytest.raises(ValueError, match="equal-length"):
        paired_bootstrap_interaction(
            np.ones(10), np.ones(10), np.ones(10), np.ones(9))
