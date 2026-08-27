"""Statistics for the language x precision experiment.

The one thing that matters most in this module is how items are paired.

BELEBELE is a parallel benchmark: the same 900 questions appear in every
language. That makes the cross-language comparison item-paired, which is a
materially tighter test than resampling each language independently -- but only
if items are actually matched by identity. They are matched here by `item_id`
and never by row position, because BELEBELE's row order is NOT aligned across
languages. On the pinned revision, zero of Assamese's 900 rows share an index
with their English counterpart and 580 of Bangla's differ. Positional pairing
would compare unrelated questions while looking entirely healthy.

Every vector below is therefore reindexed onto the manifest's canonical
item_id order before any arithmetic happens.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from .schema import SchemaError, load_manifest


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Used instead of the normal approximation because
    accuracies here sit near the 0.25 chance floor, where the normal interval
    misbehaves and can cross below it."""
    if n == 0:
        raise ValueError("Wilson CI is undefined for n=0")
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    lo, hi = max(0.0, centre - half), min(1.0, centre + half)
    # At the boundaries the true Wilson bound is exactly 0 or 1; floating point
    # leaves a ~1e-19 residue that would make the interval appear to exclude
    # its own point estimate.
    if k == 0:
        lo = 0.0
    if k == n:
        hi = 1.0
    return (lo, hi)


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided exact McNemar on paired binary correctness vectors.

    Exact rather than chi-square: the discordant counts between FP16 and a
    quantized run can be small, and the chi-square approximation is unreliable
    there in exactly the direction that manufactures significance.
    """
    a = np.asarray(a).astype(int)
    b = np.asarray(b).astype(int)
    if a.shape != b.shape:
        raise ValueError(f"unpaired vectors: {a.shape} vs {b.shape}")
    n01 = int(np.sum((a == 1) & (b == 0)))
    n10 = int(np.sum((a == 0) & (b == 1)))
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down adjustment. Returns adjusted p-values."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (key, p) in enumerate(items):
        running = max(running, (m - i) * p)
        adjusted[key] = min(1.0, running)
    return adjusted


class Aligned:
    """Correctness vectors for every cell, aligned on the canonical item order.

    Construction fails loudly if a cell is missing an item. There is no
    intersect-what-we-have mode: a partially aligned comparison is a different
    experiment from the one being reported.
    """

    def __init__(self, tidy: pd.DataFrame, manifest: dict | None = None):
        manifest = manifest or load_manifest()
        self.item_ids: list[str] = list(manifest["item_ids"])
        self.n = len(self.item_ids)
        self._cells: dict[tuple[str, str, str], np.ndarray] = {}

        for (model, lang, prec), grp in tidy.groupby(["model", "lang", "precision"]):
            s = grp.set_index("item_id")["correct"]
            if s.index.has_duplicates:
                raise SchemaError(f"({model}, {lang}, {prec}) has duplicate item_ids")
            unknown = int((~s.index.isin(self.item_ids)).sum())
            if unknown:
                raise SchemaError(
                    f"({model}, {lang}, {prec}) has {unknown} item_id(s) that "
                    f"are not in the frozen manifest. This is not the pinned "
                    f"benchmark."
                )
            # Reindex onto the canonical order. THIS is the pairing.
            aligned = s.reindex(self.item_ids)
            if aligned.isnull().any():
                raise SchemaError(
                    f"({model}, {lang}, {prec}) is missing "
                    f"{int(aligned.isnull().sum())} of {self.n} manifest items. "
                    f"Incomplete cells are not analysed."
                )
            self._cells[(model, lang, prec)] = aligned.to_numpy(dtype=int)

    def keys(self) -> Iterable[tuple[str, str, str]]:
        return self._cells.keys()

    def __contains__(self, key) -> bool:
        return key in self._cells

    def get(self, model: str, lang: str, precision: str) -> np.ndarray:
        try:
            return self._cells[(model, lang, precision)]
        except KeyError:
            raise SchemaError(
                f"cell ({model}, {lang}, {precision}) was never measured. It is "
                f"reported as missing, not estimated."
            ) from None


def paired_bootstrap_interaction(
    fp16_lang: np.ndarray, quant_lang: np.ndarray,
    fp16_ref: np.ndarray, quant_ref: np.ndarray,
    n_boot: int = 10000, seed: int = 0,
    percentiles: tuple[float, float] = (2.5, 97.5),
) -> dict[str, float]:
    """Bootstrap the interaction term.

        d_int = [Acc_fp16(lang) - Acc_q(lang)] - [Acc_fp16(ref) - Acc_q(ref)]

    Positive d_int means quantization costs this language MORE than it costs the
    reference language. This -- not the raw accuracy table -- is the claim the
    paper's main hypothesis actually makes, so it is the quantity that gets a
    confidence interval.

    A single set of item indices is drawn per replicate and applied to BOTH
    languages, which is only legitimate because all four vectors have already
    been reindexed onto the same canonical item order.
    """
    vecs = (fp16_lang, quant_lang, fp16_ref, quant_ref)
    n = len(fp16_lang)
    if any(len(v) != n for v in vecs):
        raise ValueError("interaction requires four equal-length aligned vectors")

    point = ((fp16_lang.mean() - quant_lang.mean())
             - (fp16_ref.mean() - quant_ref.mean()))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = ((fp16_lang[idx].mean(axis=1) - quant_lang[idx].mean(axis=1))
             - (fp16_ref[idx].mean(axis=1) - quant_ref[idx].mean(axis=1)))

    lo, hi = np.percentile(boots, list(percentiles))
    # Two-sided bootstrap p for H0: d_int = 0.
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {
        "delta_interaction": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_bootstrap": float(min(1.0, p)),
        "n_boot": int(n_boot),
    }
