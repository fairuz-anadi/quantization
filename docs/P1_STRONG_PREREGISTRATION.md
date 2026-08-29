# P1-Strong: pre-registration

Written **before** the run, and committed before any P1-Strong result exists.
The git timestamp on this file is the evidence for that claim.

## Why there is a second condition at all

P1-Standard (one epoch of LoRA, English and Bangla) returned a null on both
research questions. The fine-tune demonstrably worked -- Bangla cut its training
loss 0.4352 -> 0.2099, English 0.3396 -> 0.1168, and the merged models differ
from base at matched precision -- yet neither BELEBELE accuracy nor quantization
sensitivity moved.

A null from a single adaptation strength cannot distinguish two very different
claims:

  A. Language-specific adaptation does not affect this task or its quantization
     behaviour.
  B. *One epoch* of it was not enough to affect them.

P1-Strong exists to separate A from B. It is not run in the hope of reaching
significance, and the outcome is reported whichever way it falls.

## The design

Bangla only. It carries the strongest motivation: the largest measured
fine-tuning headroom (base 0.8500 [0.8052, 0.8860] against the 0.8956 ceiling,
versus English's 0.9567), the lowest base BELEBELE accuracy (0.6278), the
heaviest tokenization burden (700 median input tokens against English's 149),
and significant NF4 degradation already established in P0 (-4.6 pt,
p_holm = 0.013).

**Exactly one factor changes: `epochs` 1 -> 3.**

Frozen and identical to P1-Standard: the training corpus (multi-wiki-qa at the
pinned revision), the article-grouped 80/20 split and its seed, the equalised
3,732-item training partition, LoRA rank and target modules, learning rate,
schedule shape, optimizer, gradient accumulation, max sequence length, the
prompt template, the scoring method, and the frozen 900-item BELEBELE
evaluation set.

The cosine schedule spans the full run, so this is one 3-epoch schedule rather
than three 1-epoch schedules.

## Hypotheses

**H3.** Stronger language-specific adaptation improves Bangla BELEBELE accuracy
relative to the base model, measured at FP16.

**H4.** Stronger adaptation alters the magnitude of quantization degradation,
measured as the arm x precision interaction against the base arm.

## What each outcome means, decided in advance

* **H3 supported.** The P1-Standard null reflected insufficient adaptation
  strength, not an absent effect. Report as a dose-dependent effect, with the
  1-epoch cell as the intermediate dose.
* **H3 not supported.** The null is robust across a 3x range of adaptation
  strength. This strengthens the P1-Standard conclusion rather than weakening
  it, and is reported as the primary finding of this section.
* **H4 supported in either direction.** An adaptation-strength-dependent change
  in quantization sensitivity is the more interesting result and is reported as
  such, including if stronger adaptation makes NF4 *worse*.
* **Divergence between H3 and H4** -- for instance FP16 improving while NF4 does
  not follow -- is a trade-off finding and is reported in full.

## Analysis, fixed in advance

Primary comparisons, paired on the same 900 items:

1. FP16 accuracy, 3-epoch versus base (exact McNemar). This tests H3.
2. The arm x precision interaction for INT8 and NF4, 3-epoch versus base,
   bootstrap over items with a percentile CI. This tests H4.

Secondary, reported but not used to support a claim on its own: 3-epoch versus
1-epoch on the same items, which estimates the dose step directly.

Holm correction is applied within the H4 family (two precisions). The
P1-Standard results are **not** recomputed, re-corrected, or pooled with these;
the two conditions are reported side by side as separate rows.

## What would invalidate this run

* Training loss failing to fall below the 1-epoch run's 0.2099 last decile.
  Three epochs that do not fit better than one are not a stronger condition,
  and the comparison would be uninterpretable.
* Any P1-Standard artefact being overwritten. The epoch marker in the run name
  and result alias exists to prevent this; if a `ft-ben_Beng` file is modified
  by this run, stop.
