# H5, third model: the ordered prediction, recorded before NF4 is run

Written after the Gemma FP16 gate and **before any Gemma quantized cell exists**.
The git timestamp is the evidence. H5 itself is unchanged and is not restated
here; see docs/H5_PREREGISTRATION.md.

## Why this document exists

The BLOOMZ arm tested H5 with two burden points per language, which is the
minimum that can show a difference and cannot show an ordering. Gemma's measured
burden lands BETWEEN BLOOMZ and Qwen for four of five languages, so the three
models now form an ordered series. An ordering is a much stronger test than a
difference, and it is only a test if the expected order is fixed in advance.

Median tokens per rendered BELEBELE item, measured on the frozen 900-item set:

    language    BLOOMZ    Gemma-2-2b    Qwen2.5-3B
    eng_Latn       147          149           149     (no spread -- control)
    npi_Deva       169          294           602
    ben_Beng       170          374           700
    asm_Beng       202          446           750
    sin_Sinh      1108          560           962     (Gemma LOWEST)

## The prediction

Under H5, FP16-to-NF4 degradation should follow the burden ordering within each
language.

**P4.** For ben_Beng, npi_Deva and asm_Beng, Gemma's NF4 degradation falls
between BLOOMZ's and Qwen's:

    BLOOMZ  <=  Gemma  <=  Qwen

Known values, from completed runs:

    language    BLOOMZ NF4 loss    Gemma    Qwen NF4 loss
    ben_Beng             0.0011        ?          0.0456
    npi_Deva             0.0111        ?          0.0333
    asm_Beng             0.0156        ?          0.0367

**P5.** For eng_Latn, where all three tokenizers agree to within two tokens,
Gemma's degradation is NOT ordered with respect to the other two. English is the
control: no burden spread, so no burden-driven ordering is expected.

**P6.** sin_Sinh is a NEW cell, not a continuation. BLOOMZ floored there and
carries no degradation value, so no three-way ordering exists. Gemma has the
LOWEST Sinhala burden of the three models (560 against Qwen's 962), so under H5
its Sinhala degradation should be below Qwen's 0.0611 -- the largest single
degradation measured anywhere in this project.

## What counts as failure

P4 failing in any of the three languages is reported as failing. A model landing
outside the BLOOMZ-Qwen interval, or the ordering inverting, is evidence against
a burden-degradation association and is reported as such.

These predictions are NOT added to the pre-registered H5 family. That family is
fixed at the three directional predictions in docs/H5_PREREGISTRATION.md and the
Bangla result stands as reported: difference -0.0444, 95% CI [-0.0767, -0.0133],
pre-registered 3-test Holm p = 0.0225, broader 8-comparison Holm p = 0.0525.
Both corrections are disclosed in the paper. P4-P6 are a separate, later,
exploratory-but-pre-stated robustness test and are labelled that way.

## Confounds, unchanged and uncontrolled

Three models differing in parameter count (3.0B / 2.6B / 3.1B), architecture,
pretraining corpus, training date (2022 / 2024 / 2024), instruction tuning, and
language exposure. Burden co-varies with training-data coverage: a model that
tokenizes a language cheaply usually saw more of it. Nothing here isolates
tokenization as a cause, and the paper must not claim it does.
