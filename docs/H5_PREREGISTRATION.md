# H5: tokenization burden and quantization degradation — pre-registration

Written and committed **before any replication-model evaluation has been run**.
At the time of this commit the only model evaluated in this project is
Qwen2.5-3B-Instruct. The git timestamp is the evidence for that claim.

## The hypothesis, as specified

> **H5.** Quantization-induced degradation is associated with tokenization
> burden, measured as mean tokens per BELEBELE item. Across models, languages
> with substantially higher tokenization burden are expected to show greater
> FP16-to-NF4 degradation.

**This is an association / mechanism hypothesis. No causal claim is made or
will be made.** Tokenization burden is not manipulated; it is a property of a
tokenizer that co-varies with training-data composition, script, and language
resource level. A correlation between burden and degradation is consistent with
several causal structures and distinguishes none of them.

H5 is fixed as of this document and is not revised in light of any result.

## What motivated it

Within the completed P0 grid (Qwen2.5-3B-Instruct, five languages, 900 items
each), FP16-to-NF4 degradation tracks tokens per item and does **not** track
base accuracy:

    NF4 loss vs median tokens/item : Spearman rho = +0.900, p = 0.037
    NF4 loss vs base FP16 accuracy : Spearman rho = -0.100, p = 0.873
    tokens/item vs base accuracy   : Spearman rho = -0.300, p = 0.624

    lang        median tokens   NF4 loss   base FP16 acc
    eng_Latn         149          0.0111       0.8956
    npi_Deva         601          0.0333       0.4467
    ben_Beng         700          0.0456       0.6278
    asm_Beng         750          0.0367       0.4744
    sin_Sinh         962          0.0611       0.4756

This is n = 5 and rho = 0.900 is the second-highest value attainable at that
sample size, so a single language moving would break it. It is a motivating
observation, not a finding, and it is reported as such.

BELEBELE passages are parallel translations of the same content, so a
difference in tokens per item between two tokenizers is a fertility difference
rather than a content difference. That is what makes a cross-model comparison
on these items interpretable at all.

## Why BLOOMZ-3b is the test

BLOOM's training corpus covers Bengali, Assamese and Nepali but not Sinhala.
Measured on the same frozen BELEBELE items (median tokens per rendered prompt):

    lang          Qwen2.5-3B   BLOOMZ-3b     ratio
    eng_Latn          149         153        1.03x
    ben_Beng          700         164        0.23x   (4.3x cheaper)
    npi_Deva          601         176        0.29x   (3.4x cheaper)
    asm_Beng          750         197        0.26x   (3.8x cheaper)
    sin_Sinh          962        1168        1.21x   (more expensive)

The burden ordering is close to inverted for three of the five languages while
Sinhala moves the other way. That makes BLOOMZ a test of a stated prediction
rather than a second data point.

## Directional predictions, fixed in advance

Under H5, relative to Qwen2.5-3B-Instruct:

1. BLOOMZ FP16-to-NF4 degradation for **ben_Beng, npi_Deva and asm_Beng** is
   SMALLER than Qwen's for those languages, and closer to the English level.
2. BLOOMZ FP16-to-NF4 degradation for **sin_Sinh** is NOT smaller than Qwen's,
   and may be larger.
3. Within BLOOMZ alone, degradation is rank-associated with tokens per item
   across the five languages.

Any of these failing is reported. H5 is not adjusted to accommodate an outcome.

## What would make a language uninterpretable

A model at or near chance in a language cannot exhibit quantization
degradation: there is nothing left to lose, and a small delta then means
"already broken" rather than "robust". This is the same floor effect that
constrains the P1-Strong 3-epoch arm.

Chance on a four-option item is 0.25. **The floor gate is applied on FP16
accuracy alone, before any INT8 or NF4 evaluation is run**, and is fixed here:

    Wilson 95% lower bound > 0.30   -> language is interpretable for this model
    otherwise                       -> language is reported as floored, and no
                                       degradation claim is made for that cell

The gate is applied to every language uniformly. Languages are not dropped or
retained on the basis of whether they favour H5, and floored cells are reported
as floored rather than omitted.

## Analysis, fixed in advance

* Degradation per cell: FP16 accuracy minus quantized accuracy, paired on the
  same 900 items, exact McNemar.
* H5 test 1 and 2: BLOOMZ degradation compared with Qwen degradation for the
  same language, as a difference of paired degradations across models. The two
  models are evaluated on identical items but are different models, so this is
  not a within-item paired contrast and its interval is obtained by bootstrap
  over items.
* H5 test 3: Spearman rank correlation between median tokens per item and NF4
  degradation, within each model separately and pooled across models. With five
  languages per model the within-model test is severely underpowered and is
  reported with that stated, not as confirmatory.
* Holm correction within the family of the three directional predictions.

## Confounds that are stated, not controlled

* **Instruction tuning.** Qwen2.5-3B-Instruct is instruction-tuned; BLOOMZ-3b is
  xP3-tuned but a weaker instruction follower. Differences between the models
  are not attributable to tokenization alone.
* **Model size and family.** 3.1B versus 3.0B parameters, entirely different
  architectures, training corpora and dates (2024 versus 2022).
* **Burden co-varies with training-data coverage.** A language a model
  tokenizes cheaply is usually one it saw more of. Burden and competence cannot
  be separated by this design.

These make H5 a test of an association across a second tokenizer, not an
isolation of a mechanism, and the paper must say so.
