"""Decide whether a candidate model can be evaluated at all -- before any GPU.

Adding a second model to a study is cheap to start and expensive to abandon
halfway. Three things can make a model unusable here, and two of them are
detectable from the tokenizer alone, in seconds, on a laptop:

  1. `letter_logit` scoring reads the logit of " A".." D" and REQUIRES each to
     be exactly one token. Phi-3.5-mini fails this: its letter tokens are not
     single tokens, so `model.option_token_ids` raises and no amount of GPU time
     helps. Discovering that after an 8 GB download is pure waste.

  2. The model may be gated. That is a five-minute fix (accept the licence, mint
     a token) but only if you learn about it before scheduling a Kaggle session.

  3. FLOOR EFFECTS. A model at chance in a language cannot show quantization
     degradation, because there is nothing left to lose -- its small delta means
     "already broken", not "robust". This needs a GPU and is what `--load` and
     the FP16-first protocol are for; the tokenizer pass cannot see it.

It also reports TOKENIZATION BURDEN per language, measured on the real frozen
BELEBELE items rather than a sample sentence. In P0 that quantity correlated
with NF4 degradation at rho=+0.900 (p=0.037, n=5) while base accuracy did not
(rho=-0.100), so it is the quantity a cross-model comparison is really about.
BELEBELE passages are parallel translations, so a difference in tokens per item
between two tokenizers is a fertility difference and not a content difference.

    python scripts/probe_model_compat.py --hf-id google/gemma-2-2b-it
    python scripts/probe_model_compat.py --hf-id bigscience/bloomz-3b --load
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantlang import config as cfg_mod  # noqa: E402
from quantlang import data as data_mod  # noqa: E402


def tokenizer_checks(cfg: dict, hf_id: str, revision: str | None) -> dict:
    """Everything decidable without weights."""
    from transformers import AutoTokenizer

    out: dict = {"hf_id": hf_id, "revision": revision}
    try:
        tok = AutoTokenizer.from_pretrained(hf_id, revision=revision)
    except Exception as exc:                       # gated, missing, offline
        gated = "gated repo" in str(exc).lower()
        out["tokenizer_loaded"] = False
        out["gated"] = gated
        out["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        out["ok"] = False
        return out
    out["tokenizer_loaded"] = True
    out["gated"] = False
    out["vocab_size"] = len(tok)

    letters = cfg_mod.require(cfg, "scoring.option_letters")
    prefix = cfg_mod.require(cfg, "scoring.option_prefix")
    enc = {l: tok.encode(f"{prefix}{l}", add_special_tokens=False) for l in letters}
    out["option_encodings"] = enc
    single = all(len(v) == 1 for v in enc.values())
    ids = [v[0] for v in enc.values() if len(v) == 1]
    out["letters_single_token"] = single
    out["letters_distinct"] = len(set(ids)) == len(ids) if single else False
    out["ok"] = single and out["letters_distinct"]
    return out


def burden(cfg: dict, hf_id: str, revision: str | None, langs: list[str],
           n: int) -> dict:
    """Tokens per rendered BELEBELE prompt, on the frozen item set."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id, revision=revision)
    per: dict = {}
    for lang in langs:
        rows = data_mod.load_language(cfg, lang)[:n]
        lens = [len(tok.encode(data_mod.build_prompt(cfg, r),
                               add_special_tokens=False)) for r in rows]
        per[lang] = {"n_items": len(lens),
                     "median_tokens": int(statistics.median(lens)),
                     "mean_tokens": round(statistics.mean(lens), 1),
                     "max_tokens": max(lens)}
    return per


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", required=True)
    ap.add_argument("--revision", default=None,
                    help="pin before any run that goes in the paper")
    ap.add_argument("--langs", nargs="*", default=None,
                    help="defaults to benchmark.languages")
    ap.add_argument("--n-items", type=int, default=60,
                    help="items per language for the burden estimate")
    ap.add_argument("--load", action="store_true",
                    help="also load the weights at FP16 and score one item. "
                         "Needs a GPU; catches architectures the plain causal-LM "
                         "path cannot open, such as multimodal checkpoints.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    cfg = cfg_mod.load()
    langs = args.langs or cfg_mod.require(cfg, "benchmark.languages")

    report = {"model": tokenizer_checks(cfg, args.hf_id, args.revision)}
    m = report["model"]

    print(f"model      : {args.hf_id}")
    print(f"revision   : {args.revision or 'UNPINNED -- pin before the real run'}")
    if not m["tokenizer_loaded"]:
        print(f"\nFATAL: tokenizer would not load.")
        if m["gated"]:
            print("  This is a GATED repository. Accept the licence on the model "
                  "page, mint an access token, and expose it as HF_TOKEN "
                  "(Kaggle: Add-ons -> Secrets). Nothing else here can run first.")
        print(f"  {m['error']}")
        return 1

    print(f"vocab      : {m['vocab_size']:,}")
    print(f"\n=== letter_logit compatibility ===")
    for letter, ids in m["option_encodings"].items():
        mark = "ok" if len(ids) == 1 else f"FAIL ({len(ids)} tokens)"
        print(f"  {cfg_mod.require(cfg, 'scoring.option_prefix')!r}+{letter!r} "
              f"-> {ids}  {mark}")
    if not m["ok"]:
        print("\nFATAL: letter_logit scoring is unavailable for this tokenizer. "
              "Every answer letter must be exactly one distinct token. Choose a "
              "different model -- do NOT change the scoring method, which would "
              "make this model's numbers incomparable with P0's.")
        return 1
    print("  -> letter_logit is usable")

    print(f"\n=== tokenization burden (frozen BELEBELE items, "
          f"n<={args.n_items}/lang) ===")
    report["burden"] = burden(cfg, args.hf_id, args.revision, langs, args.n_items)
    base = None
    for lang, b in sorted(report["burden"].items(),
                          key=lambda kv: kv[1]["median_tokens"]):
        base = base or b["median_tokens"]
        print(f"  {lang:10} median {b['median_tokens']:5d}  mean "
              f"{b['mean_tokens']:7.1f}  max {b['max_tokens']:5d}  "
              f"({b['median_tokens'] / base:.2f}x the cheapest language)")

    max_seq = cfg_mod.require(cfg, "scoring.max_input_tokens")
    over = {l: b["max_tokens"] for l, b in report["burden"].items()
            if b["max_tokens"] > max_seq}
    if over:
        print(f"\n  WARNING: {over} exceed scoring.max_input_tokens={max_seq}. "
              f"Those items truncate, and truncation is not equal across "
              f"languages -- check n_truncated in the eval output.")

    if args.load:
        print(f"\n=== loading FP16 weights on {args.device} ===")
        try:
            from quantlang import model as model_mod
            tok, mdl, _ = model_mod.load(cfg, args.hf_id, args.revision or "main",
                                         "fp16", args.device)
            ids = model_mod.option_token_ids(cfg, tok)
            rows = data_mod.load_language(cfg, langs[0])[:1]
            import torch
            enc = tok(data_mod.build_prompt(cfg, rows[0]), return_tensors="pt")
            with torch.no_grad():
                lg = mdl(**{k: v.to(args.device) for k, v in enc.items()}).logits
            print(f"  loaded and scored one {langs[0]} item; "
                  f"letter logits {[round(float(lg[0, -1, i]), 3) for i in ids]}")
            report["load"] = {"ok": True}
        except Exception as exc:
            print(f"  FATAL: {type(exc).__name__}: {str(exc)[:300]}")
            print("  A multimodal checkpoint (e.g. gemma-3-4b-it) does not open "
                  "through the plain causal-LM path and needs its text decoder "
                  "reached explicitly. Prefer a text-only sibling.")
            report["load"] = {"ok": False, "error": str(exc)[:300]}
            return 1

    print("\nCOMPATIBLE: this model can be evaluated by the P0 procedure "
          "unchanged.")
    print("  Next: FP16 only, all languages, before spending anything on INT8 "
          "or NF4. A model at chance in a language cannot show quantization "
          "degradation, and its small delta would mean 'already broken' rather "
          "than 'robust'.")

    if args.outdir:
        p = Path(args.outdir) / f"model_compat_{args.hf_id.replace('/', '_')}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
