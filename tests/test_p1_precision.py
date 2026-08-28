"""LoRA is really applied, the merge really lands, quantization really happens.

Every failure this file guards against is SILENT. An adapter that matched no
module trains to a falling loss and changes nothing. A merge that does not fold
the weights writes a checkpoint identical to the base. A quantizer that fails
over to FP16 produces a perfectly plausible accuracy column that is secretly a
second copy of the FP16 column. In all three cases the FT arm would simply
reproduce the base arm, and delta(L,Q) would be an elaborate zero.

Two tiers:

  * CPU tests build a tiny randomly-initialised Qwen2 and exercise the real
    code paths -- masking, the training loop, adapter attachment, merge, save,
    reload. These run everywhere and are the bulk of the coverage.
  * CUDA tests cover what only bitsandbytes on a real GPU can show. They skip
    off-GPU and run on Kaggle.

The tiny model is a STRUCTURAL fixture. It has random weights, produces no
measurement, and nothing derived from it may reach the paper.
"""

import json

import pytest
import torch

from quantlang import config as cfg_mod
from quantlang import finetune
from quantlang import model as model_mod
from quantlang import p1data

peft = pytest.importorskip("peft", reason="peft is required for P1 training")

HAS_CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not HAS_CUDA, reason="needs a CUDA GPU")


@pytest.fixture(scope="module")
def cfg():
    return cfg_mod.load()


@pytest.fixture(scope="module")
def tiny_model():
    """A 2-layer Qwen2 with random weights: same architecture, trivial size."""
    from transformers import AutoConfig, AutoModelForCausalLM

    conf = AutoConfig.for_model(
        "qwen2",
        vocab_size=512, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256, tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(conf)


def _fresh(tiny_model):
    """A clean copy, so one test's training cannot leak into another."""
    import copy
    return copy.deepcopy(tiny_model)


def _examples(n=8, length=12):
    """Synthetic encoded examples with the P1 label layout."""
    out = []
    for i in range(n):
        ids = [(i * 7 + j) % 500 + 1 for j in range(length)]
        label = 100 + (i % 4)
        ids.append(label)
        out.append({
            "input_ids": ids,
            "labels": [finetune.IGNORE_INDEX] * (len(ids) - 1) + [label],
            "item_id": f"synthetic#{i}",
            "label_id": label,
        })
    return out


# --------------------------------------------------------------------------- #
# the training objective
# --------------------------------------------------------------------------- #

class _StubTokenizer:
    """Minimal tokenizer: one id per whitespace token, with truncation."""

    def __init__(self):
        self.truncation_side = "left"

    def __call__(self, text, truncation=False, max_length=None, **kw):
        ids = [(abs(hash(t)) % 400) + 1 for t in text.split()]
        if truncation and max_length and len(ids) > max_length:
            ids = ids[-max_length:] if self.truncation_side == "left" else ids[:max_length]
        return {"input_ids": ids}


def _item(gold=2):
    return {
        "item_id": "Article 000#0",
        "lang": "eng_Latn",
        "passage": "A passage about a subject that continues for a while.",
        "question": "What is the subject?",
        "options": ["first option", "second option", "third option", "fourth option"],
        "gold": gold,
        "gold_text": "second option",
    }


def test_loss_covers_only_the_answer_letter_token(cfg):
    """The single most important line in the training setup.

    letter_logit reads one token. Training on anything else teaches the model
    to produce something the scorer never looks at.
    """
    option_ids = [11, 22, 33, 44]
    ex = finetune.encode_example(cfg, _StubTokenizer(), _item(gold=2),
                                 option_ids, max_seq_tokens=64)
    labels = ex["labels"]
    assert labels[-1] == 22, "the label must be the gold letter's token id"
    assert all(v == finetune.IGNORE_INDEX for v in labels[:-1]), (
        "every prompt position must be masked out of the loss")
    assert len(labels) == len(ex["input_ids"])
    assert ex["input_ids"][-1] == 22, (
        "the label token must also be the input at the final position")


@pytest.mark.parametrize("gold,expected", [(1, 11), (2, 22), (3, 33), (4, 44)])
def test_label_tracks_the_gold_letter(cfg, gold, expected):
    ex = finetune.encode_example(cfg, _StubTokenizer(), _item(gold=gold),
                                 [11, 22, 33, 44], max_seq_tokens=64)
    assert ex["labels"][-1] == expected


def test_prompt_is_truncated_to_leave_room_for_the_label(cfg):
    """Otherwise the label would be pushed out of the window on a long passage."""
    item = _item()
    item["passage"] = "word " * 500
    ex = finetune.encode_example(cfg, _StubTokenizer(), item, [11, 22, 33, 44],
                                 max_seq_tokens=32)
    assert len(ex["input_ids"]) <= 32
    assert ex["input_ids"][-1] == 22


def test_encode_rejects_out_of_range_gold(cfg):
    item = _item()
    item["gold"] = 9
    with pytest.raises(finetune.FineTuneError, match="outside 1..4"):
        finetune.encode_example(cfg, _StubTokenizer(), item, [11, 22, 33, 44],
                                max_seq_tokens=64)


# --------------------------------------------------------------------------- #
# the adapter is real
# --------------------------------------------------------------------------- #

def test_lora_attaches_and_reports_parameter_counts(cfg, tiny_model):
    """Counts are consistent and LoRA trains a minority of the parameters.

    No threshold on trainable_percent here: this fixture has hidden_size=64
    against r=16, so LoRA is ~19% of it. On the real 3B the same configuration
    is a fraction of a percent, and the smoke run reports the actual figure.
    """
    peft_model, counts = finetune.attach_adapter(cfg, _fresh(tiny_model))
    assert counts["trainable_parameters"] > 0
    assert counts["base_parameters"] > counts["trainable_parameters"], (
        "LoRA must train fewer parameters than it freezes")
    assert (counts["base_parameters"] + counts["trainable_parameters"]
            == counts["total_parameters"])
    assert 0.0 < counts["trainable_percent"] < 100.0


def test_adapter_is_detected_as_applied(cfg, tiny_model):
    peft_model, _ = finetune.attach_adapter(cfg, _fresh(tiny_model))
    counts = finetune.assert_adapter_applied(peft_model)
    assert counts["lora_layers"] > 0
    assert counts["lora_A"] == counts["lora_B"] > 0


def test_only_lora_parameters_require_grad(cfg, tiny_model):
    peft_model, _ = finetune.attach_adapter(cfg, _fresh(tiny_model))
    for name, param in peft_model.named_parameters():
        if param.requires_grad:
            assert "lora_" in name, f"{name} is trainable but is not a LoRA weight"


def test_lora_parameters_are_fp32(cfg, tiny_model):
    """fp16 LoRA updates are small enough to vanish into the base weights."""
    peft_model, _ = finetune.attach_adapter(cfg, _fresh(tiny_model))
    for _, param in peft_model.named_parameters():
        if param.requires_grad:
            assert param.dtype is torch.float32


def test_an_adapter_that_matches_nothing_is_rejected(cfg, tiny_model):
    """The silent failure this guard exists for."""
    import copy
    broken = copy.deepcopy(cfg)
    broken["finetune"]["lora"]["target_modules"] = ["no_such_module"]
    with pytest.raises(Exception):
        finetune.attach_adapter(broken, _fresh(tiny_model))


def test_training_changes_the_adapter_weights(cfg, tiny_model):
    """Loss falling is not proof of learning; the weights must actually move."""
    peft_model, _ = finetune.attach_adapter(cfg, _fresh(tiny_model))
    before = {n: p.detach().clone()
              for n, p in peft_model.named_parameters() if p.requires_grad}
    finetune.train_lora(cfg, peft_model, _examples(), seed=1, device="cpu",
                        log_every=10_000)
    moved = sum(
        1 for n, p in peft_model.named_parameters()
        if p.requires_grad and not torch.equal(p.detach(), before[n]))
    assert moved > 0, "no LoRA parameter changed during training"


def test_training_changes_model_output(cfg, tiny_model):
    """The end-to-end consequence: the fine-tuned model is a different function."""
    base = _fresh(tiny_model).eval()
    ids = torch.tensor([[5, 9, 14, 21, 33]])
    with torch.no_grad():
        before = base(ids).logits.clone()

    peft_model, _ = finetune.attach_adapter(cfg, base)
    finetune.train_lora(cfg, peft_model, _examples(), seed=1, device="cpu",
                        log_every=10_000)
    merged = peft_model.merge_and_unload().eval()
    with torch.no_grad():
        after = merged(ids).logits

    assert not torch.allclose(before, after, atol=1e-6), (
        "merged model output is identical to the base -- the adapter had no "
        "effect, so the FT arm would just reproduce the base arm")


def test_training_is_deterministic_given_the_seed(cfg, tiny_model):
    def run(seed):
        peft_model, _ = finetune.attach_adapter(cfg, _fresh(tiny_model))
        stats = finetune.train_lora(cfg, peft_model, _examples(), seed=seed,
                                    device="cpu", log_every=10_000)
        return stats["loss_final"]

    assert run(3) == run(3)


def test_different_seeds_give_different_runs(cfg, tiny_model):
    """The sensitivity probe would measure nothing otherwise.

    Compared on the trained WEIGHTS, not on the loss. An untrained random model
    sits at ln(vocab) for every example, so its loss is degenerate and would
    look seed-independent even when the runs genuinely differ. Enough examples
    are used to produce several optimizer steps, since a single full-batch step
    accumulates the same gradient regardless of shuffle order.
    """
    def run(seed):
        peft_model, _ = finetune.attach_adapter(cfg, _fresh(tiny_model))
        stats = finetune.train_lora(cfg, peft_model, _examples(n=32), seed=seed,
                                    device="cpu", log_every=10_000)
        assert stats["n_optimizer_steps"] > 1, "order cannot matter in one step"
        weights = {n: p.detach().clone()
                   for n, p in peft_model.named_parameters() if p.requires_grad}
        return weights

    a, b = run(3), run(4)
    assert any(not torch.equal(a[n], b[n]) for n in a), (
        "two seeds produced identical adapters; the sensitivity probe would "
        "measure nothing")


def test_training_reports_the_metadata_the_brief_requires(cfg, tiny_model):
    peft_model, _ = finetune.attach_adapter(cfg, _fresh(tiny_model))
    stats = finetune.train_lora(cfg, peft_model, _examples(), seed=1,
                                device="cpu", log_every=10_000)
    for key in ("n_examples", "n_optimizer_steps", "loss_final", "loss_mean",
                "train_seconds", "lr_schedule", "base_learning_rate"):
        assert key in stats, key
    assert stats["n_optimizer_steps"] > 0


def test_non_finite_loss_stops_training(cfg, tiny_model):
    """Training through a NaN produces a checkpoint nobody can interpret."""
    peft_model, _ = finetune.attach_adapter(cfg, _fresh(tiny_model))
    bad = _examples(n=2)
    bad[0]["labels"] = [finetune.IGNORE_INDEX] * len(bad[0]["input_ids"])
    bad[1]["labels"] = [finetune.IGNORE_INDEX] * len(bad[1]["input_ids"])
    with pytest.raises(finetune.FineTuneError, match="non-finite loss"):
        finetune.train_lora(cfg, peft_model, bad, seed=1, device="cpu",
                            log_every=10_000)


# --------------------------------------------------------------------------- #
# merge and reload
# --------------------------------------------------------------------------- #

def test_merge_writes_a_loadable_fp16_checkpoint(cfg, tiny_model, tmp_path):
    from transformers import AutoModelForCausalLM

    peft_model, _ = finetune.attach_adapter(cfg, _fresh(tiny_model))
    finetune.train_lora(cfg, peft_model, _examples(), seed=1, device="cpu",
                        log_every=10_000)

    info = finetune.merge_and_save(peft_model, _StubSaveTokenizer(), tmp_path / "m")
    assert info["parameter_dtypes"] == ["torch.float16"]
    assert info["total_bytes"] > 0
    assert info["sha256"]

    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path / "m")
    assert reloaded is not None
    assert {str(p.dtype) for p in reloaded.parameters()} == {"torch.float16"}


def test_merged_checkpoint_has_no_lora_modules_left(cfg, tiny_model, tmp_path):
    """A leftover adapter would stay full-precision through quantization."""
    from transformers import AutoModelForCausalLM
    from peft.tuners.lora import LoraLayer

    peft_model, _ = finetune.attach_adapter(cfg, _fresh(tiny_model))
    finetune.merge_and_save(peft_model, _StubSaveTokenizer(), tmp_path / "m")
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path / "m")
    assert not any(isinstance(m, LoraLayer) for m in reloaded.modules())
    assert not any("lora" in n.lower() for n, _ in reloaded.named_parameters())


def test_checkpoint_digest_detects_a_changed_weight(tmp_path):
    """The digest recorded in run metadata has to be sensitive."""
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "weights.bin").write_bytes(b"\x00" * 64)
    first = finetune.directory_digest(d)
    (d / "weights.bin").write_bytes(b"\x00" * 63 + b"\x01")
    assert finetune.directory_digest(d)["sha256"] != first["sha256"]
    assert first["total_bytes"] == 64


class _StubSaveTokenizer:
    def save_pretrained(self, path):
        from pathlib import Path
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer_stub.json").write_text("{}", encoding="utf-8")


# --------------------------------------------------------------------------- #
# aliases keep P0 and P1 apart
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("lang", ["eng_Latn", "ben_Beng", "sin_Sinh",
                                  "asm_Beng", "npi_Deva"])
def test_ft_alias_is_distinct_and_filename_safe(lang):
    alias = finetune.ft_alias("qwen2.5-3b-instruct", lang)
    assert alias == f"qwen2.5-3b-instruct-ft-{lang}"
    assert alias != "qwen2.5-3b-instruct", "P1 must not collide with P0 rows"
    assert "__" not in alias, "'__' is the raw-filename field separator"


def test_ft_alias_rejects_an_unsafe_base_alias():
    with pytest.raises(finetune.FineTuneError, match="__"):
        finetune.ft_alias("bad__alias", "eng_Latn")


# --------------------------------------------------------------------------- #
# model loading stays P0-compatible
# --------------------------------------------------------------------------- #

def test_local_checkpoint_argument_is_optional():
    """P0 calls load() without it and must behave exactly as before."""
    import inspect
    sig = inspect.signature(model_mod.load)
    assert sig.parameters["local_checkpoint"].default is None
    p0_params = ["cfg", "hf_id", "revision", "precision", "device"]
    assert list(sig.parameters)[:5] == p0_params, (
        "P0's positional call signature must not move")


def test_precision_kwargs_are_unchanged_by_p1():
    """The quantizers applied to a merged checkpoint must be P0's quantizers."""
    fp16 = model_mod._quant_kwargs("fp16")
    assert fp16["dtype"] is torch.float16
    assert "quantization_config" not in fp16

    int8 = model_mod._quant_kwargs("int8_llmint8")["quantization_config"]
    assert int8.load_in_8bit is True
    assert int8.llm_int8_threshold == 6.0

    nf4 = model_mod._quant_kwargs("nf4")["quantization_config"]
    assert nf4.load_in_4bit is True
    assert nf4.bnb_4bit_quant_type == "nf4"
    assert nf4.bnb_4bit_use_double_quant is True
    assert nf4.bnb_4bit_compute_dtype is torch.float16


def test_unknown_precision_is_rejected():
    with pytest.raises(ValueError, match="unknown precision"):
        model_mod._quant_kwargs("fp8")


# --------------------------------------------------------------------------- #
# the fallback guard, exercised on fakes
# --------------------------------------------------------------------------- #

class _FakeLinear8bitLt(torch.nn.Module):
    pass


class _FakeLinear4bit(torch.nn.Module):
    pass


def _fake_model(kind):
    root = torch.nn.Module()
    if kind == "int8":
        root.layer = _FakeLinear8bitLt()
        root.layer.__class__.__name__ = "Linear8bitLt"
    elif kind == "nf4":
        root.layer = _FakeLinear4bit()
        root.layer.__class__.__name__ = "Linear4bit"
    else:
        root.layer = torch.nn.Linear(4, 4).half()
    return root


def test_int8_without_quantized_layers_is_rejected():
    """The exact shape of a silent FP16 fallback."""
    with pytest.raises(model_mod.PrecisionError, match="Linear8bitLt"):
        model_mod.assert_precision_applied(_fake_model("fp16"), "int8_llmint8")


def test_nf4_without_quantized_layers_is_rejected():
    with pytest.raises(model_mod.PrecisionError, match="Linear4bit"):
        model_mod.assert_precision_applied(_fake_model("fp16"), "nf4")


def test_fp16_with_quantized_layers_is_rejected():
    with pytest.raises(model_mod.PrecisionError, match="quantized layers present"):
        model_mod.assert_precision_applied(_fake_model("nf4"), "fp16")


def test_fp16_with_wrong_dtype_is_rejected():
    root = torch.nn.Module()
    root.layer = torch.nn.Linear(4, 4).float()
    with pytest.raises(model_mod.PrecisionError, match="parameter dtypes"):
        model_mod.assert_precision_applied(root, "fp16")


def test_fp16_accepts_a_genuine_fp16_model():
    counts = model_mod.assert_precision_applied(_fake_model("fp16"), "fp16")
    assert counts["Linear8bitLt"] == 0 and counts["Linear4bit"] == 0


# --------------------------------------------------------------------------- #
# CUDA only: what bitsandbytes alone can prove
# --------------------------------------------------------------------------- #

@cuda_only
def test_quantized_loads_produce_genuinely_different_weights(cfg, tmp_path):
    """FP16, INT8 and NF4 of one checkpoint must not be three copies of one
    thing. Run on Kaggle; the real evidence is in the smoke report."""
    from transformers import AutoConfig, AutoModelForCausalLM

    conf = AutoConfig.for_model(
        "qwen2", vocab_size=512, hidden_size=256, intermediate_size=512,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256, tie_word_embeddings=False)
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(conf).half().save_pretrained(tmp_path / "ck")

    ids = torch.tensor([[3, 8, 15, 22]], device="cuda:0")
    outputs = {}
    for precision in ("fp16", "int8_llmint8", "nf4"):
        kwargs = model_mod._quant_kwargs(precision)
        kwargs["device_map"] = {"": "cuda:0"}
        m = AutoModelForCausalLM.from_pretrained(tmp_path / "ck", **kwargs).eval()
        model_mod.assert_precision_applied(m, precision)
        with torch.no_grad():
            outputs[precision] = m(ids).logits[0, -1, :].float().cpu()
        del m
        torch.cuda.empty_cache()

    assert not torch.allclose(outputs["fp16"], outputs["nf4"], atol=1e-4), (
        "NF4 output is identical to FP16 -- quantization did not take effect")
    assert not torch.allclose(outputs["fp16"], outputs["int8_llmint8"], atol=1e-4)


@cuda_only
def test_smoke_report_records_every_required_check():
    """Reads the artefact the Kaggle smoke run leaves behind, when present."""
    from quantlang.config import REPO_ROOT
    path = REPO_ROOT / "results" / "P1" / "metadata" / "p1_smoke_report.json"
    if not path.exists():
        pytest.skip("no smoke report present yet")
    report = json.loads(path.read_text(encoding="utf-8"))
    for n in ("1_training_completes", "2_adapter_applied", "3_merge_succeeds",
              "4_merged_checkpoint_loads", "5_runs_through_p0_evaluator",
              "6_answer_letter_behaviour", "7_no_silent_fp16_fallback",
              "8_no_belebele_in_training"):
        assert n in report["checks"], n
        assert report["checks"][n]["pass"], n
