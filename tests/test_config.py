"""Config must refuse to hand out unknown values or misnamed precisions."""

import pytest
import yaml

from quantlang import config


def test_loads_and_validates():
    assert config.load()["benchmark"]["name"] == "belebele"


def test_item_id_key_is_pinned_and_composite():
    cfg = config.load()
    key = config.require(cfg, "benchmark.item_id_key")
    assert key == "link#question_number", (
        "question_number alone takes 2 distinct values across 900 items and "
        "cannot key an item."
    )


def test_require_raises_on_null_rather_than_returning_none():
    cfg = config.load()
    cfg["models"][0]["revision"] = None
    cfg.setdefault("probe", {})["unpinned"] = None
    with pytest.raises(config.ConfigError, match="NOT YET KNOWN"):
        config.require(cfg, "probe.unpinned")


def test_require_raises_on_missing_key():
    with pytest.raises(config.ConfigError, match="Missing config key"):
        config.require(config.load(), "benchmark.no_such_field")


@pytest.mark.parametrize("bad", ["int4", "int8", "INT8", "nf4_bnb"])
def test_bare_int_naming_is_rejected(bad, tmp_path):
    """Bare INT4/INT8 would imply coverage beyond bitsandbytes."""
    cfg = yaml.safe_load(config.CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["precisions"] = [bad]
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(config.ConfigError, match="Unknown precision"):
        config.load(p)


def test_reference_language_must_be_present(tmp_path):
    cfg = yaml.safe_load(config.CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["benchmark"]["languages"] = ["ben_Beng", "npi_Deva"]
    p = tmp_path / "noref.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(config.ConfigError, match="reference_language"):
        config.load(p)
