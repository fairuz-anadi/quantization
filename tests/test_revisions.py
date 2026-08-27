"""Revisions must be real 40-char commit SHAs, not tags or branch names."""

import json
import re

import pytest

from quantlang import config

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
MOVING_REFS = {"main", "master", "HEAD", "latest", "refs/heads/main"}


def test_dataset_revision_is_a_commit_sha():
    sha = config.require(config.load(), "benchmark.hf_revision")
    assert SHA_RE.match(sha), f"not a commit SHA: {sha!r}"


def test_every_model_revision_is_a_commit_sha():
    for m in config.load()["models"]:
        assert m.get("revision") is not None, f"{m['alias']} is unpinned"
        assert SHA_RE.match(m["revision"]), f"{m['alias']}: not a commit SHA"


@pytest.mark.parametrize("ref", sorted(MOVING_REFS))
def test_moving_refs_would_not_pass_the_sha_check(ref):
    """A branch name moves; pinning to one makes runs unreproducible."""
    assert not SHA_RE.match(ref)


def test_revisions_provenance_file_agrees_with_config():
    cfg = config.load()
    prov = json.loads(
        (config.REPO_ROOT / "configs" / "revisions.json").read_text(encoding="utf-8")
    )
    assert prov["dataset"]["sha"] == config.require(cfg, "benchmark.hf_revision")
    for m in cfg["models"]:
        assert prov["models"][m["alias"]]["sha"] == m["revision"]
        assert prov["models"][m["alias"]]["hf_id"] == m["hf_id"]


def test_pinned_model_is_not_gated():
    """A gated repo needs HF_TOKEN in Kaggle Secrets; surface it here, not on the GPU."""
    prov = json.loads(
        (config.REPO_ROOT / "configs" / "revisions.json").read_text(encoding="utf-8")
    )
    gated = [a for a, i in prov["models"].items() if i.get("gated")]
    assert not gated, f"gated model(s) {gated}: HF_TOKEN must be set as a Kaggle Secret"
