"""results/raw/ is populated only by `kaggle kernels output`.

This test greps the repo's own source for any code that could write there. It
is deliberately blunt: a false positive costs one comment, while a false
negative costs the provenance chain that every number in the paper rests on.
"""

import re
from pathlib import Path

from quantlang.config import REPO_ROOT

WRITE_HINTS = re.compile(
    r"(open\s*\([^)]*['\"][wax]|to_csv|write_text|write_bytes|savefig|"
    r"\.unlink\(|shutil\.(copy|move|rmtree)|os\.remove|mkdir)",
    re.IGNORECASE,
)
RAW_REF = re.compile(r"results[^A-Za-z0-9]{1,2}raw", re.IGNORECASE)

# This test itself, and the directory's own README, legitimately mention the path.
EXEMPT = {"test_raw_is_append_only.py"}


def _sources():
    for p in REPO_ROOT.rglob("*.py"):
        if ".git" in p.parts or "__pycache__" in p.parts:
            continue
        if p.name in EXEMPT:
            continue
        yield p


def test_no_source_file_writes_to_results_raw():
    offenders = []
    for path in _sources():
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            if RAW_REF.search(line) and WRITE_HINTS.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "Source code appears to write into results/raw/, which is append-only "
        "and may only be populated by `kaggle kernels output`:\n  "
        + "\n  ".join(offenders)
    )


def test_raw_directory_exists_and_documents_the_rule():
    readme = REPO_ROOT / "results" / "raw" / "README.md"
    assert readme.exists(), "results/raw/README.md is missing"
    assert "APPEND ONLY" in readme.read_text(encoding="utf-8")


def test_smoke_output_is_not_committed():
    """Smoke runs must never be able to reach the paper."""
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "results/smoke/" in gitignore
