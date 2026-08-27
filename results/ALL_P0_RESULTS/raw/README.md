# results/raw/ — APPEND ONLY

Files here are produced by exactly one process:

    kaggle kernels output <kernel> -p results/raw/

Nothing else may write to this directory. No script in this repo opens a file
here for writing, and `tests/test_raw_is_append_only.py` fails if one does.

Why this is absolute: every number in the paper is traceable to a file in this
directory, which is traceable to a real Kaggle run. A single hand-edited or
regenerated file breaks that chain, and no reader — including future you — can
tell which numbers are still trustworthy.

Files are never edited, never deleted, never regenerated locally. A run that
produced a bad file is superseded by a new run writing a new file, and the bad
file stays, so the record of what happened stays complete.

Naming (written by the kernel, parsed by scripts/build_tidy.py):

    {model_alias}__{precision}__{lang}.csv
    {model_alias}__{precision}__{lang}.manifest.json

The manifest records git SHA, model revision, dataset revision, library
versions, GPU, scoring method and wall time. A CSV without its manifest is not
admissible and build_tidy.py rejects it.
