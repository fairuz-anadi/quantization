# configs/

`experiment.yaml` — the single source of truth. `null` means NOT YET KNOWN and
is fatal at read time; it is resolved only by the script that owns it, never by
hand.

`item_id_manifest.json` — frozen output of `scripts/verify_item_ids.py`. Pins
the 900 BELEBELE item ids and their gold answers, with a sha256 over the id
list so hand-editing is detectable. Every run and every table is validated
against this file.

Regenerate only by rerunning the script. If the sha256 changes, the benchmark
contract changed and prior results are no longer comparable.
