"""Verify the Kaggle access token actually authenticates.

Naive checks give FALSE PASSES. Several Kaggle endpoints (notably
`dataset_list`) succeed unauthenticated and return an empty list, so calling
one and seeing no exception proves nothing -- an invalid token produces output
identical to a valid one. This script therefore runs a DIFFERENTIAL test: the
same auth-gated probes are issued with the real token and with a deliberately
invalid one. Identical results mean the real token is not authenticating.

Token format: Kaggle's current tokens are `KGAT_<32 hex>` (37 chars) stored as
plain text. `kaggle.json` is the legacy username/key format and is consulted
only after the token sources below.

Run:  python scripts/preflight_kaggle.py
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys

KAGGLE_DIR = pathlib.Path.home() / ".kaggle"

# Resolution order mirrors kagglesdk.kaggle_env.get_access_token_from_env.
TOKEN_FILES = [
    ("~/.kaggle/access_token", KAGGLE_DIR / "access_token"),
    ("~/.kaggle/access_token.txt", KAGGLE_DIR / "access_token.txt"),
]
LEGACY_JSON = KAGGLE_DIR / "kaggle.json"
TOKEN_RE = re.compile(r"^KGAT_[0-9a-f]{32}$")

PROBE_SRC = """
import kaggle, json
kaggle.api.authenticate()
out = {}
probes = {
    "kernels_list_mine": lambda: kaggle.api.kernels_list(mine=True),
    "competitions_list": lambda: kaggle.api.competitions_list(),
}
for name, fn in probes.items():
    try:
        out[name] = f"OK:{len(list(fn()))}"
    except Exception as e:
        s = str(e)
        out[name] = "401" if "401" in s else ("403" if "403" in s else type(e).__name__)
print(json.dumps(out))
"""


def find_token() -> tuple[str | None, str | None]:
    env = os.environ.get("KAGGLE_API_TOKEN")
    if env:
        return env.strip(), "KAGGLE_API_TOKEN env var"
    for label, path in TOKEN_FILES:
        if not path.exists():
            continue
        raw = path.read_bytes()
        if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
            raise SystemExit(
                f"FATAL: {label} is UTF-16 encoded, which corrupts the token.\n"
                f"PowerShell 5.1's `>` and `Out-File` write UTF-16 by default.\n"
                f"Rewrite it from Git Bash, or use "
                f"`Set-Content -Encoding ascii -NoNewline`."
            )
        tok = raw.decode("utf-8-sig").strip()
        if tok:
            return tok, label
    return None, None


def _run(env_overrides: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    for k in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        env.pop(k, None)
    if env_overrides:
        env.update(env_overrides)
    proc = subprocess.run(
        [sys.executable, "-c", PROBE_SRC],
        capture_output=True, text=True, env=env, timeout=180,
    )
    for line in reversed(proc.stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise SystemExit(f"probe produced no result:\n{proc.stdout}\n{proc.stderr[-800:]}")


def main() -> int:
    token, source = find_token()
    if not token:
        raise SystemExit(
            "FATAL: no Kaggle access token found.\n"
            "  Looked for: KAGGLE_API_TOKEN, ~/.kaggle/access_token, "
            "~/.kaggle/access_token.txt\n"
            "  kaggle.com -> Settings -> API -> copy the token, then save it to\n"
            "  ~/.kaggle/access_token (plain text, no quotes, no trailing newline)."
        )

    print(f"token source : {source}")
    print(f"token length : {len(token)}")
    print(f"token prefix : {token[:5]}...")
    if not TOKEN_RE.match(token):
        print(
            "  NOTE: token does not match the expected KGAT_<32 hex> shape.\n"
            "        A hint only -- the live test below decides."
        )
    if LEGACY_JSON.exists():
        print(
            f"  NOTE: legacy kaggle.json present; it is only consulted after the\n"
            f"        token above. Delete it to avoid confusion."
        )

    print("\nrunning differential auth test (real vs deliberately invalid token)...")
    real = _run({"KAGGLE_API_TOKEN": token})
    ctrl = _run({"KAGGLE_API_TOKEN": "KGAT_" + "0" * 32})

    print(f"\n  {'probe':22s} {'real':>10s} {'invalid':>10s}")
    for name in sorted(real):
        print(f"  {name:22s} {real[name]:>10s} {ctrl.get(name, '?'):>10s}")

    if real == ctrl:
        raise SystemExit(
            "\nFATAL: the real token behaves identically to an invalid one.\n"
            "It is NOT authenticating.\n\n"
            "Fix:\n"
            "  1. kaggle.com -> Settings -> API -> copy the token with the copy\n"
            "     button (do not retype it).\n"
            "  2. Save it to ~/.kaggle/access_token as plain ASCII text.\n"
            "  3. Kaggle requires PHONE VERIFICATION before the kernels API and GPU\n"
            "     are usable: Settings -> Phone Verification. Without it these\n"
            "     endpoints refuse the request even with a valid token."
        )

    failed = [n for n, v in real.items() if not v.startswith("OK")]
    if failed:
        raise SystemExit(
            f"\nFATAL: authenticated probe(s) failed: {failed}\n"
            f"The token differs from the control, so it IS being read and is\n"
            f"distinguishable from garbage -- but these endpoints were refused.\n"
            f"The usual cause is an account that has not completed PHONE\n"
            f"VERIFICATION, which gates the kernels API and GPU access.\n"
            f"  kaggle.com -> Settings -> Phone Verification"
        )

    print("\nPASS: token authenticates and kernel endpoints are reachable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
