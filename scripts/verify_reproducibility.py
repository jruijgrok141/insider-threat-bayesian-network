"""Verify that export_figures.py produces identical CSV outputs on consecutive runs."""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXPORT = REPO / "report" / "export_figures.py"
FIG_DIR = REPO / "report" / "figures"


def csv_hashes() -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(FIG_DIR.glob("table_*.csv"))
    }


def run_export() -> None:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "42"
    subprocess.check_call([sys.executable, str(EXPORT)], cwd=REPO, env=env)


def main() -> None:
    if not EXPORT.is_file():
        raise FileNotFoundError(EXPORT)
    print("Running export_figures.py (pass 1)...")
    run_export()
    first = csv_hashes()
    print("Running export_figures.py (pass 2)...")
    run_export()
    second = csv_hashes()
    if first != second:
        changed = [name for name in sorted(set(first) | set(second)) if first.get(name) != second.get(name)]
        raise SystemExit(f"Reproducibility check failed; CSV hashes differ for: {', '.join(changed)}")
    print(f"OK: {len(first)} CSV table(s) are identical across two consecutive export runs.")


if __name__ == "__main__":
    main()
