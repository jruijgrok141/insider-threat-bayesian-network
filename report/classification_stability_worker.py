"""Worker for isolated classification-stability seeds (spawned subprocess)."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

_REPORT_DIR = Path(__file__).resolve().parent
if str(_REPORT_DIR) not in sys.path:
    sys.path.insert(0, str(_REPORT_DIR))

import pandas as pd
from sklearn.model_selection import train_test_split

from export_figures import TARGET, build_insider_bn, classification_scores
from reproducibility import set_global_seed


def run_seed(data_path: Path, seed: int) -> list[dict]:
    set_global_seed(seed)
    bn, _ = build_insider_bn()
    data = pd.read_csv(data_path)
    cls_pool = data.sample(n=300, random_state=seed)
    train_cls, test_cls = train_test_split(
        cls_pool,
        train_size=100,
        test_size=100,
        random_state=seed,
        stratify=cls_pool[TARGET],
    )
    rows = []
    for result in classification_scores(bn, train_cls, test_cls, seed=seed):
        rows.append(
            {
                "seed": seed,
                "Model": result["Model"],
                "AUC": result["AUC"],
                "Accuracy": result["Accuracy"],
            }
        )
    return rows


if __name__ == "__main__":
    data_path = Path(sys.argv[1])
    seed = int(sys.argv[2])
    import json

    print(json.dumps(run_seed(data_path, seed)))
