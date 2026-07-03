"""Reproducibility helpers for pyAgrum-based pipelines in this repository."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyagrum as gum

if TYPE_CHECKING:
    from pyagrum import BayesNet

DEFAULT_SEED = 42
N_SYNTHETIC_CASES = 2000
DATA_DIR = Path(__file__).resolve().parent / "data"
SYNTHETIC_CASES_CSV = DATA_DIR / "synthetic_cases_n2000_seed42.csv"
CLASSIFICATION_STABILITY_CACHE = DATA_DIR / "classification_stability"


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """Fix pyAgrum, NumPy, and Python RNG state for repeatable runs."""
    gum.initRandom(seed)
    np.random.seed(seed)
    random.seed(seed)


def configure_data_generator(bn: BayesNet, seed: int = DEFAULT_SEED) -> gum.BNDatabaseGenerator:
    """Return a generator with deterministic variable order and fixed RNG."""
    set_global_seed(seed)
    gen = gum.BNDatabaseGenerator(bn)
    gen.setTopologicalVarOrder()
    return gen


def draw_synthetic_cases(
    bn: BayesNet,
    n: int = N_SYNTHETIC_CASES,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Draw *n* labelled cases from *bn* with a fixed seed."""
    gen = configure_data_generator(bn, seed)
    gen.drawSamples(n)
    return gen.to_pandas()


def load_synthetic_cases(
    bn: BayesNet | None = None,
    n: int = N_SYNTHETIC_CASES,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Load the pinned synthetic dataset, generating it once if missing."""
    if SYNTHETIC_CASES_CSV.is_file():
        return pd.read_csv(SYNTHETIC_CASES_CSV)
    if bn is None:
        raise ValueError("bn is required when synthetic_cases CSV is missing")
    df = draw_synthetic_cases(bn, n=n, seed=seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SYNTHETIC_CASES_CSV, index=False)
    return df


# Injected into generated notebooks (keep in sync with functions above).
NOTEBOOK_SETUP_CELL = """import random

def set_global_seed(seed):
    gum.initRandom(seed)
    np.random.seed(seed)
    random.seed(seed)

def draw_synthetic_cases(bn, n=2000, seed=SEED):
    set_global_seed(seed)
    gen = gum.BNDatabaseGenerator(bn)
    gen.setTopologicalVarOrder()
    gen.drawSamples(n)
    return gen.to_pandas()

set_global_seed(SEED)"""
