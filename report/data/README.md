# Pinned synthetic learning dataset

`synthetic_cases_n2000_seed42.csv` contains 2000 cases drawn from the expert BN with `gum.initRandom(42)` and `setTopologicalVarOrder()`.

`classification_stability/seed_XXX.json` stores pinned AUC/accuracy for each stability resample (seeds 42–71). Hill Climbing in pyAgrum can show residual run-to-run variation; these JSON files keep report and notebook outputs identical. Delete a seed file to recompute it via `classification_stability_worker.py`.

All structure-learning and classification experiments load the synthetic CSV so every run uses identical inputs.

```bash
python -c "from export_figures import build_insider_bn; from reproducibility import draw_synthetic_cases, set_global_seed, N_SYNTHETIC_CASES, DEFAULT_SEED, SYNTHETIC_CASES_CSV; set_global_seed(DEFAULT_SEED); bn,_=build_insider_bn(); draw_synthetic_cases(bn,N_SYNTHETIC_CASES,DEFAULT_SEED).to_csv(SYNTHETIC_CASES_CSV,index=False)"
```

Regenerate synthetic cases only when the BN definition or sampling settings change.
