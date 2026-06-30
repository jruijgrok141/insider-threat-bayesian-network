# Assignment coverage — Insider-threat BN

Checklist against the Bayesian Reasoning and Learning assignment requirements.

## Task 1 — Expert BN in pyAgrum

| Requirement | Status | Where |
|-------------|--------|-------|
| Domain + scope (non-trivial) | ✅ | Cyber security / insider threat; proposal + report §1 |
| 5–10 nodes | ✅ | 8 nodes |
| Variables + states | ✅ | `states` in notebook |
| Structure **not** learned automatically (Task 1) | ✅ | Manual DAG + expert CPTs |
| Causal relations justified | ✅ | Proposal, report §2.1, DAG figure |
| Parameters (not arbitrary) | ✅ | Expert + noisy-OR; report §2.2 |
| Inference: marginals | ✅ | Notebook: marginals pivot |
| Inference: conditionals | ✅ | Notebook: conditional queries |
| Plausibility / scenarios | ✅ | Scenario table + chart |
| Sensitivity / validation | ✅ | Notebook: leak sensitivity |
| Short description in report | ✅ | `report/Report_Insider_Threat_BN.*` |

## Task 2.1 — Tutorials

| Requirement | Status | Note |
|-------------|--------|------|
| pyAgrum tutorials | ⚪ | Not submitted; briefly mentioned in report §2.3 |

## Task 2.2 — Learning from data

| Requirement | Status | Where |
|-------------|--------|-------|
| Data generated from own network | ✅ | `BNDatabaseGenerator` |
| *n* = 100, 500, 1000 | ✅ | `recovery_df` |
| Search-and-score (Hill Climbing + BIC) | ✅ | `HillClimb` |
| Constraint-based (MIIC) | ✅ | `MIIC` |
| Structure comparison (F1, Hamming, visual) | ✅ | Tables + `gnb.sideBySide` |
| Binary class variable | ✅ | `InsiderThreatIncident` |
| Train 100 / Test 100 | ✅ | `train_cls`, `test_cls` |
| Original / learned / naive Bayes | ✅ | Classification cells |
| ROC curve (minimum) | ✅ | ROC plot + AUC table |

## Task 3 — Report (~6 pages)

| Section | Status | File |
|---------|--------|------|
| Abstract + notation tables | ✅ | `report/Report_Insider_Threat_BN.{md,tex}` |
| Introduction | ✅ | same |
| Methods (RQ1–RQ4, setup, learning) | ✅ | same |
| Results (inference, structure, ROC, noisy-OR) | ✅ | same + `figures/*.pdf` |
| Conclusions and discussion | ✅ | same |
| References | ✅ | same (bib in `.tex`; inline in `.md`) |

## Project-specific (proposal, not in assignment PDF)

| Topic | Status |
|-------|--------|
| Noisy-OR (RQ4) | ✅ | Main model + comparison in prototype |
| Chockalingam et al. (2017) | ✅ | Proposal + report references |

## Artifacts in this repository

- **Prototype:** `notebooks/insider_threat_bn_prototype.ipynb`
- **Report:** `report/Report_Insider_Threat_BN.md` (+ `.tex`, `.pdf`)
- **Proposal:** `proposal/Research_Proposal_Cybersecurity_Insider_Threat.md`
