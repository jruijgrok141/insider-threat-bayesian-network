# Insider-threat Bayesian network (pyAgrum)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](requirements.txt)

An **8-node causal Bayesian network** for insider-related cyber security risk, built for the master course *Bayesian Reasoning and Learning* (Open University of the Netherlands).

**Author:** Jan Ruijgrok · **Institution:** Open University of the Netherlands

## Quick links

| Resource | Description |
|----------|-------------|
| [Report (PDF)](report/Report_Insider_Threat_BN.pdf) | Short report (~7 pages main text + references) |
| [Report (Markdown)](report/Report_Insider_Threat_BN.md) | Same content, readable on GitHub |
| [Notebook (HTML)](docs/insider_threat_bn_prototype.html) | Executed 8-node analysis — no Python required (regenerate after notebook or `export_figures.py`) |
| [Notebook (`.ipynb`)](notebooks/insider_threat_bn_prototype.ipynb) | Full reproducible pipeline |
| [Toy walkthrough (HTML)](docs/toy_4node_walkthrough.html) | Executed 4-node teaching notebook |
| [Toy walkthrough (`.ipynb`)](notebooks/toy_4node_walkthrough.ipynb) | Minimal 4-node pipeline (same steps as assignment) |
| [Research proposal](proposal/Research_Proposal_Cybersecurity_Insider_Threat.md) | Domain scope and research questions |

![Causal DAG (8 nodes)](report/figures/fig01_dag.png)

## What this project covers

- **Task 1:** Manual causal BN (8 nodes), inference, scenario validation, sensitivity analysis
- **Task 2.2a:** Structure learning from synthetic data (Hill Climbing + BIC, MIIC) at *n* ∈ {100, 500, 1000}
- **Task 2.2b:** Classification (original BN, learned BN, naive Bayes); Train/Test = 100/100; ROC/AUC
- **Noisy-OR:** Parsimonious incident-node parameterisation with monotonic multi-cause risk

All learning data are **synthetic** (generated from the ground-truth BN). No identifiable employee records were used. This is a **proof of concept**, not operational SOC validation.

## Repository layout

```
insider-threat-bayesian-network/
├── README.md
├── requirements.txt
├── report/                  # Written report + figure export script
│   ├── Report_Insider_Threat_BN.pdf
│   ├── Report_Insider_Threat_BN.{tex,md}
│   ├── export_figures.py
│   └── figures/
├── notebooks/               # Prototype + toy walkthrough notebooks
├── proposal/                # Research proposal
├── references/              # Open-access background papers
├── docs/                    # HTML exports (GitHub Pages)
└── scripts/
    └── build_delivery.py    # Assemble a submission zip folder locally
```

## Reproduce the analysis

**Requirements:** Python 3.10+

```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook insider_threat_bn_prototype.ipynb
```

Run all cells in order. The notebook covers Task 1 inference, single-run structure learning, classification (reference split + 30-resample stability), and noisy-OR. **Structure stability over 30 resamples** (report Table/Figure 3) is produced by `report/export_figures.py`, which also writes the CSV tables cited in the report.

To regenerate report figures and numeric tables (authoritative for §3.2–3.4):

```bash
python report/export_figures.py
```

To refresh the executed HTML in `docs/` (after running the notebook or `export_figures.py`):

```bash
jupyter nbconvert --execute --to html notebooks/insider_threat_bn_prototype.ipynb --output insider_threat_bn_prototype --output-dir docs
```

To regenerate the notebook from its Python builder:

```bash
python notebooks/build_insider_threat_notebook.py
```

## GitHub Pages

Enable **Settings → Pages → branch `main`, folder `/docs`** to publish the executed notebook at:

`https://jruijgrok141.github.io/insider-threat-bayesian-network/`

## Local delivery bundle

For course submission, assemble a minimal folder (PDF + notebook + references):

```bash
python scripts/build_delivery.py
```

Output is written to `dist/` (gitignored).

## License

MIT — see [LICENSE](LICENSE). Background reference PDFs remain subject to their original publishers' terms.

## Acknowledgments

Course: *Bayesian Reasoning and Learning*, Open University of the Netherlands. Built with [pyAgrum](https://agrum.gitlab.io/).
