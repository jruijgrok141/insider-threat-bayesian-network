"""Assemble dist/ with report PDF, prototype notebook, HTML export, references, and README."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "dist"
PROTOTYPE_OUT = OUT / "prototype"
REFERENCES_OUT = OUT / "references"

NB_BUILDER = REPO / "notebooks" / "build_insider_threat_notebook.py"
REPORT_PDF = REPO / "report" / "Report_Insider_Threat_BN.pdf"
REPORT_TEX = REPO / "report" / "Report_Insider_Threat_BN.tex"
NOTEBOOK_SRC = REPO / "notebooks" / "insider_threat_bn_prototype.ipynb"
REFERENCES_SRC = REPO / "references"

REFERENCE_PDFS = (
    "axelrad2013_bayesian_network_insider_threats.pdf",
    "chockalingam2017_bayesian_networks_cyber_security_systematic_review.pdf",
    "greitzer2010_pnnl19665_behavioral_model_insider_threats.pdf",
)

README = """# Insider-threat Bayesian network — delivery bundle

Minimal folder for course submission: report, notebook, and background references.

## Contents

| Path | Description |
|------|-------------|
| `Report_Insider_Threat_BN.pdf` | Short report |
| `prototype/insider_threat_bn_prototype.ipynb` | Full analysis notebook |
| `prototype/insider_threat_bn_prototype.html` | Executed notebook (all outputs embedded) |
| `references/*.pdf` | Open-access background papers |

## Run the notebook

```bash
pip install -r requirements.txt
cd prototype
jupyter notebook insider_threat_bn_prototype.ipynb
```

Alternatively, open `prototype/insider_threat_bn_prototype.html` in a browser.

All learning data are **synthetic**. Proof of concept only — not operational SOC validation.
"""


def ensure_report_pdf() -> None:
    needs_build = not REPORT_PDF.is_file()
    if REPORT_TEX.is_file() and REPORT_PDF.is_file():
        needs_build = REPORT_TEX.stat().st_mtime > REPORT_PDF.stat().st_mtime
    if not needs_build:
        return
    if not REPORT_TEX.is_file():
        raise FileNotFoundError(f"Missing report source: {REPORT_TEX}")
    print("Building report PDF...")
    report_dir = REPO / "report"
    for _ in range(2):
        subprocess.check_call(
            ["pdflatex", "-interaction=nonstopmode", REPORT_TEX.name],
            cwd=report_dir,
        )


def ensure_notebook() -> None:
    if not NB_BUILDER.is_file():
        raise FileNotFoundError(f"Missing notebook builder: {NB_BUILDER}")
    print("Generating prototype notebook...")
    subprocess.check_call([sys.executable, str(NB_BUILDER)])


def export_notebook_html(notebook_src: Path, out_dir: Path) -> Path:
    print("Executing notebook and exporting HTML (may take a few minutes)...")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--execute",
            "--to",
            "html",
            str(notebook_src),
            "--output",
            "insider_threat_bn_prototype",
            f"--output-dir={out_dir}",
            "--ExecutePreprocessor.timeout=600",
        ]
    )
    html_path = out_dir / "insider_threat_bn_prototype.html"
    if not html_path.is_file():
        raise FileNotFoundError(f"HTML export failed: {html_path}")
    return html_path


def copy_references() -> None:
    REFERENCES_OUT.mkdir(parents=True, exist_ok=True)
    for name in REFERENCE_PDFS:
        src = REFERENCES_SRC / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing reference PDF: {src}")
        shutil.copy2(src, REFERENCES_OUT / name)


def copy_delivery() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    PROTOTYPE_OUT.mkdir()

    if not REPORT_PDF.is_file():
        raise FileNotFoundError(f"Missing report PDF: {REPORT_PDF}")
    if not NOTEBOOK_SRC.is_file():
        raise FileNotFoundError(f"Missing notebook: {NOTEBOOK_SRC}")

    shutil.copy2(REPORT_PDF, OUT / "Report_Insider_Threat_BN.pdf")
    shutil.copy2(NOTEBOOK_SRC, PROTOTYPE_OUT / "insider_threat_bn_prototype.ipynb")
    export_notebook_html(NOTEBOOK_SRC, PROTOTYPE_OUT)
    copy_references()
    (OUT / "README.md").write_text(README, encoding="utf-8")

    print(f"Delivery ready: {OUT}")
    print("  - Report_Insider_Threat_BN.pdf")
    print("  - prototype/insider_threat_bn_prototype.ipynb")
    print("  - prototype/insider_threat_bn_prototype.html")
    print("  - references/ (3 PDFs)")
    print("  - README.md")


def main() -> None:
    ensure_notebook()
    ensure_report_pdf()
    copy_delivery()


if __name__ == "__main__":
    main()
