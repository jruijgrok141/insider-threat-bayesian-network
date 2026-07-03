"""Export report figures (PNG + PDF) for Report_Insider_Threat_BN."""
from __future__ import annotations

import itertools
import json
import os

# Single-thread BLAS/OpenMP avoids rare non-deterministic reductions in learners.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import tempfile
from pathlib import Path

import re

import subprocess

_REPORT_DIR = Path(__file__).resolve().parent
if str(_REPORT_DIR) not in sys.path:
    sys.path.insert(0, str(_REPORT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import networkx as nx
import numpy as np
import pandas as pd
import pyagrum as gum
import pyagrum.lib.bn_vs_bn as bnvsbn
import pyagrum.skbn as skbn
import seaborn as sns
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.model_selection import train_test_split

from reproducibility import (
    CLASSIFICATION_STABILITY_CACHE,
    DEFAULT_SEED,
    N_SYNTHETIC_CASES,
    SYNTHETIC_CASES_CSV,
    draw_synthetic_cases,
    load_synthetic_cases,
    set_global_seed,
)

TARGET = "InsiderThreatIncident"
SEED = DEFAULT_SEED
N_STABILITY_RUNS = 30
STABILITY_SEEDS = list(range(SEED, SEED + N_STABILITY_RUNS))
CI_LO, CI_HI = 2.5, 97.5
FIG_DIR = Path(__file__).resolve().parent / "figures"
WORKER = Path(__file__).resolve().parent / "classification_stability_worker.py"

states = {
    "JobDissatisfaction": ["low", "medium", "high"],
    "SecurityAwareness": ["low", "medium", "high"],
    "PolicyCompliance": ["good", "poor"],
    "ConcerningBehaviour": ["no", "yes"],
    "PrivilegedAccess": ["appropriate", "excessive"],
    "TechnicalControls": ["weak", "adequate"],
    "MonitoringAlert": ["no", "yes"],
    TARGET: ["no", "yes"],
}


def add_var(bn, name, labels):
    v = gum.LabelizedVariable(name, name, 0)
    for lab in labels:
        v.addLabel(lab)
    bn.add(v)


def noisy_or_p_yes(active, leak=0.94, inhibitions=(0.65, 0.60, 0.70)):
    p_no = leak
    for is_active, inh in zip(active, inhibitions):
        if is_active:
            p_no *= inh
    return 1.0 - p_no


def fill_incident_noisy_or(bn, leak=0.94, inhibitions=(0.65, 0.60, 0.70)):
    for cb in states["ConcerningBehaviour"]:
        for pa in states["PrivilegedAccess"]:
            for tc in states["TechnicalControls"]:
                active = (cb == "yes", pa == "excessive", tc == "weak")
                py = noisy_or_p_yes(active, leak=leak, inhibitions=inhibitions)
                bn.cpt(TARGET)[{
                    "ConcerningBehaviour": cb,
                    "PrivilegedAccess": pa,
                    "TechnicalControls": tc,
                }] = [1 - py, py]


def build_insider_bn(use_noisy_or_incident=True):
    bn = gum.BayesNet("InsiderThreatRisk")
    for name, labels in states.items():
        add_var(bn, name, labels)
    arcs = [
        ("JobDissatisfaction", "ConcerningBehaviour"),
        ("SecurityAwareness", "PolicyCompliance"),
        ("PolicyCompliance", "PrivilegedAccess"),
        ("ConcerningBehaviour", TARGET),
        ("PrivilegedAccess", TARGET),
        ("TechnicalControls", TARGET),
        ("TechnicalControls", "MonitoringAlert"),
        ("ConcerningBehaviour", "MonitoringAlert"),
    ]
    for a, b in arcs:
        bn.addArc(a, b)
    bn.cpt("JobDissatisfaction").fillWith([0.35, 0.45, 0.20])
    bn.cpt("SecurityAwareness").fillWith([0.25, 0.50, 0.25])
    bn.cpt("TechnicalControls").fillWith([0.28, 0.72])
    bn.cpt("ConcerningBehaviour")[{"JobDissatisfaction": "low"}] = [0.88, 0.12]
    bn.cpt("ConcerningBehaviour")[{"JobDissatisfaction": "medium"}] = [0.70, 0.30]
    bn.cpt("ConcerningBehaviour")[{"JobDissatisfaction": "high"}] = [0.45, 0.55]
    bn.cpt("PolicyCompliance")[{"SecurityAwareness": "low"}] = [0.25, 0.75]
    bn.cpt("PolicyCompliance")[{"SecurityAwareness": "medium"}] = [0.55, 0.45]
    bn.cpt("PolicyCompliance")[{"SecurityAwareness": "high"}] = [0.82, 0.18]
    bn.cpt("PrivilegedAccess")[{"PolicyCompliance": "good"}] = [0.90, 0.10]
    bn.cpt("PrivilegedAccess")[{"PolicyCompliance": "poor"}] = [0.35, 0.65]
    for tc in states["TechnicalControls"]:
        for cb in states["ConcerningBehaviour"]:
            p_alert = 0.08 if tc == "adequate" else 0.35
            if cb == "yes":
                p_alert = min(0.95, p_alert + 0.40)
            bn.cpt("MonitoringAlert")[{"TechnicalControls": tc, "ConcerningBehaviour": cb}] = [
                1 - p_alert,
                p_alert,
            ]
    if use_noisy_or_incident:
        fill_incident_noisy_or(bn)
    else:
        for cb in states["ConcerningBehaviour"]:
            for pa in states["PrivilegedAccess"]:
                for tc in states["TechnicalControls"]:
                    z = -2.2
                    z += 1.6 if cb == "yes" else 0.0
                    z += 1.3 if pa == "excessive" else 0.0
                    z += 1.1 if tc == "weak" else 0.0
                    py = 1 / (1 + np.exp(-z))
                    bn.cpt(TARGET)[{
                        "ConcerningBehaviour": cb,
                        "PrivilegedAccess": pa,
                        "TechnicalControls": tc,
                    }] = [1 - py, py]
    return bn, arcs


def p_incident_yes(model, evidence=None):
    ie = gum.LazyPropagation(model)
    if evidence:
        ie.setEvidence(evidence)
    ie.makeInference()
    return float(ie.posterior(TARGET)[{TARGET: "yes"}])


def save_fig(fig, stem: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  wrote {path}")


def _dag_display_name(name: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "\n", name)


def fig_dag(bn, arcs):
    g = nx.DiGraph()
    g.add_nodes_from(states.keys())
    g.add_edges_from(arcs)

    # Layered left-to-right: roots → intermediates → outcome; tech along the top.
    pos = {
        "JobDissatisfaction": (0.0, 1.8),
        "SecurityAwareness": (0.0, 0.0),
        "TechnicalControls": (2.0, 4.0),
        "ConcerningBehaviour": (2.2, 1.8),
        "PolicyCompliance": (2.2, 0.0),
        "PrivilegedAccess": (4.2, 0.0),
        "MonitoringAlert": (4.5, 4.0),
        TARGET: (6.5, 1.8),
    }

    category = {
        "JobDissatisfaction": "people",
        "SecurityAwareness": "people",
        "ConcerningBehaviour": "people",
        "PolicyCompliance": "process",
        "PrivilegedAccess": "process",
        "TechnicalControls": "technology",
        "MonitoringAlert": "technology",
        TARGET: "outcome",
    }
    fill = {
        "people": "#dbeafe",
        "process": "#dcfce7",
        "technology": "#fef9c3",
        "outcome": "#fecaca",
    }
    stroke = {
        "people": "#1d4ed8",
        "process": "#15803d",
        "technology": "#a16207",
        "outcome": "#991b1b",
    }
    groups = [
        ("People", "people", (-0.5, -0.5, 3.1, 2.65)),
        ("Process", "process", (1.65, -0.5, 3.15, 1.0)),
        ("Technology", "technology", (1.45, 3.35, 3.45, 1.15)),
    ]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    for title, key, (x, y, w, h) in groups:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.08,rounding_size=0.15",
                facecolor=fill[key],
                edgecolor=stroke[key],
                alpha=0.22,
                linewidth=1.0,
                zorder=0,
            )
        )
        ax.text(
            x + w / 2,
            y + h + 0.12,
            title,
            fontsize=9,
            fontweight="bold",
            color=stroke[key],
            ha="center",
            va="bottom",
            clip_on=False,
            zorder=1,
        )

    node_colors = [fill[category[n]] for n in g.nodes]
    node_edges = [stroke[category[n]] for n in g.nodes]
    nx.draw_networkx_nodes(
        g,
        pos,
        node_size=2200,
        node_color=node_colors,
        edgecolors=node_edges,
        linewidths=1.8,
        ax=ax,
    )
    edge_styles = {
        ("TechnicalControls", TARGET): "arc3,rad=-0.22",
        ("ConcerningBehaviour", "MonitoringAlert"): "arc3,rad=0.12",
    }
    default_style = "arc3,rad=0.0"
    for u, v in g.edges:
        nx.draw_networkx_edges(
            g,
            pos,
            edgelist=[(u, v)],
            ax=ax,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=18,
            width=1.8,
            edge_color="#374151",
            connectionstyle=edge_styles.get((u, v), default_style),
            min_source_margin=18,
            min_target_margin=18,
        )

    labels = {n: _dag_display_name(n) for n in g.nodes}
    label_pos = {n: (xy[0], xy[1] - 0.42) for n, xy in pos.items()}
    nx.draw_networkx_labels(
        g,
        label_pos,
        labels=labels,
        font_size=8,
        font_weight="bold",
        verticalalignment="top",
        horizontalalignment="center",
        ax=ax,
    )

    ax.set_title(
        "Causal DAG — insider-threat risk (People–Process–Technology)",
        fontsize=12,
        pad=12,
    )
    ax.set_xlim(-0.9, 7.4)
    ax.set_ylim(-0.9, 4.85)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    save_fig(fig, "fig01_dag")
    plt.close(fig)


def fig_scenarios(bn):
    baseline = p_incident_yes(bn)
    scenario_defs = [
        ("Baseline", {}),
        ("High risk", {
            "ConcerningBehaviour": "yes",
            "PrivilegedAccess": "excessive",
            "TechnicalControls": "weak",
        }),
        ("Low risk", {
            "ConcerningBehaviour": "no",
            "PrivilegedAccess": "appropriate",
            "TechnicalControls": "adequate",
        }),
        ("Alert, low incident", {
            "MonitoringAlert": "yes",
            "ConcerningBehaviour": "no",
            "TechnicalControls": "adequate",
        }),
        ("High dissatisfaction", {"JobDissatisfaction": "high"}),
    ]
    rows = [(name, p_incident_yes(bn, ev) if ev else baseline) for name, ev in scenario_defs]
    df = pd.DataFrame(rows, columns=["Scenario", "P(incident=yes)"])

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=df, x="Scenario", y="P(incident=yes)", palette="rocket", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel(r"$P(\mathrm{InsiderThreatIncident}=\mathrm{yes})$")
    ax.set_title("Incident probability by scenario")
    for i, v in enumerate(df["P(incident=yes)"]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    plt.xticks(rotation=18, ha="right")
    fig.tight_layout()
    save_fig(fig, "fig02_scenarios")
    plt.close(fig)
    return df


def skeleton(edges):
    return {frozenset(e) for e in edges}


def percentile_ci(values, lo=CI_LO, hi=CI_HI):
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.percentile(arr, lo)), float(np.percentile(arr, hi))


def summarize_by_group(df, group_cols, metrics):
    rows = []
    for keys, grp in df.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for metric in metrics:
            mean, lo, hi = percentile_ci(grp[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
        row["n_runs"] = len(grp)
        rows.append(row)
    return pd.DataFrame(rows)


def structure_metrics_for_sample(bn, arcs, sample_n, seed=SEED):
    set_global_seed(seed)
    true_sk = skeleton(arcs)
    tf = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    tf.close()
    sample_n.to_csv(tf.name, index=False)
    learner_hc = gum.BNLearner(tf.name)
    learner_hc.useGreedyHillClimbing()
    learner_hc.useScoreBIC()
    set_global_seed(seed)
    learned_hc = learner_hc.learnBN()
    learner_miic = gum.BNLearner(tf.name)
    learner_miic.useMIIC()
    set_global_seed(seed)
    learned_miic = learner_miic.learnBN()
    os.unlink(tf.name)
    rows = []
    for name, model in [("HillClimb", learned_hc), ("MIIC", learned_miic)]:
        cmp = bnvsbn.GraphicalBNComparator(bn, model)
        sk_scores = cmp.skeletonScores()
        dir_scores = cmp.scores()
        ham = cmp.hamming()
        e = [(model.variable(i).name(), model.variable(j).name()) for i, j in model.arcs()]
        sk = skeleton(e)
        tp = len(true_sk & sk)
        fp = len(sk - true_sk)
        fn = len(true_sk - sk)
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        rows.append({
            "Algorithm": name,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "SkeletonF1": sk_scores["fscore"],
            "DirectedF1": dir_scores["fscore"],
            "Hamming": ham["hamming"],
            "StructuralHamming": ham["structural hamming"],
        })
    return rows


def run_structure_learning(bn, arcs, data, seed=SEED):
    rows = []
    for n in [100, 500, 1000]:
        sample_n = data.sample(n=n, random_state=seed)
        for metrics in structure_metrics_for_sample(bn, arcs, sample_n, seed=seed):
            rows.append({"n": n, "seed": seed, **metrics})
    return pd.DataFrame(rows)


def run_structure_stability(bn, arcs, data, seeds=STABILITY_SEEDS):
    rows = []
    for i, seed in enumerate(seeds, start=1):
        print(f"  structure stability run {i}/{len(seeds)} (seed={seed})")
        for n in [100, 500, 1000]:
            sample_n = data.sample(n=n, random_state=seed)
            for metrics in structure_metrics_for_sample(bn, arcs, sample_n, seed=seed):
                rows.append({"n": n, "seed": seed, **metrics})
    return pd.DataFrame(rows)


def fig_structure_recovery(structure_summary):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    palette = {"HillClimb": "#2563eb", "MIIC": "#dc2626"}
    for ax, metric, title in zip(
        axes,
        ["SkeletonF1", "DirectedF1"],
        ["Skeleton F1", "Directed F1"],
    ):
        mean_col = f"{metric}_mean"
        lo_col = f"{metric}_ci_low"
        hi_col = f"{metric}_ci_high"
        for algo, grp in structure_summary.groupby("Algorithm"):
            grp = grp.sort_values("n")
            color = palette.get(algo, None)
            ax.plot(
                grp["n"],
                grp[mean_col],
                marker="o",
                label=algo,
                color=color,
                linewidth=2,
            )
            ax.fill_between(
                grp["n"],
                grp[lo_col],
                grp[hi_col],
                alpha=0.18,
                color=color,
            )
        ax.set_ylim(0, 1)
        ax.set_xscale("log")
        ax.set_xticks([100, 500, 1000])
        ax.set_xticklabels(["100", "500", "1000"])
        ax.set_title(title)
        ax.set_xlabel("Training sample size $n$")
        ax.legend(title="Algorithm")
    fig.suptitle(
        "Structure recovery vs. sample size "
        f"(mean and 95% CI over {N_STABILITY_RUNS} resamples)",
        y=1.04,
    )
    fig.tight_layout()
    save_fig(fig, "fig03_structure_recovery")
    plt.close(fig)


def infer_probs(model, df):
    probs = []
    for _, row in df.iterrows():
        ev = {c: row[c] for c in df.columns if c != TARGET}
        ie = gum.LazyPropagation(model)
        ie.setEvidence(ev)
        ie.makeInference()
        probs.append(float(ie.posterior(TARGET)[{TARGET: "yes"}]))
    return np.array(probs)


def classification_scores(bn, train_cls, test_cls, seed=SEED):
    set_global_seed(seed)
    train_csv = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    train_csv.close()
    train_cls.to_csv(train_csv.name, index=False)
    learner_true = gum.BNLearner(train_csv.name, bn)
    learner_true.useSmoothingPrior()
    set_global_seed(seed)
    bn_true_fitted = learner_true.learnParameters(bn.dag())
    learner_hc = gum.BNLearner(train_csv.name)
    learner_hc.useGreedyHillClimbing()
    learner_hc.useScoreBIC()
    set_global_seed(seed)
    bn_hc = learner_hc.learnBN()
    clf_naive = skbn.BNClassifier(learningMethod="NaiveBayes", scoringType="BIC")
    clf_naive.fit(data=train_cls, targetName=TARGET)
    y_true = (test_cls[TARGET] == "yes").astype(int).to_numpy()
    probs = {
        "Original BN + noisy-OR CPT": infer_probs(bn_true_fitted, test_cls),
        "Learned BN HC": infer_probs(bn_hc, test_cls),
        "Naive Bayes": clf_naive.predict_proba(test_cls.drop(columns=[TARGET]))[:, 1],
    }
    os.unlink(train_csv.name)
    rows = []
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        rows.append({
            "Model": name,
            "AUC": auc(fpr, tpr),
            "Accuracy": accuracy_score(y_true, (p >= 0.5).astype(int)),
            "fpr": fpr,
            "tpr": tpr,
        })
    return rows


def run_classification_stability(bn, data, seeds=STABILITY_SEEDS):
    data_path = SYNTHETIC_CASES_CSV
    if not data_path.is_file():
        DATA_DIR = data_path.parent
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        data.to_csv(data_path, index=False)
    CLASSIFICATION_STABILITY_CACHE.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, seed in enumerate(seeds, start=1):
        cache_path = CLASSIFICATION_STABILITY_CACHE / f"seed_{seed:03d}.json"
        if cache_path.is_file():
            print(f"  classification stability run {i}/{len(seeds)} (seed={seed}, cached)")
            rows.extend(json.loads(cache_path.read_text(encoding="utf-8")))
            continue
        print(f"  classification stability run {i}/{len(seeds)} (seed={seed})")
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(seed)
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")
        out = subprocess.check_output(
            [sys.executable, str(WORKER), str(data_path), str(seed)],
            env=env,
            text=True,
        )
        cache_path.write_text(out, encoding="utf-8")
        rows.extend(json.loads(out))
    return pd.DataFrame(rows)


def fig_classification_stability(classification_runs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    order = [
        "Original BN + noisy-OR CPT",
        "Learned BN HC",
        "Naive Bayes",
    ]
    for ax, metric, title in zip(
        axes,
        ["AUC", "Accuracy"],
        ["AUC", "Accuracy (threshold 0.5)"],
    ):
        sns.boxplot(
            data=classification_runs,
            x="Model",
            y=metric,
            order=order,
            hue="Model",
            palette="Set2",
            legend=False,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(
            ["Original BN", "Learned BN", "Naive Bayes"],
            rotation=12,
            ha="right",
        )
    fig.suptitle(
        "Classification stability "
        f"({N_STABILITY_RUNS} stratified Train/Test = 100/100 splits)",
        y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, "fig06_classification_stability")
    plt.close(fig)


def run_classification(bn, train_cls, test_cls):
    return classification_scores(bn, train_cls, test_cls)


def fig_roc(results):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for r in results:
        ax.plot(r["fpr"], r["tpr"], label=f"{r['Model']} (AUC={r['AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (Train/Test = 100/100)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save_fig(fig, "fig04_roc")
    plt.close(fig)


def fig_noisy_or(bn):
    labels = [
        ("0 active", {"ConcerningBehaviour": "no", "PrivilegedAccess": "appropriate", "TechnicalControls": "adequate"}),
        ("1 (behaviour)", {"ConcerningBehaviour": "yes", "PrivilegedAccess": "appropriate", "TechnicalControls": "adequate"}),
        ("2 active", {"ConcerningBehaviour": "yes", "PrivilegedAccess": "excessive", "TechnicalControls": "adequate"}),
        ("3 active", {"ConcerningBehaviour": "yes", "PrivilegedAccess": "excessive", "TechnicalControls": "weak"}),
    ]
    or_df = pd.DataFrame([
        {"Scenario": lab, "P(incident=yes)": p_incident_yes(bn, ev)} for lab, ev in labels
    ])
    specs = [
        ("ConcerningBehaviour", "yes", "no"),
        ("PrivilegedAccess", "excessive", "appropriate"),
        ("TechnicalControls", "weak", "adequate"),
    ]
    curve = []
    for k in range(4):
        probs = []
        for active_idx in itertools.combinations(range(3), k):
            ev = {specs[i][0]: specs[i][1] if i in active_idx else specs[i][2] for i in range(3)}
            probs.append(p_incident_yes(bn, ev))
        curve.append({"k": k, "p": float(np.mean(probs))})
    curve_df = pd.DataFrame(curve)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sns.barplot(data=or_df, x="Scenario", y="P(incident=yes)", palette="Oranges_r", ax=axes[0])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Noisy-OR scenarios")
    axes[0].tick_params(axis="x", rotation=12)
    axes[1].plot(curve_df["k"], curve_df["p"], marker="o", color="#c2410c", linewidth=2)
    axes[1].set_xticks(range(4))
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Number of active risk causes")
    axes[1].set_ylabel(r"Mean $P(\mathrm{incident}=\mathrm{yes})$")
    axes[1].set_title("Response curve")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Noisy-OR parametrisation", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig05_noisy_or")
    plt.close(fig)


def main():
    set_global_seed(SEED)
    sns.set_theme(style="whitegrid")
    print("Building model...")
    bn, arcs = build_insider_bn()
    print("Figure 1: DAG")
    fig_dag(bn, arcs)
    print("Figure 2: Scenarios")
    fig_scenarios(bn)
    print(f"Loading synthetic data ({SYNTHETIC_CASES_CSV.name})...")
    data = load_synthetic_cases(bn, n=N_SYNTHETIC_CASES, seed=SEED)
    cls_pool = data.sample(n=300, random_state=SEED)
    train_cls, test_cls = train_test_split(
        cls_pool,
        train_size=100,
        test_size=100,
        random_state=SEED,
        stratify=cls_pool[TARGET],
    )
    print("Figure 3: Structure recovery (reference run + stability)")
    recovery_df = run_structure_learning(bn, arcs, data, seed=SEED)
    recovery_df.to_csv(FIG_DIR / "table_structure_recovery.csv", index=False)
    structure_runs = run_structure_stability(bn, arcs, data)
    structure_runs.to_csv(FIG_DIR / "table_structure_stability_runs.csv", index=False)
    structure_summary = summarize_by_group(
        structure_runs,
        ["n", "Algorithm"],
        ["SkeletonF1", "DirectedF1", "Hamming"],
    )
    structure_summary.to_csv(FIG_DIR / "table_structure_stability.csv", index=False)
    fig_structure_recovery(structure_summary)
    print("Figure 4: ROC (reference split seed=42)")
    results = run_classification(bn, train_cls, test_cls)
    pd.DataFrame([{k: r[k] for k in ("Model", "AUC", "Accuracy")} for r in results]).to_csv(
        FIG_DIR / "table_classification.csv", index=False
    )
    fig_roc(results)
    print("Figure 6: Classification stability")
    classification_runs = run_classification_stability(bn, data)
    classification_runs.to_csv(FIG_DIR / "table_classification_stability_runs.csv", index=False)
    classification_summary = summarize_by_group(
        classification_runs,
        ["Model"],
        ["AUC", "Accuracy"],
    )
    classification_summary.to_csv(FIG_DIR / "table_classification_stability.csv", index=False)
    fig_classification_stability(classification_runs)
    print("Figure 5: Noisy-OR")
    fig_noisy_or(bn)
    print("Done.")


if __name__ == "__main__":
    main()
