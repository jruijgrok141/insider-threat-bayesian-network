"""Generate insider_threat_bn_prototype.ipynb from scratch."""
import json
from pathlib import Path

def cell_md(text: str):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def cell_code(text: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = [
    cell_md(
        """# Prototype: Insider-threat cyber security risk with pyAgrum

Full workflow for the Bayesian Reasoning and Learning assignment + research proposal.

### Link to the assignment

| Component | Assignment | This notebook |
|-----------|------------|---------------|
| **Task 1** | Manual causal BN (5–10 nodes), inference, validation | Sections 1–4 |
| **Task 2.2a** | Structure learning (search-and-score + constraint-based) | Section 5 |
| **Task 2.2b** | Classification original / learned / naive Bayes | Section 7 |
| **RQ4** | Noisy-OR on incident node | Section 6 |

Report outline: `report/Report_Insider_Threat_BN.md`

### Structure (step by step)

| Step | Section | Content |
|------|---------|---------|
| 0 | Imports | Libraries and `TARGET` / `SEED` |
| 1 | Variables | 8 nodes + People–Process–Technology domain |
| 2 | Model | DAG, CPTs, noisy-OR on `InsiderThreatIncident` |
| 3 | Task 1 | Scenario inference + validation (marginals, conditionals, sensitivity) |
| 4 | Task 2.2a | Synthetic data, structure recovery, visualisation |
| 5 | RQ4 | Noisy-OR demo, parameter comparison, monotonicity |
| 6 | Task 2.2b | Classification + ROC/AUC |

**Tip:** Run cells **in order**; later work uses `bn`, `data`, `train_cls` / `test_cls`."""
    ),
    cell_md(
        """## Variables (8 nodes)

| Variable | States | Domain | Description |
|----------|--------|--------|-------------|
| `JobDissatisfaction` | low, medium, high | People | Degree of employee dissatisfaction or frustration with their work or work environment. |
| `SecurityAwareness` | low, medium, high | People | Employee knowledge and awareness of security risks and safe behaviour. |
| `PolicyCompliance` | good, poor | Process | Extent to which employees and processes comply with security policy. |
| `ConcerningBehaviour` | no, yes | People | Whether behaviour has been observed that suggests possible malicious intent or misuse (e.g. anomalous data use). |
| `PrivilegedAccess` | appropriate, excessive | Technology / Process | Whether access rights are appropriate for the role, or excessively broad. |
| `TechnicalControls` | weak, adequate | Technology | Strength of preventive and detective technical controls (firewalls, logging, DLP, etc.). |
| `MonitoringAlert` | no, yes | Technology | Whether the monitoring system generated an alert based on behaviour or technical signals. |
| `InsiderThreatIncident` | no, yes | Outcome | Whether an insider-related security incident actually occurred (e.g. data breach or access misuse). |

The outcome variable `InsiderThreatIncident` is the **class label** for Task 2.2b."""
    ),
    cell_md(
        """### Step 0 — Imports and settings

| Element | Meaning |
|---------|---------|
| `TARGET` | `InsiderThreatIncident` — same outcome name throughout |
| `SEED = 42` | Reproducible samples and train/test split |
| `set_global_seed` / `draw_synthetic_cases` | Fixed pyAgrum + NumPy RNG and topological sampling order |
| `gum` / `gnb` | Build BN, inference, notebook visualisations |
| `bnvsbn` | Compare ground-truth vs. learned network |
| `skbn` | Naive Bayes classifier (Task 2.2b) |"""
    ),
    cell_code(
        """import warnings
warnings.filterwarnings('ignore')

import os
import tempfile
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score

import pyagrum as gum
import pyagrum.skbn as skbn
import pyagrum.lib.notebook as gnb
import pyagrum.lib.bn_vs_bn as bnvsbn
import pyagrum.skbn as skbn
import seaborn as sns
from sklearn.metrics import roc_curve, auc, accuracy_score

TARGET = 'InsiderThreatIncident'
SEED = 42
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)"""
    ),
    cell_code(
        """import random

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

def load_synthetic_cases(bn, n=2000, seed=SEED):
    from pathlib import Path
    for csv_path in (
        Path('report/data/synthetic_cases_n2000_seed42.csv'),
        Path('../report/data/synthetic_cases_n2000_seed42.csv'),
    ):
        if csv_path.is_file():
            return pd.read_csv(csv_path)
    return draw_synthetic_cases(bn, n=n, seed=seed)

set_global_seed(SEED)"""
    ),
    cell_md(
        """### Step 2 — Build model (Task 1)

The next cell defines the full BN in one block:

1. **`states`** and **`variable_descriptions`** — variables + summary table
2. **`noisy_or_p_yes` / `fill_incident_noisy_or`** — RQ4 parametrisation of incident node
3. **`build_insider_bn`** — DAG + all CPTs (expert judgement)
4. **`bn`** — main model with `use_noisy_or_incident=True`

**Causal structure (8 arcs):** People/Process factors on the left (`JobDissatisfaction`, `SecurityAwareness`) → behaviour/compliance/access; Technology (`TechnicalControls`) → incident and `MonitoringAlert`; three parents of `InsiderThreatIncident` via noisy-OR."""
    ),
    cell_code(
        """states = {
    'JobDissatisfaction': ['low', 'medium', 'high'],
    'SecurityAwareness': ['low', 'medium', 'high'],
    'PolicyCompliance': ['good', 'poor'],
    'ConcerningBehaviour': ['no', 'yes'],
    'PrivilegedAccess': ['appropriate', 'excessive'],
    'TechnicalControls': ['weak', 'adequate'],
    'MonitoringAlert': ['no', 'yes'],
    TARGET: ['no', 'yes'],
}

variable_descriptions = {
    'JobDissatisfaction': 'Degree of employee dissatisfaction or frustration with their work or work environment.',
    'SecurityAwareness': 'Employee knowledge and awareness of security risks and safe behaviour.',
    'PolicyCompliance': 'Extent to which employees and processes comply with security policy.',
    'ConcerningBehaviour': 'Whether behaviour has been observed that suggests possible malicious intent or misuse.',
    'PrivilegedAccess': 'Whether access rights are appropriate for the role, or excessively broad.',
    'TechnicalControls': 'Strength of preventive and detective technical controls (firewalls, logging, DLP).',
    'MonitoringAlert': 'Whether the monitoring system generated an alert based on signals.',
    TARGET: 'Whether an insider-related security incident actually occurred.',
}

pd.DataFrame([
    {
        'Variable': name,
        'States': ', '.join(labels),
        'Description': variable_descriptions[name],
    }
    for name, labels in states.items()
])

def add_var(bn, name, labels):
    v = gum.LabelizedVariable(name, name, 0)
    for lab in labels:
        v.addLabel(lab)
    bn.add(v)

def noisy_or_p_yes(active, leak=0.94, inhibitions=(0.65, 0.60, 0.70)):
    \"\"\"P(Y=yes) for binary noisy-OR; inhibitions apply when cause is active.\"\"\"
    p_no = leak
    for is_active, inh in zip(active, inhibitions):
        if is_active:
            p_no *= inh
    return 1.0 - p_no

def fill_incident_noisy_or(bn, leak=0.94, inhibitions=(0.65, 0.60, 0.70)):
    for cb in states['ConcerningBehaviour']:
        for pa in states['PrivilegedAccess']:
            for tc in states['TechnicalControls']:
                active = (
                    cb == 'yes',
                    pa == 'excessive',
                    tc == 'weak',
                )
                py = noisy_or_p_yes(active, leak=leak, inhibitions=inhibitions)
                bn.cpt(TARGET)[{
                    'ConcerningBehaviour': cb,
                    'PrivilegedAccess': pa,
                    'TechnicalControls': tc,
                }] = [1 - py, py]

def build_insider_bn(use_noisy_or_incident=True):
    bn = gum.BayesNet('InsiderThreatRisk')
    for name, labels in states.items():
        add_var(bn, name, labels)

    arcs = [
        ('JobDissatisfaction', 'ConcerningBehaviour'),
        ('SecurityAwareness', 'PolicyCompliance'),
        ('PolicyCompliance', 'PrivilegedAccess'),
        ('ConcerningBehaviour', TARGET),
        ('PrivilegedAccess', TARGET),
        ('TechnicalControls', TARGET),
        ('TechnicalControls', 'MonitoringAlert'),
        ('ConcerningBehaviour', 'MonitoringAlert'),
    ]
    for a, b in arcs:
        bn.addArc(a, b)

    bn.cpt('JobDissatisfaction').fillWith([0.35, 0.45, 0.20])
    bn.cpt('SecurityAwareness').fillWith([0.25, 0.50, 0.25])
    bn.cpt('TechnicalControls').fillWith([0.28, 0.72])

    bn.cpt('ConcerningBehaviour')[{'JobDissatisfaction': 'low'}] = [0.88, 0.12]
    bn.cpt('ConcerningBehaviour')[{'JobDissatisfaction': 'medium'}] = [0.70, 0.30]
    bn.cpt('ConcerningBehaviour')[{'JobDissatisfaction': 'high'}] = [0.45, 0.55]

    bn.cpt('PolicyCompliance')[{'SecurityAwareness': 'low'}] = [0.25, 0.75]
    bn.cpt('PolicyCompliance')[{'SecurityAwareness': 'medium'}] = [0.55, 0.45]
    bn.cpt('PolicyCompliance')[{'SecurityAwareness': 'high'}] = [0.82, 0.18]

    bn.cpt('PrivilegedAccess')[{'PolicyCompliance': 'good'}] = [0.90, 0.10]
    bn.cpt('PrivilegedAccess')[{'PolicyCompliance': 'poor'}] = [0.35, 0.65]

    for tc in states['TechnicalControls']:
        for cb in states['ConcerningBehaviour']:
            p_alert = 0.08 if tc == 'adequate' else 0.35
            if cb == 'yes':
                p_alert = min(0.95, p_alert + 0.40)
            bn.cpt('MonitoringAlert')[{'TechnicalControls': tc, 'ConcerningBehaviour': cb}] = [1 - p_alert, p_alert]

    if use_noisy_or_incident:
        fill_incident_noisy_or(bn)
    else:
        for cb in states['ConcerningBehaviour']:
            for pa in states['PrivilegedAccess']:
                for tc in states['TechnicalControls']:
                    z = -2.2
                    z += 1.6 if cb == 'yes' else 0.0
                    z += 1.3 if pa == 'excessive' else 0.0
                    z += 1.1 if tc == 'weak' else 0.0
                    py = 1 / (1 + np.exp(-z))
                    bn.cpt(TARGET)[{
                        'ConcerningBehaviour': cb,
                        'PrivilegedAccess': pa,
                        'TechnicalControls': tc,
                    }] = [1 - py, py]
    return bn

bn = build_insider_bn(use_noisy_or_incident=True)
arcs = [(bn.variable(i).name(), bn.variable(j).name()) for i, j in bn.arcs()]
bn"""
    ),
    cell_md(
        """### Step 2b — Visualise DAG

**Figure: causal DAG insider threat (People–Process–Technology)**

| Element | Meaning |
|---------|---------|
| **Node** | Discrete variable; arrows point from parent to child |
| **Left side** | People/Process risk factors (dissatisfaction, awareness, compliance) |
| **Bottom right** | `InsiderThreatIncident` — binary outcome/event |
| **Top right** | `MonitoringAlert` — detection signal, **not** a parent of incident |

**Reading guide:** Follow paths such as `SecurityAwareness → PolicyCompliance → PrivilegedAccess → InsiderThreatIncident`. `MonitoringAlert` shares parents with the incident but does not directly influence the incident in this model — relevant for the “alert without incident path” scenario.

`gnb.showBN` is the preferred view; on failure NetworkX draws the same structure with fixed positions."""
    ),
    cell_code(
        """# DAG visualisation
try:
    gnb.showBN(bn, size='14')
except Exception:
    g = nx.DiGraph()
    g.add_nodes_from(states.keys())
    g.add_edges_from(arcs)
    pos = {
        'JobDissatisfaction': (-1.5, 1.0),
        'SecurityAwareness': (-1.5, -0.2),
        'ConcerningBehaviour': (-0.4, 1.0),
        'PolicyCompliance': (-0.4, -0.2),
        'PrivilegedAccess': (0.5, -0.2),
        'TechnicalControls': (0.5, 1.0),
        'MonitoringAlert': (1.4, 1.0),
        TARGET: (1.4, -0.2),
    }
    plt.figure(figsize=(12, 6))
    nx.draw_networkx_nodes(g, pos, node_size=3200, node_color='#dbeafe', edgecolors='#1e3a8a', linewidths=1.4)
    nx.draw_networkx_labels(g, pos, font_size=8)
    nx.draw_networkx_edges(g, pos, arrows=True, arrowstyle='-|>', arrowsize=18, width=1.8, edge_color='#374151')
    plt.title('DAG: Insider-threat risk (People–Process–Technology)', fontsize=13)
    plt.axis('off')
    plt.tight_layout()
    plt.show()"""
    ),
    cell_md(
        """## Task 1 — Inference and scenarios

### Step 3a — Baseline and scenario analysis

`LazyPropagation` computes exact posterior probabilities. We measure $P(\\text{InsiderThreatIncident}=\\text{yes})$:

- **Baseline** — without evidence (population risk)
- **Four scenarios** — high/low risk, alert without incident path, high dissatisfaction (indirect via behaviour)

This answers Assignment Task 1: *“what is the incident risk under recognisable situations?”*"""
    ),
    cell_code(
        """def p_incident_yes(model, evidence=None):
    ie = gum.LazyPropagation(model)
    if evidence:
        ie.setEvidence(evidence)
    ie.makeInference()
    return float(ie.posterior(TARGET)[{TARGET: 'yes'}])

ie = gum.LazyPropagation(bn)
ie.makeInference()
baseline = p_incident_yes(bn)
print(f'Baseline P({TARGET}=yes): {baseline:.3f}')

scenarios = {
    'High risk (behaviour + access + controls)': {
        'ConcerningBehaviour': 'yes',
        'PrivilegedAccess': 'excessive',
        'TechnicalControls': 'weak',
    },
    'Low risk': {
        'ConcerningBehaviour': 'no',
        'PrivilegedAccess': 'appropriate',
        'TechnicalControls': 'adequate',
        'PolicyCompliance': 'good',
    },
    'Alert without incident path': {
        'MonitoringAlert': 'yes',
        'ConcerningBehaviour': 'no',
        'TechnicalControls': 'adequate',
    },
    'Dissatisfaction → behaviour': {
        'JobDissatisfaction': 'high',
    },
}

rows = [('Baseline', baseline)]
for name, ev in scenarios.items():
    rows.append((name, p_incident_yes(bn, ev)))

risk_df = pd.DataFrame(rows, columns=['Scenario', f'P({TARGET}=yes)'])
risk_df"""
    ),
    cell_md(
        """**Figure: bar chart incident probability per scenario**

| Aspect | Explanation |
|--------|-------------|
| **Y-axis (0–1)** | Posterior $P(\\text{InsiderThreatIncident}=\\text{yes})$ |
| **Baseline** | Reference without observations |
| **High risk** | All three noisy-OR activators active → expected highest bar |
| **Low risk** | Protective configuration → lowest bar |
| **Alert without incident path** | Alert observed, but behaviour/controls OK → typically between baseline and high risk |
| **Dissatisfaction → behaviour** | Only `JobDissatisfaction=high` → indirect effect, weaker than high-risk profile |

Labels above bars show two decimals; use this figure in presentation and report (scenario figure)."""
    ),
    cell_code(
        """ax = sns.barplot(data=risk_df, x='Scenario', y=f'P({TARGET}=yes)', palette='magma')
ax.set_ylim(0, 1)
ax.set_title('Insider-threat incident probability per scenario')
for i, v in enumerate(risk_df[f'P({TARGET}=yes)']):
    ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.show()"""
    ),
    cell_md(
        """## Task 1 (continued): marginals, conditionals and sensitivity

Assignment: validate whether marginals and conditionals are plausible; test robustness under CPT variation.

### Step 3b — Marginal posterior

Long-format table: each row = `(Variable, State, P)`. No pivot → no confusing `NaN`. Root marginals match CPTs; other nodes are mixed over their parents."""
    ),
    cell_code(
        """# Marginals (baseline, without evidence)
ie0 = gum.LazyPropagation(bn)
ie0.makeInference()
marginals = []
for name in states:
    post = ie0.posterior(name)
    for lab in states[name]:
        marginals.append({
            'Variable': name,
            'State': lab,
            'P': float(post[{name: lab}]),
        })
marginal_df = pd.DataFrame(marginals).sort_values(['Variable', 'State']).round(3).reset_index(drop=True)
marginal_df  # long format: each row is one (variable, state) pair; no NaN"""
    ),
    cell_md(
        """### Step 3c — Conditional queries

Four representative questions: three times incident risk given one risk factor, plus alert probability given concerning behaviour. This shows that the model **distinguishes** between factors (behaviour vs. access vs. controls)."""
    ),
    cell_code(
        """# Example conditionals (Task 1)
queries = [
    ('P(incident | concerning behaviour)', {'ConcerningBehaviour': 'yes'}),
    ('P(incident | excessive access)', {'PrivilegedAccess': 'excessive'}),
    ('P(incident | weak controls)', {'TechnicalControls': 'weak'}),
    ('P(alert | concerning behaviour)', {'ConcerningBehaviour': 'yes'}),
]
cond_rows = []
for label, ev in queries:
    target_var = TARGET if 'incident' in label else 'MonitoringAlert'
    ieq = gum.LazyPropagation(bn)
    ieq.setEvidence(ev)
    ieq.makeInference()
    py = float(ieq.posterior(target_var)[{target_var: 'yes'}])
    cond_rows.append({'Query': label, 'P(yes)': py})
pd.DataFrame(cond_rows)"""
    ),
    cell_md(
        """### Step 3d — Sensitivity noisy-OR `leak`

`leak` = $P(\\text{no incident} \\mid \\text{no active cause})$. Higher `leak` → lower baseline. Check whether the **ranking** high vs. low risk remains stable over `leak ∈ {0.90, 0.94, 0.97}` — that supports robustness in the report."""
    ),
    cell_code(
        """# Sensitivity: leak parameter noisy-OR on incident CPT
leak_values = [0.90, 0.94, 0.97]
sens_rows = []
for leak in leak_values:
    bn_s = build_insider_bn(use_noisy_or_incident=True)
    fill_incident_noisy_or(bn_s, leak=leak)
    sens_rows.append({
        'leak': leak,
        'baseline': p_incident_yes(bn_s),
        'high_risk': p_incident_yes(bn_s, {
            'ConcerningBehaviour': 'yes',
            'PrivilegedAccess': 'excessive',
            'TechnicalControls': 'weak',
        }),
    })
pd.DataFrame(sens_rows)"""
    ),
    cell_md(
        """## Task 2.2a: structure learning (search-and-score vs constraint-based)

Assignment: synthetic data from the hand-built network; compare $n \\in \\{100,500,1000\\}$ for Hill Climbing (BIC, search-and-score) and MIIC (constraint-based).

### Step 4a — Synthetic data

`BNDatabaseGenerator` draws 2000 cases from the manual BN. Then: pool of 300 → **stratified** split 100 train / 100 test on `InsiderThreatIncident` (Task 2.2b requirement). The full `data` (2000) is used for structure learning with varying $n$."""
    ),
    cell_code(
        """# Synthetic data (pool for structure learning + separate classification set)
data = load_synthetic_cases(bn, n=2000, seed=SEED)

# Classification per assignment: Train=100, Test=100 (disjoint, stratified)
cls_pool = data.sample(n=300, random_state=SEED)
train_cls, test_cls = train_test_split(
    cls_pool, train_size=100, test_size=100, random_state=SEED, stratify=cls_pool[TARGET]
)
print('Structure pool:', data.shape, '| Classification:', train_cls.shape, test_cls.shape)"""
    ),
    cell_md(
        """### Step 4b — Structure recovery and metrics

Per $(n, \\text{algorithm})$:

| Metric | Meaning |
|--------|---------|
| **SkeletonF1** | Undirected structure correct |
| **DirectedF1** | Arrow directions correct |
| **Hamming** | Number of differing directed arcs |
| **StructuralHamming** | Skeleton + orientation errors |
| **TP/FP/FN** | Skeleton true/false pos/neg |

Expectation: scores increase with $n$; Hill Climbing often performs well on data from the same generative model."""
    ),
    cell_code(
        """def skeleton(edges):
    return {frozenset(e) for e in edges}

true_skeleton = skeleton(arcs)
rows = []
for n in [100, 500, 1000]:
    sample_n = data.sample(n=n, random_state=SEED)
    set_global_seed(SEED)
    tf = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
    tf.close()
    sample_n.to_csv(tf.name, index=False)

    learner_hc = gum.BNLearner(tf.name)
    learner_hc.useGreedyHillClimbing()
    learner_hc.useScoreBIC()
    learned_hc = learner_hc.learnBN()

    learner_miic = gum.BNLearner(tf.name)
    learner_miic.useMIIC()
    learned_miic = learner_miic.learnBN()
    os.unlink(tf.name)

    for name, model in [('HillClimb', learned_hc), ('MIIC', learned_miic)]:
        cmp = bnvsbn.GraphicalBNComparator(bn, model)
        sk_scores = cmp.skeletonScores()
        dir_scores = cmp.scores()
        ham = cmp.hamming()
        e = [(model.variable(i).name(), model.variable(j).name()) for i, j in model.arcs()]
        sk = skeleton(e)
        tp = len(true_skeleton & sk)
        fp = len(sk - true_skeleton)
        fn = len(true_skeleton - sk)
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        rows.append((
            n, name, tp, fp, fn, precision, recall,
            sk_scores['fscore'], dir_scores['fscore'],
            ham['hamming'], ham['structural hamming'],
        ))

recovery_df = pd.DataFrame(rows, columns=[
    'n', 'Algorithm', 'TP', 'FP', 'FN', 'Precision', 'Recall',
    'SkeletonF1_pyagrum', 'DirectedF1_pyagrum', 'Hamming', 'StructuralHamming',
])
recovery_df"""
    ),
    cell_md(
        """**Figure: structure recovery vs. sample size (two panels)**

| Panel | Metric | Reading guide |
|-------|--------|---------------|
| **Left** | Skeleton F1 | 1.0 = perfectly recovered “who connects to whom” |
| **Right** | Directed F1 | Orientation is stricter; often lower than skeleton |
| **X-axis (log)** | $n = 100, 500, 1000$ | More data → usually rising lines |
| **Colour/line** | HillClimb vs. MIIC | Compare who benefits faster from more data |

The table below the figure shows Hamming distances — useful for the report (compact numbers alongside the trend)."""
    ),
    cell_code(
        """fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharex=True)
for ax, metric, title in zip(
    axes,
    ['SkeletonF1_pyagrum', 'DirectedF1_pyagrum'],
    ['Skeleton F1', 'Directed F1'],
):
    sns.lineplot(data=recovery_df, x='n', y=metric, hue='Algorithm', marker='o', ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_xscale('log')
plt.suptitle('Structure recovery vs. sample size (insider-threat BN)')
plt.tight_layout()
plt.show()

recovery_df[['n', 'Algorithm', 'Hamming', 'StructuralHamming']].sort_values(['n', 'Algorithm'])"""
    ),
    cell_md(
        """### Step 4c — Visual comparison of DAGs

**Figure: `gnb.sideBySide` — ground truth (left) vs. Hill Climbing on $n=1000$ (right)**

Look for: missing arcs (false negative), extra arcs (false positive), reversed arrows. Check whether `InsiderThreatIncident` still has the expected parents. On fallback the cell only prints the arc counts."""
    ),
    cell_code(
        """# Visual comparison: ground truth vs learned (n=1000, Hill Climbing)
sample_1k = data.sample(n=1000, random_state=SEED)
tf_vis = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
tf_vis.close()
sample_1k.to_csv(tf_vis.name, index=False)
learner_vis = gum.BNLearner(tf_vis.name)
learner_vis.useGreedyHillClimbing()
learner_vis.useScoreBIC()
bn_learned_vis = learner_vis.learnBN()
os.unlink(tf_vis.name)
try:
    gnb.sideBySide(bn, bn_learned_vis, size='10')
except Exception:
    print('Ground-truth arcs:', len(list(bn.arcs())))
    print('Learned arcs:', len(list(bn_learned_vis.arcs())))"""
    ),
    cell_md(
        """## RQ4: Noisy-OR parametrisation (required)

`InsiderThreatIncident` is already parametrised via noisy-OR in the **main model**:
three **independent risk causes** can each activate the incident:

| Cause (active when) | Variable |
|---------------------|----------|
| Concerning behaviour | `ConcerningBehaviour = yes` |
| Access misuse | `PrivilegedAccess = excessive` |
| Weak controls | `TechnicalControls = weak` |

Formula: $P(I=\\text{no} \\mid \\text{parents}) = \\text{leak} \\times \\prod_{i: \\text{cause}_i} \\text{inhibition}_i$, so $P(I=\\text{yes}) = 1 - P(I=\\text{no})$.

Below we compare this with a **logistic full CPT** (same structure, without noisy-OR).

### Step 5a — Parameter efficiency

Noisy-OR uses **4 parameters** (leak + 3 inhibitions) vs. 7 free parameters for a full 3-parent binary CPT ($2^3-1$ rows). The demo BN `NoisyOR_Insider` illustrates the principle with three binary parents."""
    ),
    cell_code(
        """# Compact noisy-OR submodel (3 binary risk activators → incident)
def add_binary_var(bn, name, yes_prob=0.3):
    v = gum.LabelizedVariable(name, name, 0)
    v.addLabel('no')
    v.addLabel('yes')
    bn.add(v)
    bn.cpt(name).fillWith([1 - yes_prob, yes_prob])

def build_noisy_or_demo(leak=0.02, inhibitions=(0.50, 0.45, 0.55)):
    bn_or = gum.BayesNet('NoisyOR_Insider')
    for n in ['RiskBehaviour', 'ExcessiveAccess', 'WeakControls', TARGET]:
        add_binary_var(bn_or, n, yes_prob=0.30)
    for parent in ['RiskBehaviour', 'ExcessiveAccess', 'WeakControls']:
        bn_or.addArc(parent, TARGET)
    causes = ['RiskBehaviour', 'ExcessiveAccess', 'WeakControls']
    for rb, ea, wc in itertools.product(['no', 'yes'], repeat=3):
        active = (rb == 'yes', ea == 'yes', wc == 'yes')
        py = noisy_or_p_yes(active, leak=leak, inhibitions=inhibitions)
        bn_or.cpt(TARGET)[{
            'RiskBehaviour': rb,
            'ExcessiveAccess': ea,
            'WeakControls': wc,
        }] = [1 - py, py]
    return bn_or

bn_or_demo = build_noisy_or_demo()
bn_tabular = build_insider_bn(use_noisy_or_incident=False)

# Number of free parameters in incident node CPT
def free_params(model, child):
    cpt = model.cpt(child)
    n_parents = cpt.nbrDim() - 1
    prod = 1
    for i in range(n_parents):
        prod *= cpt.variable(i).domainSize()
    card_child = model.variable(child).domainSize()
    return (card_child - 1) * prod

param_cmp = pd.DataFrame([
    {'Model': 'Full CPT (logistic)', 'FreeParams_Incident': free_params(bn_tabular, TARGET)},
    {'Model': 'Noisy-OR (3 inhibitions + leak)', 'FreeParams_Incident': 4},
    {'Model': 'Noisy-OR demo (3 bin parents)', 'FreeParams_Incident': free_params(bn_or_demo, TARGET)},
])
param_cmp"""
    ),
    cell_md(
        """### Step 5b — Monotonicity (more active causes → higher risk)

The table compares the **demo submodel** (0–3 active causes) with the **main model** (none / behaviour / access / all). Noisy-OR expectation: $P(\\text{incident})$ increases monotonically with the number of active activators."""
    ),
    cell_code(
        """# Inference: noisy-OR monotonicity (more active causes → higher risk)
or_scenarios = [
    ('No active cause', {'RiskBehaviour': 'no', 'ExcessiveAccess': 'no', 'WeakControls': 'no'}),
    ('One cause', {'RiskBehaviour': 'yes', 'ExcessiveAccess': 'no', 'WeakControls': 'no'}),
    ('Two causes', {'RiskBehaviour': 'yes', 'ExcessiveAccess': 'yes', 'WeakControls': 'no'}),
    ('Three causes', {'RiskBehaviour': 'yes', 'ExcessiveAccess': 'yes', 'WeakControls': 'yes'}),
]
or_rows = []
for label, ev in or_scenarios:
    or_rows.append({'Scenario': label, f'P({TARGET}=yes)': p_incident_yes(bn_or_demo, ev)})

# Same logic on main model (noisy-OR CPT)
main_or_rows = [
    ('None', {'ConcerningBehaviour': 'no', 'PrivilegedAccess': 'appropriate', 'TechnicalControls': 'adequate'}),
    ('Behaviour', {'ConcerningBehaviour': 'yes', 'PrivilegedAccess': 'appropriate', 'TechnicalControls': 'adequate'}),
    ('Access', {'ConcerningBehaviour': 'no', 'PrivilegedAccess': 'excessive', 'TechnicalControls': 'adequate'}),
    ('All', {'ConcerningBehaviour': 'yes', 'PrivilegedAccess': 'excessive', 'TechnicalControls': 'weak'}),
]
for label, ev in main_or_rows:
    or_rows.append({'Scenario': f'Main: {label}', f'P({TARGET}=yes)': p_incident_yes(bn, ev)})

or_df = pd.DataFrame(or_rows)
or_df"""
    ),
    cell_md(
        """**Figure: noisy-OR — two panels + optional demo DAG**

| Panel | Content |
|-------|---------|
| **Left** | Demo BN: 0 → 1 → 2 → 3 active causes; bars should **rise monotonically** |
| **Right** | Main model with real variable names (`Main: None` … `Main: All`) |
| **Optional `showBN(bn_or_demo)`** | Minimal 3→1 structure for presentation explanation |

Red/orange palettes emphasise rising risk. Use this for RQ4 in the report: *compact parametrisation + monotone response*."""
    ),
    cell_code(
        """try:
    gnb.showBN(bn_or_demo, size='8')
except Exception:
    pass

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.barplot(data=or_df.iloc[:4], x='Scenario', y=f'P({TARGET}=yes)', ax=axes[0], palette='Reds_r')
axes[0].set_ylim(0, 1)
axes[0].set_title('Noisy-OR demo: monotonic risk accumulation')
axes[0].tick_params(axis='x', rotation=12)

main_plot = or_df[or_df['Scenario'].str.startswith('Main')]
sns.barplot(data=main_plot, x='Scenario', y=f'P({TARGET}=yes)', ax=axes[1], palette='Oranges_r')
axes[1].set_ylim(0, 1)
axes[1].set_title('Main model (incident CPT = noisy-OR)')
axes[1].tick_params(axis='x', rotation=15)
plt.tight_layout()
plt.show()"""
    ),
    cell_md(
        """**Figure: response curve — mean incident probability vs. number of active causes**

For each $k \\in \\{0,1,2,3\\}$ we average $P(\\text{incident})$ over **all combinations** with exactly $k$ active activators (symmetric noisy-OR). Two lines:

- **Noisy-OR demo** — three binary parents
- **Main model** — real states (`yes`/`excessive`/`weak`)

Both curves should **rise** in $k$; deviations indicate CPT or parametrisation errors."""
    ),
    cell_code(
        """# Response curve: number of active noisy-OR causes vs. P(incident)
def mean_p_by_active_binary(model, parent_names):
    rows = []
    for k in range(len(parent_names) + 1):
        probs = []
        for active in itertools.combinations(parent_names, k):
            ev = {p: ('yes' if p in active else 'no') for p in parent_names}
            probs.append(p_incident_yes(model, ev))
        rows.append({'active': k, 'p': float(np.mean(probs))})
    return pd.DataFrame(rows)

def mean_p_main_by_active():
    specs = [
        ('ConcerningBehaviour', 'yes', 'no'),
        ('PrivilegedAccess', 'excessive', 'appropriate'),
        ('TechnicalControls', 'weak', 'adequate'),
    ]
    rows = []
    for k in range(len(specs) + 1):
        probs = []
        for active_idx in itertools.combinations(range(len(specs)), k):
            ev = {specs[i][0]: specs[i][1] if i in active_idx else specs[i][2] for i in range(len(specs))}
            probs.append(p_incident_yes(bn, ev))
        rows.append({'active': k, 'p': float(np.mean(probs))})
    return pd.DataFrame(rows)

or_curve = mean_p_by_active_binary(bn_or_demo, ['RiskBehaviour', 'ExcessiveAccess', 'WeakControls'])
main_curve = mean_p_main_by_active()

plt.figure(figsize=(8, 4.5))
plt.plot(or_curve['active'], or_curve['p'], marker='o', label='Noisy-OR demo')
plt.plot(main_curve['active'], main_curve['p'], marker='s', label='Main model (noisy-OR CPT)')
plt.xlabel('Number of active risk causes')
plt.ylabel(f'Mean P({TARGET}=yes)')
plt.title('Noisy-OR: risk rises with active causes')
plt.ylim(0, 1)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""
    ),
    cell_md(
        """## Task 2.2b: classification (original, learned, naive Bayes)

Assignment: class variable `InsiderThreatIncident`; ROC/AUC on Test (100 cases); Train (100 cases).

### Step 6a — Train three models

| Model | Structure | Parameters |
|-------|-----------|------------|
| **Original BN + noisy-OR CPT** | Fixed expert DAG | `learnParameters` on train |
| **Learned BN HC** | Hill Climbing on train | Structure + CPTs learned |
| **Naive Bayes** | Star to incident | pyAgrum `BNClassifier` |

`infer_probs` computes per test case $P(\\text{InsiderThreatIncident}=\\text{yes})$ via `LazyPropagation`."""
    ),
    cell_code(
        """# pyAgrum: original (causal DAG + learned parameters) vs learned HC vs naive Bayes
set_global_seed(SEED)
train_csv = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
train_csv.close()
train_cls.to_csv(train_csv.name, index=False)

learner_true = gum.BNLearner(train_csv.name, bn)
learner_true.useSmoothingPrior()
bn_true_fitted = learner_true.learnParameters(bn.dag())

learner_hc = gum.BNLearner(train_csv.name)
learner_hc.useGreedyHillClimbing()
learner_hc.useScoreBIC()
bn_hc = learner_hc.learnBN()

clf_naive = skbn.BNClassifier(learningMethod='NaiveBayes', scoringType='BIC')
clf_naive.fit(data=train_cls, targetName=TARGET)

def infer_probs(model, df):
    probs = []
    for _, row in df.iterrows():
        ev = {c: row[c] for c in df.columns if c != TARGET}
        ie = gum.LazyPropagation(model)
        ie.setEvidence(ev)
        ie.makeInference()
        probs.append(float(ie.posterior(TARGET)[{TARGET: 'yes'}]))
    return np.array(probs)

y_true = (test_cls[TARGET] == 'yes').astype(int).to_numpy()
probs_true = infer_probs(bn_true_fitted, test_cls)
probs_hc = infer_probs(bn_hc, test_cls)
probs_naive = clf_naive.predict_proba(test_cls.drop(columns=[TARGET]))[:, 1]
os.unlink(train_csv.name)"""
    ),
    cell_code(
        """def summarize(name, probs):
    fpr, tpr, _ = roc_curve(y_true, probs)
    return {
        'Model': name,
        'AUC': auc(fpr, tpr),
        'Accuracy': accuracy_score(y_true, (probs >= 0.5).astype(int)),
        'fpr': fpr,
        'tpr': tpr,
    }

results = [
    summarize('Original BN + noisy-OR CPT', probs_true),
    summarize('Learned BN HC', probs_hc),
    summarize('Naive Bayes (pyagrum)', probs_naive),
]
pd.DataFrame([{k: v for k, v in r.items() if k in ['Model', 'AUC', 'Accuracy']} for r in results])"""
    ),
    cell_md(
        """**Figure: ROC curve — classification `InsiderThreatIncident`**

| Element | Explanation |
|---------|-------------|
| **X-axis FPR** | Fraction of negatives incorrectly classified as incident |
| **Y-axis TPR** | Fraction of incidents correctly detected (recall) |
| **Diagonal** | Random classifier (AUC = 0.5) |
| **AUC in legend** | Higher = better risk ranking across test cases |
| **Three models** | Original (expert structure) vs. fully learned vs. Naive Bayes baseline |

**Interpretation for Task 2.2b:** report AUC and accuracy (threshold 0.5). With small train ($n=100$) the Learned BN may overfit; Original BN combines domain knowledge with data."""
    ),
    cell_code(
        """plt.figure(figsize=(8, 6))
for r in results:
    plt.plot(r['fpr'], r['tpr'], label=f"{r['Model']} (AUC={r['AUC']:.3f})")
plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title(f'ROC: classification of {TARGET}')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()"""
    ),
    cell_md(
        """### Step 6b — Stability: 95% interval over repeated splits

With Train/Test $=100/100$ a single random split can be misleading. We repeat the experiment over 30 seeds (42–71): per seed a new pool of 300 cases and a stratified 100/100 split.

Report **mean** and **95% percentile interval** (2.5th–97.5th percentile) for AUC and accuracy. Overlapping intervals mean: no model is clearly better on these small datasets."""
    ),
    cell_code(
        """N_STABILITY = 30
STABILITY_SEEDS = list(range(SEED, SEED + N_STABILITY))

def percentile_ci(values, lo=2.5, hi=97.5):
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.percentile(arr, lo)), float(np.percentile(arr, hi))

def repo_root():
    from pathlib import Path
    for candidate in (Path.cwd(), Path.cwd().parent):
        if (candidate / 'report' / 'classification_stability_worker.py').is_file():
            return candidate
    raise FileNotFoundError('Could not locate report/classification_stability_worker.py')

def run_classification_stability_subprocess(data, seeds=STABILITY_SEEDS):
    import json
    import subprocess
    import sys
    root = repo_root()
    data_path = root / 'report' / 'data' / 'synthetic_cases_n2000_seed42.csv'
    cache_dir = root / 'report' / 'data' / 'classification_stability'
    if not data_path.is_file():
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(data_path, index=False)
    cache_dir.mkdir(parents=True, exist_ok=True)
    worker = root / 'report' / 'classification_stability_worker.py'
    rows = []
    for seed in seeds:
        cache_path = cache_dir / f'seed_{seed:03d}.json'
        if cache_path.is_file():
            rows.extend(json.loads(cache_path.read_text(encoding='utf-8')))
            continue
        env = os.environ.copy()
        env['PYTHONHASHSEED'] = str(seed)
        env.setdefault('OMP_NUM_THREADS', '1')
        out = subprocess.check_output(
            [sys.executable, str(worker), str(data_path), str(seed)],
            env=env,
            text=True,
        )
        cache_path.write_text(out, encoding='utf-8')
        rows.extend(json.loads(out))
    return pd.DataFrame(rows)

stability_df = run_classification_stability_subprocess(data)
summary_rows = []
for model, grp in stability_df.groupby('Model'):
    auc_mean, auc_lo, auc_hi = percentile_ci(grp['AUC'])
    acc_mean, acc_lo, acc_hi = percentile_ci(grp['Accuracy'])
    summary_rows.append({
        'Model': model,
        'AUC_mean': auc_mean,
        'AUC_95%_CI': f'[{auc_lo:.3f}, {auc_hi:.3f}]',
        'Accuracy_mean': acc_mean,
        'Accuracy_95%_CI': f'[{acc_lo:.3f}, {acc_hi:.3f}]',
    })
pd.DataFrame(summary_rows)"""
    ),
    cell_code(
        """fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
order = ['Original BN + noisy-OR CPT', 'Learned BN HC', 'Naive Bayes']
for ax, metric, title in zip(axes, ['AUC', 'Accuracy'], ['AUC', 'Accuracy (threshold 0.5)']):
    sns.boxplot(data=stability_df, x='Model', y=metric, order=order, hue='Model', palette='Set2', legend=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(['Original BN', 'Learned BN', 'Naive Bayes'], rotation=12, ha='right')
plt.suptitle(f'Classification stability ({N_STABILITY} repeated 100/100 splits)')
plt.tight_layout()
plt.show()"""
    ),
    cell_code(
        """# Ground-truth vs learned comparison (compact)
cmp = bnvsbn.GraphicalBNComparator(bn, bn_hc)
print('Skeleton scores:', cmp.skeletonScores())
print('Directed scores:', cmp.scores())
cmp"""
    ),
    cell_md(
        """---

## Wrap-up — deliverables for the assignment

| Task | Key result in this notebook |
|------|-----------------------------|
| **Task 1** | DAG, scenarios, marginals, conditionals, sensitivity |
| **Task 2.2a** | `recovery_df`, structure figure, `sideBySide` |
| **RQ4** | Noisy-OR vs. full CPT, monotonicity, response curve |
| **Task 2.2b** | AUC/accuracy table, ROC figure, stability interval |

See `report/Report_Insider_Threat_BN.md` for the written report. Figures with CI: `python report/export_figures.py`."""
    ),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

out = Path(__file__).parent / "insider_threat_bn_prototype.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {out}")
