# Research Proposal: Bayesian Network for Insider-Related Cyber Security Risk

Jan Ruijgrok (852796035)  
Open Universiteit Nederland

## Abstract

Cyber security incidents are difficult to model statistically because breach and insider-threat data are scarce, incomplete, and rarely shared across organizations. Bayesian Networks (BNs) are nevertheless widely used in this domain precisely because they combine expert knowledge with limited empirical evidence and support inference under partial observability. Following a systematic review of standard BN models in cyber security (Chockalingam, Pieters, Herdeiro Teixeira, & van Gelder, 2017), this project develops a compact, causally motivated BN (8 nodes) for **malicious insider-related security risk** in an organizational IT environment.

The central objective is to assess whether a manually designed BN yields both (i) plausible probabilistic inferences—aligned with patterns reported in the literature—and (ii) competitive classification performance relative to data-driven learned structures and a naive Bayes baseline. The working expectation is that the expert-specified causal model will offer superior interpretability for risk analysts and security officers, while learned structures may approach or exceed predictive performance as training sample size increases.

Methodologically, the proposal follows the course assignment through three stages: (1) design and validation of an original BN in pyAgrum, (2) structure-learning experiments under varying sample sizes, and (3) comparative classifier evaluation using ROC analysis and AUC.

## Background

Standard (static) Bayesian Networks have become an established modelling technique in cyber and information security. Chockalingam et al. (2017) identified **17 standard BN models** in the literature and analysed them along eight characteristics, including data sources for DAGs and CPTs, threat-actor type, application area, and validation approach. Their main findings are directly relevant to this project:

- **Data scarcity:** Historical incident data are limited; **expert knowledge** dominates CPT specification in most reviewed models (11 of 17 relied exclusively on experts for CPTs).
- **Threat focus:** Standard BNs are used disproportionately for problems involving **malicious insiders**; integrated models covering both insiders and outsiders remain a research gap.
- **Scope:** Variables often span the **People–Process–Technology** elements of security; predictive purposes (reasoning from causes to effects) outnumber purely diagnostic uses.
- **Applications:** Risk management, threat hunting, and forensic investigation are common; models typically remain **small** (16 of 17 use fewer than 40 nodes).

Representative insider-oriented work includes psychosocial and behavioural precursors (Greitzer et al., 2012; 2010) and Bayesian models combining multiple insider indicators (Axelrad et al., 2013). This proposal does not replicate those large-scale models but **abstracts their causal logic** into a tractable 8-node network suitable for a master's assignment: explicit structure, interpretable CPTs, synthetic data for learning experiments, and classifier benchmarking.

The modelling approach is grounded in probabilistic graphical models (Koller & Friedman, 2009), BN practice (Scutari & Denis, 2021), and the cyber-security BN landscape synthesised by Chockalingam et al. (2017).

## Research Questions

**Working hypothesis:** A domain-informed causal BN constitutes a robust and interpretable baseline for insider-related incident risk, whereas structure learning provides incremental structural recovery and predictive gains primarily at larger sample sizes.

- **RQ1 (Model validity):** Does the manually specified BN produce plausible marginals, conditional probabilities, and directionally coherent effects for \(P(\text{InsiderThreatIncident}=\text{yes} \mid \text{evidence})\)?
- **RQ2 (Structure learning and sample size):** To what extent do search-and-score and constraint-based algorithms recover the original structure at training sizes \(n=100\), \(n=500\), and \(n=1000\)?
- **RQ3 (Classification performance):** How does predictive performance (ROC/AUC) compare across (a) the original BN, (b) a learned BN, and (c) a naive Bayes classifier on a held-out test set?
- **RQ4 (Noisy-OR parametrization):** To what extent does a noisy-OR parameterization of `InsiderThreatIncident` reduce CPT complexity while preserving plausible, monotonic risk accumulation compared with a full tabular CPT?

## Research Methods

### Problem definition and network design

**Target quantity:**

\[
P(\text{InsiderThreatIncident}=\text{yes} \mid \text{evidence})
\]

where **InsiderThreatIncident** is the binary class variable (materialized insider-related security incident).

**Design principle:** An 8-node DAG integrating People, Process, and Technology factors, consistent with the scope patterns in Chockalingam et al. (2017) and insider-threat BN literature.

### Variables (8 nodes)

| Variable | States | P–P–T element | Description |
|----------|--------|----------------|-------------|
| JobDissatisfaction | {low, medium, high} | People | Level of employee dissatisfaction or frustration with work or the work environment. |
| SecurityAwareness | {low, medium, high} | People | Employee knowledge and awareness of security risks and safe behaviour. |
| PolicyCompliance | {good, poor} | Process | Extent to which employees and processes adhere to security policy. |
| ConcerningBehaviour | {no, yes} | People | Whether behaviour has been observed that suggests possible malicious intent or misuse (e.g. anomalous data access). |
| PrivilegedAccess | {appropriate, excessive} | Technology / Process | Whether access rights are appropriate for the role or excessively granted. |
| TechnicalControls | {weak, adequate} | Technology | Strength of preventive and detective technical measures (firewalls, logging, DLP, etc.). |
| MonitoringAlert | {no, yes} | Technology | Whether the monitoring system has generated an alert based on behavioural or technical signals. |
| InsiderThreatIncident | {no, yes} | Outcome | Whether a malicious insider-related security incident has actually occurred (e.g. data breach or access misuse). |

### Initial causal structure

```
JobDissatisfaction        → ConcerningBehaviour
SecurityAwareness         → PolicyCompliance
PolicyCompliance          → PrivilegedAccess
ConcerningBehaviour       → InsiderThreatIncident
PrivilegedAccess          → InsiderThreatIncident
TechnicalControls         → InsiderThreatIncident
TechnicalControls         → MonitoringAlert
ConcerningBehaviour       → MonitoringAlert
```

**Rationale (brief):** Dissatisfaction and observable concerning behaviour capture psychosocial precursors (cf. Greitzer et al.; Axelrad et al.). Security awareness and policy compliance model process discipline affecting misuse of access. Technical controls and monitoring represent detective/preventive technology layers. The incident node aggregates human and technical pathways—an interpretable proxy for “insider threat materialized” rather than a full enterprise SOC model.

### Parameterization

CPTs are initialized using:

1. **Expert judgement** with monotonic relations where appropriate (e.g., higher dissatisfaction → higher \(P(\text{ConcerningBehaviour}=\text{yes})\));
2. **Directional consistency** with insider-threat literature and the review's observation that expert knowledge is the dominant CPT source;
3. **Limited calibration** for internal plausibility (no claim of population-level calibration from real incident databases).

The goal is an **internally consistent, defensible** model for inference and simulation—not replication of classified organizational data.

### Task 1: Analysis in pyAgrum

- Inspection of variable marginals;
- Conditional queries, e.g. \(P(\text{InsiderThreatIncident} \mid \text{ConcerningBehaviour}, \text{PrivilegedAccess})\);
- Evidence-propagation scenarios (e.g., alert raised while access remains excessive);
- Sensitivity checks on key CPT entries.

### Task 2.2: Learning experiments

**Data generation:** Synthetic samples from the original BN.

- Structure recovery: \(n \in \{100, 500, 1000\}\);
- Classifier comparison: **Train = 100 / Test = 100** (per assignment), stratified on **InsiderThreatIncident**.

**Algorithms:** Search-and-score (e.g., hill climbing) and constraint-based (PC-style / essential-graph workflow in pyAgrum).

**Structure comparison:** Visual diff and counts of correct, missing, reversed, and spurious arcs vs. ground truth.

### Classification comparison

**Class variable:** InsiderThreatIncident {no, yes}.

**Models:**

1. **original** — manually designed BN;
2. **learned** — BN structure (and parameters) learned from training data;
3. **naive** — naive Bayes on the same features.

**Evaluation:** ROC curve and AUC (primary); optionally accuracy and confusion matrix.

### Task 2.3: Noisy-OR parametrisation (required)

The CPT of **InsiderThreatIncident** is specified with a **noisy-OR** over three activators:

| Activator | Active when |
|-----------|-------------|
| Concerning behaviour | `ConcerningBehaviour = yes` |
| Access misuse | `PrivilegedAccess = excessive` |
| Weak controls | `TechnicalControls = weak` |

With leak probability \(q\) and inhibitions \(p_i\): \(P(I=\text{no}\mid\cdot)=q\prod_{i:\text{cause}_i} p_i\). This is compared with a **full logistic CPT** on the same parent set (parameter count, scenario rankings, response curves).

## Positioning relative to Chockalingam et al. (2017)

| Review finding | How this project responds |
|----------------|---------------------------|
| Expert-heavy CPTs | Explicit expert elicitation and documentation |
| Malicious insider focus | Core theme of the 8-node model |
| Small models (&lt;40 nodes) | 8 nodes, shallow structure |
| Gap: insider + outsider integration | Acknowledged as limitation; see Future Work roadmap |
| Limited validation in literature | Sensitivity analysis + synthetic recovery experiments |

## Expected Contributions

- An interpretable BN for insider-related incident risk with explicit People–Process–Technology assumptions;
- Empirical evidence on structure-learning robustness at small and moderate \(n\);
- A unified comparison of causal, learned, and naive Bayes classifiers in one security domain;
- A reproducible pyAgrum workflow aligned with the course assignment.

## Time Schedule (10 Weeks)

| Weeks | Activity |
|-------|----------|
| 1–2 | Literature (review + 1–2 exemplar papers), variable and arc specification |
| 3 | CPT design, Task 1 inference and plausibility checks |
| 4 | Synthetic data pipeline |
| 5–6 | Structure learning at \(n=100/500/1000\) |
| 7 | Classifiers and ROC/AUC |
| 8 | Sensitivity analysis and limitation section |
| 9–10 | Report, figures, reproducibility |

## Risk Analysis

| Risk | Mitigation |
|------|------------|
| Oversimplification of insider threat | Clear scope; cite review gaps; no claim of operational SOC deployment |
| Subjective CPTs | Monotonicity checks, sensitivity analysis, comparison with learned parameters |
| Small-\(n\) structure instability | Multiple random seeds; report structural patterns, not single runs |
| Simulation circularity (data from same BN) | Transparent generation protocol; interpret AUC as method comparison, not external validation |

## Future Work and Roadmap Toward SOC-Relevant Deployment

The present project deliberately remains a **proof of concept**: an interpretable, compact BN evaluated on synthetic data. It is **not** an operational Security Operations Center (SOC) product. Nevertheless, the same modelling philosophy—causal structure, explicit uncertainty, and analyst-facing explanations—maps naturally onto insider-threat workflows in enterprise SOCs. This section outlines a realistic path from the assignment prototype toward SOC-relevant tooling.

### What a SOC product must deliver beyond academic inference

SOC analysts typically require more than a posterior probability. They need:

- **Prioritisation:** which users, teams, or accounts warrant attention now;
- **Explanation:** why a case is elevated (which evidence nodes and causal pathways matter);
- **Actionability:** recommended next steps (access review, enhanced monitoring, HR/security joint review);
- **Trust and governance:** controlled false-positive rates, audit trails, and defensible reasoning under privacy constraints.

The proposed BN already supports **explainability** and **uncertainty-aware scoring**. A product would additionally require **live data integration**, **workflow embedding**, and **validation on organisational evidence**—not merely larger networks.

### Roadmap (staged)

#### Stage 1 — From synthetic to semi-real pilot data

- Connect **operational signal sources**: HR or people analytics (e.g., dissatisfaction proxies), identity and access management (privileged access posture), governance/risk/compliance (policy compliance), and SIEM or UEBA (technical controls, monitoring alerts).
- Replace purely synthetic CPTs with **structured expert elicitation** supplemented by **limited historical cases** (even tens of labelled incidents can support calibration and threshold setting).
- Evaluate with **analyst-centred metrics** (ranking usefulness, explanation quality) in addition to ROC/AUC on simulated data.

#### Stage 2 — Analyst MVP (enrichment, not SIEM replacement)

Deliver a minimum viable capability:

- **Risk score plus top causal explanations** per subject (e.g., “elevated risk driven by concerning behaviour and excessive access; controls adequate → people/process dominated”).
- Deploy as **SIEM/SOAR enrichment** or a lightweight dashboard rather than replacing existing detection stacks.
- Enforce **human-in-the-loop** decision making: the BN advises; analysts adjudicate (critical for privacy, labour law, and proportionality).

Plausible product shapes at this stage:

| Form | Role in the SOC |
|------|------------------|
| Risk-scoring plugin | Enriches alerts and entity records in SIEM/SOAR |
| Insider-threat dashboard | Prioritised queue with causal narratives |
| What-if simulator | Tests interventions (“revoke excessive access”, “mandate training”) before action |

#### Stage 3 — Technical extensions aligned with literature gaps

Building on Chockalingam et al. (2017) and related insider-threat BN work:

- **Insider–outsider integration** (e.g., social engineering, collusion) to address the review’s lack of combined threat-actor models;
- **Temporal modelling** (dynamic BNs or sliding windows) to capture deteriorating behaviour over weeks;
- **Complementary attack-path modelling** (e.g., Bayesian attack graphs) for technical kill-chain context alongside people-centric factors;
- **Sector-specific variants** (e.g., ICS/control-room constraints) where people-centric variables differ from generic IT offices.

#### Stage 4 — Operational trust and continuous assurance

- **Drift monitoring** when input distributions shift; trigger CPT or threshold review;
- **Configurable false-positive budgets** per organisational risk appetite;
- **Governance:** data-protection impact assessment, minimisation of HR-sensitive attributes, role-based access to scores;
- **Validation beyond ROC:** red-team scenarios, tabletop exercises, and periodic expert review.

### How the current project enables this path

| Current assignment artefact | SOC-relevant carry-over |
|---------------------------|-------------------------|
| 8-node P–P–T causal DAG | Template for explainable risk decomposition |
| Noisy-OR incident node | Analyst-intuitive “multiple weak signals” composition |
| Scenario and sensitivity analysis | Foundation for what-if and policy exploration |
| Causal vs. learned vs. naive comparison | Evidence on when **not** to let data alone redefine structure (audit/compliance) |
| pyAgrum workflow | Prototype engine for inference and later integration |

### Scope boundary (explicit)

Without an organisation willing to share **appropriate data** and SOC staff willing to **adopt** explainable scores, the model remains academic. Progress toward productisation depends less on adding nodes than on **data partnerships**, **workflow fit**, and **sustained validation**—areas explicitly out of scope for the current assignment but central to future work.

## References

Axelrad, E. T., Sticha, P. J., Brdiczka, O., & Shen, J. (2013). A Bayesian network model for predicting insider threats. In *2013 IEEE Security and Privacy Workshops* (pp. 82–89). IEEE.

Chockalingam, S., Pieters, W., Herdeiro Teixeira, A., & van Gelder, P. (2017). Bayesian network models in cyber security: A systematic review. In H. Lipmaa, A. Mitrokotsa, & R. Matulevicius (Eds.), *Proceedings of NordSec 2017* (LNCS 10674, pp. 105–122). Springer. https://doi.org/10.1007/978-3-319-70290-2_7

Greitzer, F. L., et al. (2012). Identifying at-risk employees: Modeling psychosocial precursors of potential insider threats. In *Proceedings of HICSS 2012* (pp. 2392–2401). IEEE.

Koller, D., & Friedman, N. (2009). *Probabilistic Graphical Models: Principles and Techniques*. MIT Press.

Scutari, M., & Denis, J.-B. (2021). *Bayesian Networks: With Examples in R* (2nd ed.). Chapman and Hall/CRC.
