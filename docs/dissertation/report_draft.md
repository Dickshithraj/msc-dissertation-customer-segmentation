# Optimizing Targeted Marketing: A Framework for Customer Segmentation and Automated Cluster-Based Notification

**MSc Advanced Computer Science (Data Analytics) — COMP5200M**
**Dickshith Raj Nagaraj — University of Leeds**
**Supervisor: Dr Fahad Panolan**

> ⚠️ DRAFT for review — follows the School of Computing template for a Software Product /
> System Design project. Rewrite in your own voice, declare AI assistance per School policy,
> verify every reference marked [Verify], and insert the system-architecture diagram (Figure 3.1)
> and dashboard screenshots (Chapter 4) where indicated.

## Abstract
Online retailers accumulate vast transactional records, yet a large share of marketing
communication is still delivered as undifferentiated "blanket" campaigns that waste budget and
erode customer goodwill. This project designs, implements, and evaluates a reproducible software
framework that closes the gap between raw transaction data and action by converting that data
into an automated, financially-evaluated marketing-notification plan. Using the UCI Online
Retail II dataset, the system cleans approximately 1.07 million transactions down to 793,591
audited purchase lines, and engineers an extended Recency–Frequency–Monetary (RFM) feature set
for 5,878 unique customers. It then benchmarks six clustering algorithms drawn from five
algorithmic families (centroid, density, probabilistic, hierarchical, and graph-based) under
principled hyper-parameter selection and 50-round bootstrap stability analysis, and applies a
transparent, pre-declared rule that selects HDBSCAN as the operational segmentation. The
resulting clusters drive a rule-based notification engine that is modulated by each customer's
lifetime value and churn risk. Customer lifetime value is estimated with the BG/NBD and
Gamma–Gamma probabilistic models; churn is modelled by six supervised classifiers under a strict
target-leakage guard, with pairwise differences assessed for statistical significance using
McNemar's test. Finally, a 10,000-iteration Monte Carlo simulation quantifies the campaign's
return on investment against an untargeted baseline: the targeted plan achieves an estimated
mean ROI of 38.4× and a **+168% profit uplift** over blanket marketing. The principal
contribution is a closed-loop, auditable pipeline — from raw data to a costed action plan —
built with reproducibility and statistical rigour throughout, together with a clear demonstration
that behaviour-driven targeting is financially superior to undifferentiated contact.

## Acknowledgements
I would like to thank my supervisor, Dr Fahad Panolan, for his guidance, patience, and
constructive feedback throughout this project. I am also grateful to the University of Leeds
School of Computing for providing the resources and learning environment that made this work
possible, and to the maintainers of the UCI Machine Learning Repository and the open-source
Python ecosystem on which the implementation depends. *(Add any personal acknowledgements here.)*

## Table of Contents

## List of Figures

## List of Tables

## Chapter 1. Introduction

### 1.1 Project Aim
The aim of this project is to design, implement, and evaluate a reproducible software framework
that uses unsupervised machine learning to segment retail customers from their behavioural
transaction data, and that converts those segments into an automated, financially-evaluated
marketing-notification plan.

The commercial context is a well-recognised inefficiency in retail marketing. Modern Customer
Relationship Management (CRM) is shifting from *static* profiling — grouping customers by fixed
demographic attributes such as age or location — toward *dynamic*, behaviour-driven engagement,
in which segments emerge from what customers actually do and are recomputed as behaviour changes
[15], [24]. Despite this shift, a great deal of outbound communication remains untargeted: the
same email, the same offer, and the same timing are sent to an entire customer base regardless of
whether a recipient is a loyal high-value buyer or a lapsed one-time purchaser. This is wasteful
on two fronts. It spends marketing budget on customers who will not respond, and it risks
fatiguing or alienating valuable customers with irrelevant contact, thereby *reducing* the very
lifetime value the campaign was meant to protect.

The core concepts addressed by this project are defined below for a technically literate reader
who is new to the marketing-analytics domain. **Customer segmentation** is the unsupervised
partitioning of a customer base into groups whose members behave similarly. **RFM** (Recency,
Frequency, Monetary) is the standard behavioural feature framework for retail, measuring how
recently, how often, and how much a customer purchases; it is favoured because each dimension
maps directly onto an intuitive marketing meaning [15], [25]. **Clustering** algorithms discover
behavioural groups from these features without any labelled "ground truth". **Customer Lifetime
Value (CLV)** estimates a customer's expected future monetary contribution, and **churn** is the
risk that a customer stops purchasing altogether. The practical importance of joining these
concepts is direct: better-targeted, well-timed communication improves retention and lifetime
value while reducing wasted spend, whereas static profiling leaves an unbridged gap between raw
transactional data and actionable strategy. This project is fundamentally about bridging that
gap in a way that is transparent, reproducible, and financially justified.

### 1.2 Objectives
The project is scoped by six measurable objectives, each of which is discharged and evidenced in
this report:

- **O1** — Conduct a systematic literature survey of customer segmentation, clustering algorithms,
  customer lifetime value, churn modelling, and the shift toward dynamic, analytics-driven CRM,
  and use it to justify the methodological choices (Chapter 2).
- **O2** — Acquire real retail transaction data and perform auditable cleaning, customer-level RFM
  feature engineering, and skew-aware scaling (Chapter 4.1–4.2).
- **O3** — Implement and benchmark multiple clustering algorithms with principled hyper-parameter
  selection (the Elbow Method and Silhouette Analysis, GMM by BIC, DBSCAN by the k-distance knee),
  and evaluate cluster validity and stability using internal validity indices and bootstrap
  analysis (Chapters 4.3–4.4 and 5.2).
- **O4** — Build a predictive layer estimating each customer's 12-month lifetime value (BG/NBD +
  Gamma–Gamma) and churn probability (six classifiers compared under a leakage guard with
  statistical significance testing) (Chapters 4.5–4.6 and 5.3).
- **O5** — Design and implement a notification-logic framework that translates cluster
  characteristics, CLV tier, and churn band into actionable, prioritised marketing triggers per
  customer, exposed through an on-demand recommendation API and an interactive dashboard
  (Chapters 4.7–4.8).
- **O6** — Quantify the return on investment and profit uplift of the targeted plan relative to an
  untargeted (static) baseline using a Monte Carlo cost–benefit framework (Chapters 4.8 and 5.4).

### 1.3 Deliverables
The objectives map onto five concrete deliverables:

1. A version-controlled Python codebase implementing the full pipeline: extract–transform–load
   (ETL), feature engineering, six-algorithm clustering, cluster validation, CLV and churn
   modelling, the rule-based notification engine, and the Monte Carlo ROI simulation (O2–O6).
2. A reproducible experimentation and validation record — the tables and figures under
   `outputs/` — documenting model selection, internal validation metrics, and statistical tests
   (O3, O4).
3. A notification-logic design framework mapping cluster characteristics to marketing triggers,
   exposed via an on-demand `recommend(customer_id)` function and an interactive dashboard (O5).
4. A commercial-viability assessment quantifying the expected financial return and the ROI uplift
   over static marketing (O6).
5. This MSc project report (O1 and the written evaluation of all objectives).

### 1.4 Ethical, Legal, and Social Issues
The project raises limited but non-trivial ethical, legal, social, and professional
considerations, which are addressed by design rather than as an afterthought.

**Legal and data protection.** The dataset (UCI Online Retail II [15]) contains no personally
identifiable information beyond an anonymised numeric Customer ID; no names, addresses, contact
details, or payment data are present. No attempt is made to re-identify any individual, and the
work is consistent with the principles of the UK GDPR — in particular purpose limitation and
data minimisation, since only the fields required for behavioural modelling are used. The raw
data is treated as strictly read-only and is never altered in place; all derived artefacts are
written to separate locations, preserving provenance.

**Ethical and social.** The project involves no human participants and no primary data collection,
so no ethical approval or consent process is required (see Appendix A of the accompanying Scoping
and Planning document). A wider social concern nevertheless applies to any behaviour-based
targeting system: such systems can, if misused, over-contact customers, entrench unfair treatment
between customer groups, or exploit vulnerable individuals through manipulative offers. This
framework mitigates that risk structurally. The notification engine is deliberately *rule-based
and auditable* rather than an opaque black box, so every recommendation can be inspected, traced
to the segment, CLV tier, and churn band that produced it, and overridden by a human operator.
Contact timing is derived from each customer's own purchase cadence rather than a fixed drumbeat,
which discourages over-contact.

**Professional.** The work follows standard software-engineering practice — modular design under
version control, a configuration module in place of hard-coded constants, automated unit tests,
and fixed random seeds for reproducibility — reflecting the professional responsibility to
produce reliable, maintainable, and reproducible analytical software.

## Chapter 2. Background Research

### 2.1 Literature Survey
Customer segmentation has a long history in marketing analytics, and the RFM framework remains
the dominant behavioural model for retail because it is interpretable and maps directly onto
marketing strategy [15], [25]. Chen, Sain, and Guo [15] applied RFM-based segmentation with data
mining to online-retail data — the same dataset family used in this project — and demonstrated
that behavioural segments derived from transactions are markedly more actionable than segments
based on static demographic attributes, because they reflect current engagement rather than fixed
identity. Subsequent theoretical work has sought to enrich RFM. Zhang, Bradlow, and Small [26]
generalise RFM to RFMC by adding a "clumpiness" dimension that captures whether purchases are
evenly spaced or bursty, showing that regularity of purchasing carries predictive value beyond
recency and frequency alone. Other authors have proposed RFM ranking and scoring schemes as
lightweight alternatives to clustering [25]. This project adopts the classical RFM core and
augments it with four additional behavioural signals (tenure, average order value, average
inter-purchase interval, and product breadth) rather than the clumpiness formulation, since these
signals are directly interpretable to a marketing stakeholder and inexpensive to compute.

A substantial body of applied work compares clustering algorithms for segmentation, and its
consistent finding motivates a central design decision of this project. Studies benchmarking
centroid-based methods (K-Means) against density-based methods (DBSCAN, HDBSCAN) repeatedly report
that *no single algorithm dominates*: the appropriate choice depends on cluster geometry, the
presence and treatment of outliers, and whether the number of clusters is known in advance [20],
[21], [40]. Recent work has also automated the selection of the number of clusters *k* in
RFM-based K-Means [27], addressing a common weakness of centroid methods. Because the literature
offers no universally best clustering algorithm, this project does not commit to one a priori;
instead it benchmarks six algorithms across five families and selects among them by an explicit
rule (Section 2.3). Segmentation has been studied across many sectors — including retail [15],
banking [29], and broader machine-learning-driven consumer analytics [28] — confirming the
generality of the approach while underlining that the *validation* of segments, not merely their
production, is what distinguishes rigorous work.

The literature increasingly frames segmentation within *dynamic* CRM, embedding analytics in the
marketing engine so that the recommended next action for a customer can be recomputed as their
behaviour changes [24], [36]. Within this framing, two predictive tasks recur alongside
segmentation. First, probabilistic CLV models estimate future value in *non-contractual* retail
settings, where customers can lapse silently without formally cancelling. The BG/NBD "counting
your customers" model [17] and the companion Gamma–Gamma monetary-value model [16] are the
established tools here and have been applied in domains as varied as financial services [30]; more
recent work combines probabilistic and machine-learning approaches to CLV [31]. Second, churn
prediction is typically framed as supervised classification over behavioural features, with
ensemble methods such as random forests [8] and gradient-boosted trees [9] generally leading on
retail and telecom benchmarks [32], an area comprehensively surveyed by recent literature [33]. A
recurring methodological theme in this predictive work is the danger of *target leakage* when the
classification label is itself derived from an input feature, and the accompanying need to handle
*class imbalance*, since churners are typically a small minority. Both issues are treated
explicitly in this project (Section 4.6).

Finally, several integrated studies combine two or more of these tasks: segmentation with churn
prediction on online-retail data [34], hybrid RFM-and-clustering frameworks for churn [22], and
segmentation pipelines that feed marketing strategy [35] or personalised recommendation [37]. The
financial value of targeting is increasingly expressed as *uplift* over an untargeted baseline,
drawing on causal-inference and uplift-modelling techniques [38], [39]. However, most prior work
stops at the modelling stage: it produces segments or predictions but does not close the loop to
an auditable campaign-action layer, and rarely quantifies the ROI *uplift* of targeting over
static marketing in an end-to-end, reproducible manner. This unaddressed gap — the missing link
between validated analytics and a costed, auditable action plan — is precisely the contribution
of this project.

### 2.2 Methods and Techniques
This section reviews the specific techniques adopted, grouping the clustering algorithms by
family. **Centroid-based** K-Means partitions data into *k* spherical clusters by iteratively
minimising within-cluster variance [1], [2]; it is fast and interpretable but assumes convex,
similarly-sized clusters and requires *k* to be fixed in advance. **Density-based** DBSCAN groups
dense regions and labels sparse points as noise without requiring *k* [3], [4], which makes it
robust to outliers; HDBSCAN extends DBSCAN hierarchically and selects clusters by their stability
across a range of density thresholds, removing DBSCAN's sensitive global radius parameter [5],
[6]. **Probabilistic** Gaussian Mixture Models (GMM) fit a weighted sum of Gaussian components via
the Expectation–Maximisation algorithm [7], giving soft cluster memberships. **Hierarchical**
agglomerative clustering merges points bottom-up, with Ward linkage favouring compact clusters,
and **graph-based** spectral clustering embeds the data via the eigenvectors of a
similarity-graph Laplacian before clustering in the embedded space, which can capture
non-convex structure.

Hyper-parameter selection uses established techniques matched to each algorithm: the Elbow Method
with automated knee detection [10] and Silhouette Analysis [11] for K-Means, the Bayesian
Information Criterion for GMM component count, and the k-distance knee for DBSCAN's radius [3].
Because these heuristics can disagree — as they do here for K-Means — the project reports them
side by side rather than trusting a single one. Cluster quality is then assessed with three
complementary internal validity indices: the **Silhouette** coefficient measuring cohesion versus
separation [11], the **Davies–Bouldin** index (lower is better) [12], and the
**Calinski–Harabasz** index (higher is better) [13]. Because internal indices reward geometric
tightness but say nothing about *robustness*, cluster **stability** is measured separately by
resampling: each algorithm is refitted on 50 bootstrap samples and the agreement between
partitions is quantified by the Adjusted Rand Index (ARI) [14].

The predictive layer uses BG/NBD [17] and Gamma–Gamma [16] for CLV, and a panel of six churn
classifiers — logistic regression, random forest [8], XGBoost [9], gradient boosting, decision
tree, and k-nearest neighbours — compared with McNemar's test for paired classifiers [18]. The
decision layer is an auditable rule-based engine, and the commercial evaluation uses Monte Carlo
simulation to propagate response-rate uncertainty into a full distribution of ROI. The
implementation is in Python using scikit-learn [19], the `lifetimes` library for CLV, the
`hdbscan` library, and XGBoost [9].

### 2.3 Choice of Methods
The methodological choices follow directly from the literature and from the requirements of an
auditable, reproducible system. Rather than committing to a single clustering algorithm — which
the comparative literature [20], [21], [40] shows cannot be justified a priori — this project
benchmarks six algorithms across five families and adopts a *transparent, pre-declared selection
rule*: disqualify solutions whose noise fraction is excessive, require a mean bootstrap ARI of at
least 0.70 for stability, and among the survivors maximise the Silhouette coefficient. This makes
the final choice reproducible and defensible rather than a matter of taste. HDBSCAN is
particularly well suited to marketing because its explicit "noise" group naturally identifies
atypical customers who should *not* be forced into a segment or targeted with a generic campaign.

The feature set extends RFM with four behavioural signals, and any feature whose absolute skewness
exceeds 0.5 receives a `log1p` transform — chosen over a plain logarithm because it is defined at
zero, which matters for one-time buyers whose inter-purchase interval and some derived quantities
are zero — followed by standardisation, so that distance-based algorithms are not dominated by
Monetary's very large range. CLV is modelled with BG/NBD + Gamma–Gamma as the established
non-contractual approach [16], [17]. Churn is modelled with six classifiers under a strict leakage
guard: because the churn label is derived from Recency, Recency is *excluded* from the churn
feature set, and models are compared with McNemar's test so that apparent performance differences
are only claimed when statistically supported [18]. The notification engine is deliberately
rule-based for auditability, and ROI is estimated by Monte Carlo simulation that includes an
explicit static baseline so that the *uplift* of targeting can be isolated and reported. The
overall process follows the CRISP-DM methodology [23], proceeding through business understanding,
data understanding, data preparation, modelling, evaluation, and (via the dashboard) deployment.

### 2.4 Mathematical Formulation
To make every stage of the pipeline explicitly quantifiable, this section defines the key
quantities, objective functions, and evaluation statistics used throughout the project. Equation
numbers are referenced from the implementation (Chapter 4) and evaluation (Chapter 5) chapters. In
the notation below, *n* is the number of customers, *x* denotes a feature vector, *μ* and *σ* are a
feature's mean and standard deviation, and *K* is a number of clusters or classes.

**Feature engineering and preprocessing.** For customer *i* with snapshot date *t*₍now₎, the three
core RFM quantities are the days since the last purchase, the count of distinct invoices, and total
spend, respectively:

@eq Rᵢ = t₍now₎ − maxⱼ tᵢⱼ ,   Fᵢ = |invoices(i)| ,   Mᵢ = Σⱼ qᵢⱼ · pᵢⱼ

The decision to transform a feature is quantified by its sample skewness; a feature is
`log1p`-transformed when |γ₁| > 0.5, and then standardised to zero mean and unit variance:

@eq γ₁ = (1/n) Σᵢ (xᵢ − μ)³ ⁄ σ³

@eq x′ = ln(1 + x)

@eq z = (x − μ) ⁄ σ

Equation (2) quantifies distributional asymmetry; (3) compresses the long right tail while
remaining defined at *x* = 0; and (4) places all features on a common scale so that no single
feature dominates the Euclidean distances used downstream.

**Clustering objectives.** K-Means chooses assignments and centroids *μₖ* that minimise the
total within-cluster sum of squares (the inertia), while a Gaussian Mixture Model maximises the
log-likelihood of a weighted sum of *K* Gaussians:

@eq J = Σₖ Σ(x ∈ Cₖ) ‖x − μₖ‖²

@eq ℒ = Σᵢ ln Σₖ πₖ · 𝒩(xᵢ | μₖ, Σₖ)

Density methods (DBSCAN, HDBSCAN) instead define clusters as maximal regions of density above a
threshold, labelling low-density points as noise, so they optimise no single closed-form objective.

**Cluster validity and stability.** Three internal indices quantify partition quality. The
Silhouette coefficient of point *i* contrasts its mean intra-cluster distance *a*(*i*) with the
mean distance *b*(*i*) to the nearest other cluster; the Davies–Bouldin index averages the
worst-case cluster-pair overlap (lower is better); and the Calinski–Harabasz index is the ratio of
between- to within-cluster dispersion (higher is better):

@eq s(i) = ( b(i) − a(i) ) ⁄ max{ a(i), b(i) } ,   s(i) ∈ [−1, 1]

@eq DB = (1⁄K) Σᵢ maxⱼ≠ᵢ [ (σᵢ + σⱼ) ⁄ d(cᵢ, cⱼ) ]

@eq CH = [ tr(B) ⁄ (K − 1) ] ⁄ [ tr(W) ⁄ (n − K) ]

Cluster stability under 50 bootstrap resamples is quantified by the chance-corrected Adjusted Rand
Index between the full-data partition and each resampled partition, and the shared PCA view retains
the components with the largest explained-variance ratio:

@eq ARI = ( RI − E[RI] ) ⁄ ( max RI − E[RI] ) ,   ARI ∈ [−1, 1]

@eq EVRₘ = λₘ ⁄ Σⱼ λⱼ

**Customer lifetime value.** The BG/NBD model has purchase-rate parameters (*r*, *α*) and dropout
parameters (*a*, *b*). For a customer with *x* repeat purchases, last-purchase time *tₓ*, and
observation window *T*, the probability the customer is still active is:

@eq P(alive | x, tₓ, T) = [ 1 + δ(x>0) · (a ⁄ (b + x − 1)) · ((α + T) ⁄ (α + tₓ))^(r+x) ]⁻¹

@eq E[Y(t) | x, tₓ, T] = expected purchases in (T, T+t],  closed-form in (r, α, a, b) via ₂F₁

The Gamma–Gamma model then estimates a customer's expected transaction value as a
credibility-weighted average of the population mean *pν*⁄(*q*−1) and the customer's own observed
mean *mₓ*, and the discounted 12-month CLV combines the two models over the horizon *H* with a
per-period discount rate *d*:

@eq E[M | mₓ, x] = ( (q − 1) ⁄ (px + q − 1) ) · ( pν ⁄ (q − 1) ) + ( px ⁄ (px + q − 1) ) · mₓ

@eq CLV = Σ(t = 1…H) [ E[purchasesₜ] · E[M] ] ⁄ (1 + d)ᵗ

**Churn modelling.** The churn label is defined by thresholding Recency at its 90th percentile
(and Recency is then excluded from the features to prevent leakage). Class imbalance is corrected
by weighting each class inversely to its frequency, and the logistic-regression baseline models the
churn probability through the logistic function:

@eq yᵢ = 1  if  Rᵢ > Q₀.₉₀(R),   else 0

@eq w_c = N ⁄ (K · N_c)

@eq P(y = 1 | x) = σ(βᵀx + β₀),   σ(z) = 1 ⁄ (1 + e⁻ᶻ)

Tree-based models (random forest, gradient boosting, decision tree) split nodes to reduce Gini
impurity, where *p_c* is the proportion of class *c* at a node:

@eq G = 1 − Σ_c p_c²

**Evaluation metrics.** Classifier quality on the imbalanced churn task is quantified by precision,
recall, and their harmonic mean (F1), together with the area under the ROC curve, interpretable as
the probability that a random churner is scored above a random non-churner:

@eq Precision = TP ⁄ (TP + FP),   Recall = TP ⁄ (TP + FN)

@eq F1 = 2 · (Precision · Recall) ⁄ (Precision + Recall)

@eq ROC-AUC = P( ŝ(x⁺) > ŝ(x⁻) )

Whether two classifiers differ significantly is quantified by McNemar's test on their discordant
predictions, where *b* and *c* count the cases each model alone gets right; the exact test used
here evaluates min(*b*, *c*) against a Binomial(*b*+*c*, ½) null:

@eq χ²₁ = (|b − c| − 1)² ⁄ (b + c),   with  min(b, c) ∼ Binomial(b + c, ½)

**Commercial ROI simulation.** For each campaign action *a* the Monte Carlo simulation draws a
response rate from a Beta prior, the number of conversions from a Binomial, and adds Gaussian noise
to per-response revenue, so that profit and ROI become distributions rather than point estimates:

@eq rₐ ∼ Beta(αₐ, βₐ),    kₐ ∼ Binomial(nₐ, rₐ)

@eq Profit = Σₐ kₐ · v̄ₐ · (1 + εₐ) − Cost,   εₐ ∼ 𝒩(0, s²)

@eq ROI = Profit ⁄ Cost

Finally, the headline business result — the advantage of targeting over sending everyone the same
message — is quantified as the percentage profit uplift over the static baseline:

@eq Uplift% = 100 · (Profit₍targeted₎ − Profit₍static₎) ⁄ Profit₍static₎

## Chapter 3. Software Requirements and System Design

### 3.1 Software Requirements
The requirements are stated as functional requirements (what the system must do) and
non-functional requirements (qualities it must exhibit), so that the implementation and testing
chapters can be evaluated against them directly.

The **functional requirements (FR)** are:

- **FR1** — Load the raw two-year transaction dataset from its source workbook.
- **FR2** — Clean the data via documented, auditable rules and persist a cleaning summary.
- **FR3** — Engineer customer-level RFM and four extended behavioural features.
- **FR4** — Preprocess features with a skew-aware `log1p` transform and standardisation.
- **FR5** — Cluster customers with multiple algorithms and select hyper-parameters automatically.
- **FR6** — Validate clusters with internal indices and bootstrap stability, and select the best
  algorithm by a transparent rule.
- **FR7** — Profile each segment in original units and assign marketing-friendly names.
- **FR8** — Estimate per-customer CLV (probability-alive, expected future purchases, discounted
  value).
- **FR9** — Predict churn with several classifiers under a target-leakage guard and compare them.
- **FR10** — Compute year-on-year segment migration.
- **FR11** — Generate a per-customer notification plan and expose an on-demand recommendation API.
- **FR12** — Simulate campaign ROI, including the uplift over a static baseline.
- **FR13** — Provide an interactive dashboard over all artefacts.

The **non-functional requirements (NFR)** are:

- **NFR1 — Reproducibility.** Identical results on every run, guaranteed by a fixed analysis
  snapshot date and fixed random seeds throughout.
- **NFR2 — Modularity.** One module per pipeline stage, orchestrated by a single entry point, so
  that stages can be developed, tested, and re-run independently.
- **NFR3 — Configurability.** All paths and hyper-parameters live in a single configuration
  module, eliminating "magic numbers" scattered through the code.
- **NFR4 — Performance.** The full pipeline runs in a few minutes on commodity CPU hardware, with
  no GPU dependency, which suits the dataset scale and maximises portability.
- **NFR5 — Auditability.** Every marketing recommendation is rule-traceable and can be overridden.
- **NFR6 — Testability.** Automated tests cover the core logic on small synthetic datasets,
  independent of the raw data.

### 3.2 System Design
The system is designed as a modular, configuration-driven pipeline orchestrated by a single entry
point (`main.py`). Each stage is an independent module under `src/` with a clear input/output
contract, and intermediate artefacts are persisted to disk — Parquet for tabular data, CSV for
results tables, and PNG for figures — so that any stage can be re-run in isolation without
recomputing its predecessors. This satisfies the modularity and performance requirements and
makes debugging and incremental development straightforward.

Data flows in a single direction through the pipeline, which mirrors the CRISP-DM data-preparation
and modelling phases: *raw transactions → cleaned transactions → customer features → scaled
features → cluster assignments → segment profiles → CLV and churn scores → notification plan → ROI
assessment*. This one-way dependency structure means there are no circular dependencies between
stages and that the state of the system after any stage is fully captured by the artefacts on
disk. A central configuration module (`src/config.py`) is the single source of truth for all file
paths and hyper-parameters, satisfying NFR3 and keeping the design free of hard-coded constants.

Two consumer interfaces sit on top of the persisted artefacts rather than re-running the pipeline:
an on-demand `recommend(customer_id)` function that returns a single customer's recommendation,
and an interactive Streamlit dashboard that visualises every artefact across eight pages. Both read
the outputs produced by the batch pipeline, which cleanly separates the expensive analytical
computation from the cheap, interactive presentation layer. Conceptually the architecture forms
five layers — a data foundation, a segmentation layer, a customer-intelligence (prediction) layer,
a decision layer, and a delivery layer — each consuming the outputs of the one before it. *(Insert
the five-layer system-architecture / data-flow diagram as Figure 3.1, and a component diagram as
Figure 3.2.)*

## Chapter 4. Software Implementation

The system is implemented in Python with one module per pipeline stage under `src/`, orchestrated
by `main.py`. This chapter describes how each component was implemented and reports the concrete
results each stage produces on the Online Retail II data. Representative source and the dashboard
interface should be included as code snippets and screenshots where indicated.

### 4.1 Data Loading and Cleaning
Both yearly worksheets of the source workbook are loaded and concatenated into a single raw frame
of 1,067,371 rows. Cleaning then applies four sequential rules, each measured against the
already-filtered data so that row impacts are not double-counted. Rule 1 drops rows with a missing
Customer ID — by far the largest reduction, removing 243,007 rows (22.77%), since anonymous
till-style transactions cannot be attributed to a customer and are useless for customer-level
modelling. Rule 2 removes cancellation invoices (identified by the prefix `C`), removing 18,744
rows (2.27%). Rule 3 removes physically impossible or non-sale lines with a quantity below one or
a price below one penny (89 rows, 0.01%). Rule 4 drops exact duplicate rows (11,940 rows, 1.48%).
After cleaning, 793,591 transaction lines remain — a total reduction of 25.65% — from which line
revenue is derived and column dtypes are enforced. Persisting the audit table is a deliberate
requirement (FR2): it makes every exclusion transparent and justifiable to an examiner or auditor.

@table outputs/tables/cleaning_summary.csv | Data cleaning: the row impact of each sequential rule. | 30

### 4.2 Feature Engineering and Preprocessing
From the cleaned transactions, the pipeline computes eight customer-level features for each of the
5,878 unique customers, with a fixed analysis snapshot date (the day after the last transaction)
so that Recency is deterministic and the run is reproducible (NFR1). Recency is the days since the
customer's last purchase; Frequency counts *distinct invoices* (not line items, to avoid inflating
frequency for large baskets); Monetary is total spend; and the extended features are Tenure (days
between first and last purchase), Average Order Value, Average Inter-Purchase Interval, and
Distinct Products. The resulting distributions are heavily right-skewed, which is typical of retail
spend: mean Monetary is £3,009 but the median is only £887, and the maximum exceeds £608,000;
Frequency has a median of 3 but a maximum of 398. Such skew would let a single feature dominate the
Euclidean distances used by most clustering algorithms.

To correct this, any feature whose absolute skewness exceeds 0.5 (Eqs. 2–4) receives a `log1p` transform —
preferred over a plain logarithm because it is defined at zero, which matters for one-time buyers
whose Tenure and Average Inter-Purchase Interval are exactly zero — followed by standardisation to
zero mean and unit variance. The effect is visible in the figures below: the raw distributions are
long-tailed, whereas the transformed features are approximately symmetric and on a common scale,
which is the correct input geometry for distance- and density-based clustering.

@fig outputs/figures/rfm_distributions.png | Distributions of the RFM features (Frequency and Monetary shown on log scales).

@fig outputs/figures/scaling_effect.png | Feature distributions before (raw) and after the log1p + standardisation preprocessing.

### 4.3 Clustering and Model Selection
Six algorithms are implemented behind a common interface: K-Means, DBSCAN, GMM, HDBSCAN,
Agglomerative (Ward linkage), and Spectral. Hyper-parameters are selected automatically and
per-algorithm. For K-Means, which minimises the within-cluster sum of squares (Eq. 5), the number
of clusters *k* is swept from 2 to 10; the Silhouette
coefficient is maximised at k = 2 (0.371), while the Elbow Method's knee-detection identifies
k = 4 on the inertia curve. This disagreement between two standard heuristics is itself an
informative result — reported explicitly rather than hidden — and it motivates the multi-index
validation of the next section. GMM's component count is chosen by minimising the BIC; DBSCAN's
radius is read from the k-distance knee [3]; HDBSCAN is essentially parameter-light, needing only a
minimum cluster size; and Agglomerative and Spectral reuse the K-Means *k* for comparability. A
shared two-dimensional PCA projection (capturing roughly 76.6% of the variance in its first two
components) is used to visualise all six partitions on the same axes.

@fig outputs/figures/cluster_pca_projection.png | PCA projection of all six clustering solutions (PC1 + PC2 ≈ 76.6% of variance).

@fig outputs/figures/kmeans_selection.png | K-Means model selection: inertia (with detected elbow), silhouette, and Davies–Bouldin versus k.

@fig outputs/figures/gmm_bic.png | GMM model selection by BIC and AIC across component counts.

@fig outputs/figures/dbscan_kdistance.png | DBSCAN k-distance plot used to select the eps radius.

### 4.4 Cluster Validation and Segment Profiling
Each solution is scored on the three internal indices (Silhouette [11], Davies–Bouldin [12], and
Calinski–Harabasz [13]; Eqs. 7–9, excluding noise points for the density methods), and its
stability is assessed by 50 bootstrap resamples measuring the ARI [14] (Eq. 10) between the full-data partition and each
resampled partition. The transparent selection rule (Section 2.3) is then applied. The selected
segmentation is profiled in the original feature units — so that a marketing reader sees pounds and
days rather than z-scores — and each cluster is assigned a marketing-friendly name by a
deterministic rule tree expressed in multiples of the population mean. The heatmap and radar
figures below visualise the segment structure; the radar deliberately plots each segment's
*percentile* against the full customer population (rather than a min–max normalisation across only
a few cluster means, which would be degenerate) to give an honest picture of relative standing.

The profile table shows the three-part structure HDBSCAN discovers. Cluster 0, "General
Customers" (3,218 members, 54.75%), is the mainstream: moderate recency, frequency around 7, and
mean spend of about £2,422. Cluster 1, "Lost Customers" (1,546 members, 26.3%), is a
sharply-defined one-time-buyer group — frequency of exactly 1, tenure of 0, high recency (~363
days), and low spend (~£297). The noise group, Cluster −1 (1,114 members, 18.95%), is *not* a
failure but a genuine finding: these are the most valuable and atypical customers (mean spend
~£8,467, frequency ~12), too heterogeneous to force into a tidy segment, and correctly flagged for
individual treatment rather than a generic campaign.

@fig outputs/figures/segment_profiles.png | HDBSCAN segment-profile heatmap (z-score colour, with raw means annotated).

@fig outputs/figures/radar_profiles.png | HDBSCAN segment radar chart (each segment's percentile versus the full customer population).

@table outputs/tables/segment_profiles.csv | HDBSCAN segment profiles: mean features (un-scaled), size, and marketing name. | 30

### 4.5 Customer Lifetime Value
The BG/NBD [17] and Gamma–Gamma [16] models are fitted on the per-customer frequency / recency /
tenure / monetary summary. BG/NBD models the "flow" of a customer's future transactions and their
latent probability of still being "alive" (active), while Gamma–Gamma models the monetary value per
transaction, having first verified the near-zero correlation between transaction frequency and
value that the model assumes. The combined output gives, for each customer, a probability-alive (Eq. 12), an
expected number of purchases over 90-, 180-, and 365-day horizons (Eq. 13), an expected
transaction value (Eq. 14), and a discounted 12-month CLV (Eq. 15).
The results confirm the classic long-tailed value structure of retail: mean CLV is about £1,418 but
the median is only £371, with a small number of customers reaching very high values (the maximum
exceeds £320,000). The mean probability-alive is 0.909 (median 0.980), reflecting that most of the
retained base is still active. This value view is what allows the notification engine to protect
high-value customers preferentially.

@fig outputs/figures/clv_distribution.png | CLV distribution and the engagement-versus-retention scatter (colour = CLV).

@table outputs/tables/clv_summary.csv | Summary statistics of the CLV model outputs. | 30

### 4.6 Churn Classification
A customer is labelled *churned* if their Recency exceeds the 90th percentile of the Recency
distribution (Eq. 16) — a threshold of roughly 535 days — which yields 587 churned customers (10%), a
realistic and deliberately imbalanced positive class. Crucially, because the label is *defined* by
Recency, Recency is then **excluded** from the churn feature set: including it would be textbook
target leakage and would produce an artificially perfect classifier. Six classifiers are trained on
a stratified hold-out split with stratified cross-validation, and class imbalance is handled with
`class_weight="balanced"` (Eq. 17; logistic regression, random forest, decision tree) or
`scale_pos_weight` (XGBoost). Full confusion counts and McNemar tests are computed for every model. The feature
importances from the best tree model rank Tenure and Average Inter-Purchase Interval as the
strongest churn signals — an intuitive result, since customers who have not settled into a regular
buying rhythm are the most likely to lapse.

@fig outputs/figures/churn_feature_importance.png | Churn feature importances from the best tree model (Recency excluded to prevent leakage).

### 4.7 Notification Engine
The notification engine is the component that closes the loop from analysis to action. Each customer
is first assigned the marketing segment of their unsupervised HDBSCAN cluster, which sets a baseline
campaign from a segment playbook (an action, channel, offer, and base priority). This baseline is
then *modulated* by two independent signals: a CLV value tier (High / Medium / Low, from spend
quantiles) and a churn-risk band (High / Medium / Low, from the predicted churn probability). A
high churn risk escalates the priority and, for valuable customers, triggers a "Priority retention
intervention"; a low-value, high-risk customer instead receives a deliberately low-cost automated
reactivation, so that spend is concentrated where value is at stake. Contact timing is not a fixed
cadence but is derived from each customer's *own* average inter-purchase interval, which respects
their natural rhythm and avoids over-contact. The engine exposes a `recommend(customer_id)`
function for on-demand single-customer lookups, used directly by the dashboard.

### 4.8 ROI Simulation and Dashboard
Campaign ROI is estimated by a 10,000-iteration Monte Carlo simulation. Each campaign action has a
documented prior response rate; for every simulated run, response rates are drawn from Beta
distributions, conversions from the resulting Binomial, and per-response revenue is perturbed with
Gaussian noise, so that the output is a full *distribution* of profit and ROI rather than a single
fragile point estimate. The same machinery simulates a *static blanket-marketing baseline* — the
same message sent to everyone — so that the uplift attributable to targeting can be isolated. An
interactive Streamlit dashboard presents every artefact across eight pages (Overview, Segments,
Lifetime Value, Churn Risk, Migration, Notifications, ROI Simulation, and a live Customer Lookup),
turning the batch outputs into a tool a non-technical marketing user can operate. Dashboard
screenshots should be inserted here.

@table outputs/tables/notification_plan.csv | Sample of the per-customer notification plan (first 10 of 5,878 rows). | 10

## Chapter 5. Software Testing and Evaluation

### 5.1 Software Testing
The core logic is covered by an automated test suite of twenty unit tests that execute on small
synthetic datasets, independent of the raw data, so that they are fast, deterministic, and test
logic rather than data. The suite verifies the four cleaning rules, the RFM derivations (including
the distinct-invoice frequency count), skew detection and the `log1p` decision, the churn
target-leakage guard (asserting that Recency is absent from the churn feature matrix), the
notification decision logic (asserting correct priority escalation for high-risk high-value
customers), and the ROI simulator. All twenty tests pass. Reproducibility (NFR1) was confirmed
separately by observing byte-identical result tables across repeated end-to-end runs of the full
pipeline, a direct consequence of the fixed snapshot date and random seeds.

### 5.2 Evaluation of the Clustering
Validation by the three internal indices combined with bootstrap stability, aggregated into an
overall rank, places **HDBSCAN first**, followed by K-Means, Spectral, Agglomerative, GMM, and
DBSCAN. HDBSCAN achieved the highest Silhouette (0.416) and the best Davies–Bouldin (0.89)
precisely because it excludes genuinely unassignable noise rather than forcing every point into a
cluster — behaviour that is not merely a metric artefact but is *operationally correct* for
marketing, where atypical customers should be handled individually. The divergence between the
Elbow (k = 4) and Silhouette (k = 2) optima for K-Means, noted in Section 4.3, confirms that a
single selection heuristic is insufficient and vindicates the multi-index approach. Spectral
clustering was the most *stable* solution under bootstrapping (mean ARI near 0.99) but did not
improve geometric separation over K-Means, illustrating that stability and separation are distinct
qualities that must both be considered. DBSCAN failed outright on this data, collapsing to a single
cluster with near-zero stability, which is why the pre-declared rule's ARI threshold correctly
eliminates it. GMM over-fragmented into ten poorly-separated components (Silhouette 0.081).

@table outputs/tables/cluster_ranking.csv | Overall clustering-algorithm ranking across all validity and stability metrics. | 30

@table outputs/tables/cluster_validation.csv | Internal validity metrics for all six algorithms. | 30

@fig outputs/figures/stability_ari.png | Bootstrap stability (Adjusted Rand Index) across 50 resamples for each algorithm.

### 5.3 Evaluation of the Churn Models
With the leakage guard in place, the best model (Random Forest) achieved a ROC-AUC of about 0.849
— a credible, honest value, and importantly *not* the suspicious ~1.0 that leakage would produce,
which serves as evidence that the guard is working. On the overall multi-metric ranking (precision, recall, and F1; Eqs. 20–22) Random
Forest edges out XGBoost, but the McNemar test [18] (Eq. 23) shows that the difference between **Random
Forest and XGBoost is not statistically significant** (p ≈ 0.88): the two are statistically tied,
so a naive "the best model is Random Forest" claim would overstate the evidence. This is exactly
the kind of nuance the significance test is designed to expose. The test also reveals a more
consequential result: Gradient Boosting and KNN, despite competitive ROC-AUC values (0.846 and
0.807), collapse to near-zero recall (0.034 and 0.068) at the default decision threshold because
they were not given imbalance handling — they achieve a high AUC while in practice failing to
identify almost any churners. Random Forest and XGBoost, with imbalance handling, retain useful
recall (0.75 and 0.74). The lesson — that ROC-AUC alone can badly mislead under class imbalance and
must be read alongside recall and a significance test — is a substantive evaluation finding.

@table outputs/tables/churn_model_comparison.csv | Churn model comparison: hold-out and cross-validated metrics with confusion counts. | 30

@table outputs/tables/churn_model_ranking.csv | Overall churn-model ranking across metrics. | 30

@table outputs/tables/churn_mcnemar_pvalues.csv | Pairwise McNemar p-values between the churn models. | 30

@fig outputs/figures/churn_roc_curves.png | ROC curves for the six churn models.

@fig outputs/figures/churn_pr_curves.png | Precision–recall curves for the six churn models (more informative than ROC under imbalance).

### 5.4 Evaluation of CLV and Commercial ROI
The year-on-year segment migration analysis (FR10) provides a dynamic check on the segmentation:
"Champions" retain 61.7% of their members year on year, "General Customers" act as a gravity well
that both retains 45.3% and absorbs inflow from every other segment, and "Lost Customers" are
persistently lost (43.4% remain lost), confirming that the segments capture real, stable
behavioural tendencies rather than noise. The commercial evaluation is the project's headline
result. The CLV distribution is strongly right-skewed (median ≈ £371) with the top decile holding a
disproportionate share of total value, which is exactly why targeting matters. The Monte Carlo
simulation (Eqs. 24–27) estimates a mean campaign ROI of **38.4×** with a 95% credible interval of
[21.8×, 60.9×] and a 100% probability of positive ROI, versus **30.4×** for the static
blanket-marketing baseline — a profit uplift of **+168%** (a mean profit of about £23,952 against
the baseline's £8,927, from a campaign cost of only about £624). Reporting the full ROI
distribution rather than a point estimate, and expressing the headline as *uplift over a baseline*
rather than an absolute figure, makes the commercial claim both robust and defensible.

@fig outputs/figures/roi_distribution.png | Monte Carlo ROI and net-profit distributions, with the static-baseline overlay.

@fig outputs/figures/segment_migration.png | Year-on-year segment migration (transition rates between segments).

@table outputs/tables/roi_simulation_summary.csv | ROI simulation summary, including the uplift over the static baseline. | 30

### 5.5 Threats to Validity
Several limitations qualify the results and are stated honestly. The customer base (5,878) is
modest, and HDBSCAN's three-part solution is comparatively coarse; a larger or more diverse
customer base might support finer segmentation. The notification engine keys campaigns on the
cluster-derived RFM *segment names* rather than raw cluster indices — a deliberate interpretability
trade-off that makes the engine legible to a marketer at the cost of a slight abstraction from the
raw clusters. The ROI response priors are documented planning assumptions rather than rates fitted
from a live campaign, so the *absolute* ROI figures are indicative; this is the reason the analysis
emphasises the *uplift over the baseline*, which is far less sensitive to the shared prior and is
therefore the more reliable quantity to quote. Finally, both the CLV and churn models assume that
historical behavioural patterns persist into the forecast horizon, an assumption that would need
monitoring in a live deployment subject to seasonality or market shifts.

## Chapter 6. Conclusions and Future Work

### 6.1 Conclusions
This project delivered a reproducible software framework that segments retail customers from their
behavioural data, forecasts their lifetime value and churn risk, and converts the combined results
into an auditable, cluster-driven marketing plan whose financial value is quantified by simulation.
All six objectives (O1–O6) were met and evidenced: a literature-grounded justification of methods
(O1); an audited ETL and feature pipeline reducing 1.07M transactions to a clean 5,878-customer
feature set (O2); a six-algorithm clustering benchmark with principled selection and bootstrap
validation that selected HDBSCAN by a transparent rule (O3); a predictive layer combining BG/NBD +
Gamma–Gamma CLV with a six-model, leakage-guarded, significance-tested churn comparison (O4); an
auditable notification engine with an on-demand API and an eight-page dashboard (O5); and a Monte
Carlo ROI assessment demonstrating a +168% profit uplift over untargeted marketing (O6). The
principal contribution is the *closed loop* from raw transactions to a financially-evaluated,
auditable action plan, delivered with methodological rigour throughout — multi-family algorithm
comparison, leakage-controlled churn modelling, statistical significance testing, and
reproducibility by construction — that distinguishes it from prior work which typically stops at
the modelling stage.

### 6.2 Future Work
Several extensions would build on this foundation:

- **Larger-scale and GPU-accelerated clustering** (for example with RAPIDS cuML) to scale well
  beyond the current customer base, and exploration of deep representation learning to derive
  richer behavioural embeddings for segmentation.
- **Real-time deployment**, wrapping the on-demand `recommend()` logic behind a streaming or API
  service so that recommendations update continuously as customer behaviour changes, realising the
  full "dynamic CRM" vision.
- **Learned uplift modelling**, replacing the planning-assumption response priors with response
  rates estimated from live A/B campaigns or causal / uplift models [38], [39], which would turn
  the indicative ROI into a measured one.
- **Tighter cluster–notification coupling**, allowing campaigns to be driven directly from cluster
  structure where interpretability constraints permit.
- **Model-efficiency work**, distilling the churn ensemble into a single lightweight model for
  low-latency scoring in a production setting.

## References

> Entries marked [Verify] were sourced via literature search and should be checked against the
> original publication before final submission. Format follows IEEE.

[1] J. MacQueen, "Some methods for classification and analysis of multivariate observations," in *Proc. 5th Berkeley Symp. Math. Statist. Probab.*, 1967, pp. 281–297.

[2] S. P. Lloyd, "Least squares quantization in PCM," *IEEE Trans. Inf. Theory*, vol. 28, no. 2, pp. 129–137, 1982.

[3] M. Ester, H.-P. Kriegel, J. Sander, and X. Xu, "A density-based algorithm for discovering clusters in large spatial databases with noise," in *Proc. KDD*, 1996, pp. 226–231.

[4] E. Schubert, J. Sander, M. Ester, H.-P. Kriegel, and X. Xu, "DBSCAN revisited, revisited: why and how you should (still) use DBSCAN," *ACM Trans. Database Syst.*, vol. 42, no. 3, pp. 1–21, 2017.

[5] R. J. G. B. Campello, D. Moulavi, and J. Sander, "Density-based clustering based on hierarchical density estimates," in *Proc. PAKDD*, 2013, pp. 160–172.

[6] L. McInnes, J. Healy, and S. Astels, "hdbscan: Hierarchical density based clustering," *J. Open Source Softw.*, vol. 2, no. 11, p. 205, 2017.

[7] A. P. Dempster, N. M. Laird, and D. B. Rubin, "Maximum likelihood from incomplete data via the EM algorithm," *J. Roy. Statist. Soc. B*, vol. 39, no. 1, pp. 1–38, 1977.

[8] L. Breiman, "Random forests," *Mach. Learn.*, vol. 45, no. 1, pp. 5–32, 2001.

[9] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. KDD*, 2016, pp. 785–794.

[10] V. Satopää, J. Albrecht, D. Irwin, and B. Raghavan, "Finding a 'kneedle' in a haystack: Detecting knee points in system behavior," in *Proc. ICDCS Workshops*, 2011, pp. 166–171.

[11] P. J. Rousseeuw, "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis," *J. Comput. Appl. Math.*, vol. 20, pp. 53–65, 1987.

[12] D. L. Davies and D. W. Bouldin, "A cluster separation measure," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. PAMI-1, no. 2, pp. 224–227, 1979.

[13] T. Caliński and J. Harabasz, "A dendrite method for cluster analysis," *Commun. Statist.*, vol. 3, no. 1, pp. 1–27, 1974.

[14] L. Hubert and P. Arabie, "Comparing partitions," *J. Classification*, vol. 2, no. 1, pp. 193–218, 1985.

[15] D. Chen, S. L. Sain, and K. Guo, "Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining," *J. Database Marketing & Customer Strategy Manag.*, vol. 19, pp. 197–208, 2012.

[16] P. S. Fader, B. G. S. Hardie, and K. L. Lee, "RFM and CLV: Using iso-value curves for customer base analysis," *J. Marketing Res.*, vol. 42, no. 4, pp. 415–430, 2005.

[17] P. S. Fader, B. G. S. Hardie, and K. L. Lee, "'Counting your customers' the easy way: An alternative to the Pareto/NBD model," *Marketing Sci.*, vol. 24, no. 2, pp. 275–284, 2005.

[18] Q. McNemar, "Note on the sampling error of the difference between correlated proportions or percentages," *Psychometrika*, vol. 12, no. 2, pp. 153–157, 1947.

[19] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.

[20] "An exploration of clustering algorithms for customer segmentation in the UK retail market," *arXiv:2402.04103*, 2024. [Verify authors.]

[21] "Comparison of K-Means and DBSCAN algorithms for customer segmentation in e-commerce," *J. Digital Market and Digital Currency*, 2024. [Verify authors/volume.]

[22] "A novel hybrid deep learning framework for customer churn prediction using RFM and embedding clustering," *Scientific Reports*, 2026. [Verify authors/volume.]

[23] P. Chapman et al., *CRISP-DM 1.0: Step-by-Step Data Mining Guide*. SPSS Inc., 2000.

[24] "Leveraging machine learning algorithms in enterprise CRM architectures for personalized marketing automation," *J. Artificial Intelligence Research*, 2023. [Verify authors/year.]

[25] "RFM ranking — An effective approach to customer segmentation," *J. King Saud Univ. – Computer and Information Sciences*, 2018. [Verify authors/volume.]

[26] Y. Zhang, E. T. Bradlow, and D. S. Small, "Predicting customer value using clumpiness: From RFM to RFMC," *Marketing Science*, vol. 34, no. 2, pp. 195–208, 2015. [Verify.]

[27] "Customer segmentation: Automatic K-optimization and RFM-based K-means clustering," in *Proc. Int. Conf. Information and Intelligent Technologies (ICIIT)*, 2025. [Verify authors.]

[28] "Intelligent customer segmentation: Unveiling consumer patterns with machine learning," *J. Umm Al-Qura Univ. for Engineering and Architecture*, 2025. [Verify authors/volume.]

[29] "Intelligent vector-based customer segmentation in the banking industry," *arXiv:2012.11876*, 2020. [Verify authors.]

[30] "Application of BG/NBD and Gamma-Gamma models to predict customer lifetime value for a financial institution," in *Proc. IEEE Int. Conf.*, 2021. [Verify authors/venue.]

[31] "A data-driven approach to customer lifetime value prediction using probability and machine learning models," 2025. [Verify authors/venue.]

[32] "Application of machine learning techniques for churn prediction in the telecommunications industry," 2024. [Verify authors/venue.]

[33] "A comprehensive survey on customer churn analysis studies," *J. Information and Telecommunication*, 2025. [Verify authors/volume.]

[34] "Customer segmentation and churn prediction in online retail," in *Advances in Artificial Intelligence*, Springer, 2020. [Verify authors.]

[35] "A framework for customer segmentation to improve marketing strategies using machine learning," *Procedia Computer Science*, 2025. [Verify authors/volume.]

[36] "AI-driven predictive analytics for CRM to enhance retention, personalization and decision-making," *Int. J. Advanced Computer Science and Applications*, vol. 16, no. 4, 2025. [Verify authors.]

[37] Z. Liu, "e-Commerce personalized recommendation based on machine learning technology," *Mobile Information Systems*, 2022. [Verify authors.]

[38] "Dynamic marketing uplift modeling: A symmetry-preserving framework integrating causal forests with deep reinforcement learning," *Symmetry*, vol. 17, no. 4, art. 610, 2025. [Verify authors.]

[39] "Contextual multi-armed bandits for causal marketing," *arXiv:1810.01859*, 2018. [Verify authors.]

[40] "Customer segmentation for targeted marketing: Exploring DBSCAN and K-means," *Preprints.org*, 2025. [Verify authors.]
