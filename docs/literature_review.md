# Literature Review — Reading List

**Project:** Optimizing Targeted Marketing: A Framework for Customer Segmentation and
Automated Cluster-Based Notification (COMP5200M, University of Leeds).

> ⚠️ **Verify before citing.** This is a working bibliography, not finished references.
> For every entry, open the link (or search the title on **Google Scholar / the Leeds
> library**) and confirm the **authors, year, venue, and page numbers** before it goes
> into your reference list. The seminal algorithm papers are given from established
> knowledge; the applied papers were sourced via web search and the snippets can be
> imprecise. Manage them in Zotero/Mendeley.

This list has two parts:

- **Part 1 — Methodology references**, grouped by pipeline stage. Cite each where you
  *explain that technique* (Background / Methodology chapters).
- **Part 2 — Closest related work**: papers that tackle the *same combined task* as this
  project. Cite these in the **Related Work** chapter to position and compare your
  contribution.

Three papers are relevant to both roles; each is listed **once** (in Part 2) and
cross-referenced. Knowledge-distillation papers were intentionally excluded — they are
not relevant to this project's scope.

---

## Part 1 — Methodology references (by pipeline stage)

### A. Clustering algorithms (Stage 3)
1. **MacQueen, J. (1967).** "Some methods for classification and analysis of multivariate observations." *Proc. 5th Berkeley Symposium on Math. Statistics and Probability.* — origin of **K-Means**. *(seminal — Google Scholar)*
2. **Lloyd, S. P. (1982).** "Least squares quantization in PCM." *IEEE Trans. Information Theory, 28(2), 129–137.* — the standard K-Means algorithm. *(seminal — Google Scholar)*
3. **Ester, M., Kriegel, H.-P., Sander, J. & Xu, X. (1996).** "A density-based algorithm for discovering clusters in large spatial databases with noise." *KDD-96, 226–231.* — **DBSCAN**. *(seminal — Google Scholar)*
4. **Schubert, E., Sander, J., Ester, M., Kriegel, H.-P. & Xu, X. (2017).** "DBSCAN Revisited, Revisited: Why and How You Should (Still) Use DBSCAN." *ACM TODS, 42(3).* — modern eps-selection guidance. *(seminal — Google Scholar)*
5. **Campello, R., Moulavi, D. & Sander, J. (2013).** "Density-Based Clustering Based on Hierarchical Density Estimates." *PAKDD 2013.* — **HDBSCAN**. [ResearchGate](https://www.researchgate.net/publication/278700103_Density-Based_Clustering_Based_on_Hierarchical_Density_Estimates)
6. **McInnes, L., Healy, J. & Astels, S. (2017).** "hdbscan: Hierarchical density based clustering." *JOSS, 2(11), 205.* — the implementation used here. [JOSS PDF](https://www.theoj.org/joss-papers/joss.00205/10.21105.joss.00205.pdf)
7. **Dempster, A., Laird, N. & Rubin, D. (1977).** "Maximum Likelihood from Incomplete Data via the EM Algorithm." *JRSS-B, 39(1), 1–38.* — basis of **GMM**. *(seminal — Google Scholar)*
8. **Breiman, L. (2001).** "Random Forests." *Machine Learning, 45(1), 5–32.* — churn baseline. *(seminal — Google Scholar)*
9. **Chen, T. & Guestrin, C. (2016).** "XGBoost: A Scalable Tree Boosting System." *KDD '16, 785–794.* — gradient-boosting churn model. *(seminal — Google Scholar)*
10. **Satopää, V., Albrecht, J., Irwin, D. & Raghavan, B. (2011).** "Finding a 'Kneedle' in a Haystack: Detecting Knee Points in System Behavior." *ICDCS Workshops, 166–171.* — DBSCAN k-distance / elbow knee detection. *(seminal — Google Scholar)*

### B. Cluster validation & stability (Stage 3b)
11. **Rousseeuw, P. (1987).** "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis." *J. Computational and Applied Mathematics, 20, 53–65.* — **Silhouette**. *(seminal — Google Scholar)*
12. **Davies, D. & Bouldin, D. (1979).** "A Cluster Separation Measure." *IEEE TPAMI, PAMI-1(2), 224–227.* — **Davies–Bouldin index**. *(seminal — Google Scholar)*
13. **Caliński, T. & Harabasz, J. (1974).** "A dendrite method for cluster analysis." *Communications in Statistics, 3(1), 1–27.* — **Calinski–Harabasz index**. *(seminal — Google Scholar)*
14. **Hubert, L. & Arabie, P. (1985).** "Comparing partitions." *Journal of Classification, 2(1), 193–218.* — **Adjusted Rand Index** (bootstrap stability metric). *(seminal — Google Scholar)*
15. **(2025).** "The Silhouette coefficient and the Davies-Bouldin index are more informative than [other indices] for unsupervised clustering internal evaluation." *PeerJ Computer Science.* — justifies the metric choice. [PeerJ](https://peerj.com/articles/cs-3309/)
16. **"From A-to-Z Review of Clustering Validation Indices."** *arXiv:2407.20246.* — survey of 17 internal indices. [arXiv](https://arxiv.org/pdf/2407.20246)

### C. RFM & customer segmentation in retail (Stages 2–3)
17. **Chen, D., Sain, S. & Guo, K. (2012).** "Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining." *J. Database Marketing & Customer Strategy Mgmt, 19, 197–208.* — **origin paper of this project's dataset; cite prominently.** [Springer](https://link.springer.com/article/10.1057/dbm.2012.17)
18. **"An Exploration of Clustering Algorithms for Customer Segmentation in the UK Retail Market."** *arXiv:2402.04103.* — same UK online-retail data, multiple algorithms. [arXiv](https://arxiv.org/pdf/2402.04103)
19. **"Customer Segmentation: Automatic K-Optimization and RFM-Based K-Means Clustering."** *ICIIT 2025 (ACM).* [ACM](https://dl.acm.org/doi/full/10.1145/3731763.3731805)
20. **"Intelligent customer segmentation: unveiling consumer patterns with machine learning" (2025).** *J. Umm Al-Qura Univ. for Eng. & Arch.* [Springer](https://link.springer.com/article/10.1007/s43995-025-00180-7)
21. **"RFM ranking – An effective approach to customer segmentation."** *J. King Saud Univ. – Computer & Information Sciences.* [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1319157818304178)
22. **"Predicting Customer Value Using Clumpiness: From RFM to RFMC."** *Marketing Science (INFORMS).* — respected RFM extension. [INFORMS](https://pubsonline.informs.org/doi/10.1287/mksc.2014.0873)

### D. Algorithm comparison — K-Means vs DBSCAN vs HDBSCAN (Stages 3–4)
23. **"Comparison of K-Means and DBSCAN Algorithms for Customer Segmentation in E-commerce."** *J. Digital Market & Digital Currency.* [JDMDC](https://jdmdc.com/index.php/JDMDC/article/view/3)
24. **"Intelligent Vector-based Customer Segmentation in the Banking Industry."** *arXiv:2012.11876.* [arXiv](https://arxiv.org/pdf/2012.11876)

### E. Customer Lifetime Value (Stage 7)
25. **Fader, P., Hardie, B. & Lee, K. (2005).** "Counting Your Customers the Easy Way: An Alternative to the Pareto/NBD Model." *Marketing Science, 24(2), 275–284.* — **the BG/NBD model used here.** *(seminal — Google Scholar)*
26. **Fader, P., Hardie, B. & Lee, K. (2005).** "RFM and CLV: Using Iso-Value Curves for Customer Base Analysis." *J. Marketing Research, 42(4), 415–430.* — links RFM ↔ CLV (Gamma-Gamma context). *(seminal — Google Scholar)*
27. **"Application of BG/NBD and Gamma-Gamma Models to Predict Customer Lifetime Value for a Financial Institution."** *IEEE.* — applied use of this exact model pair. [IEEE Xplore](https://ieeexplore.ieee.org/document/9620535/)
28. **"A data-driven approach to customer lifetime value prediction using probability and machine learning models" (2025).** *ScienceDirect.* [link](https://www.sciencedirect.com/science/article/pii/S2772662225000578)

### F. Churn prediction (Stage 8)
29. **"Application of machine learning techniques for churn prediction in the telecom business."** *ScienceDirect.* [link](https://www.sciencedirect.com/science/article/pii/S2590123024014208)
30. **"A comprehensive survey on customer churn analysis studies" (2025).** *Taylor & Francis.* — anchors the churn section. [T&F](https://www.tandfonline.com/doi/full/10.1080/24751839.2025.2528440)
> See also the multi-task churn papers in **Part 2** (#37, #38, #45, #46).

### G. Personalized marketing, CRM & notification logic (Stage 10)
31. **"Leveraging Machine Learning Algorithms in Enterprise CRM Architectures for Personalized Marketing Automation."** *J. Artificial Intelligence Research.* — the "dynamic CRM" framing. [link](https://thesciencebrigade.com/JAIR/article/view/526)
32. **Liu (2022).** "e-Commerce Personalized Recommendation Based on Machine Learning Technology." *Mobile Information Systems (Wiley).* [Wiley](https://onlinelibrary.wiley.com/doi/10.1155/2022/1761579)

### H. Uplift modeling & marketing ROI (Stage 11 — your "vs static" differentiator)
33. **"Dynamic Marketing Uplift Modeling: A Symmetry-Preserving Framework Integrating Causal Forests with Deep RL."** *Symmetry (MDPI), 17(4), 610.* [MDPI](https://www.mdpi.com/2073-8994/17/4/610)
34. **"Contextual Multi-Armed Bandits for Causal Marketing."** *arXiv:1810.01859.* — causal/incremental targeting; supports "uplift vs static baseline". [arXiv](https://arxiv.org/pdf/1810.01859)

### I. Deep / representation learning for segmentation — *Future Work only*
*(Your project is classical ML; cite these only when discussing future directions.)*
35. **"Autoencoder-based General Purpose Representation Learning for Customer Embedding."** *arXiv:2402.18164.* [arXiv](https://arxiv.org/abs/2402.18164)
36. **"A comparative dimensionality reduction study in telecom customer segmentation using deep learning and PCA."** *J. Big Data (Springer).* [Springer](https://link.springer.com/article/10.1186/s40537-020-0286-0)

---

## Part 2 — Closest related work (whole-task comparison)

These tackle the *same combination of tasks* as this project (segmentation + churn/CLV +
targeting). Use them in **Related Work** to position and compare your contribution.

### Closest matches (multi-task, your domain)
37. **"Customer Segmentation and Churn Prediction in Online Retail."** *Advances in AI (Springer/ACM).* — segmentation + churn on the online-retail domain; likely your single closest analogue. [ACM](https://dl.acm.org/doi/10.1007/978-3-030-47358-7_33)
38. **"A novel hybrid deep learning framework for customer churn prediction using RFM and embedding clustering."** *Scientific Reports (Nature).* — RFM + clustering + churn in one framework. [Nature](https://www.nature.com/articles/s41598-026-53220-0)
39. **"A Framework for Customer Segmentation to Improve Marketing Strategies Using Machine Learning."** *Procedia CS (ScienceDirect).* — segmentation → marketing strategy. [link](https://www.sciencedirect.com/science/article/pii/S1877050925009846)
40. **"Customer Segmentation for Targeted Marketing: Exploring DBSCAN & K-Means."** *Preprints.org 2025.* — K-Means vs DBSCAN benchmark *for targeted marketing* (your exact objective). [Preprints](https://www.preprints.org/manuscript/202504.1434)
41. **"AI-Driven Predictive Analytics for CRM to Enhance Retention, Personalization and Decision-Making."** *IJACSA, 16(4).* — churn + personalization + CRM decisioning. [PDF](https://thesai.org/Downloads/Volume16No4/Paper_56-AI_Driven_Predictive_Analytics_for_CRM.pdf)

### Strongly related (large overlap with one or two stages)
42. **"Machine Learning-Driven Insights for Customer Segmentation and Hyper-Personalization in E-Commerce."** *ResearchGate.* [link](https://www.researchgate.net/publication/389887650_MACHINE_LEARNING-DRIVEN_INSIGHTS_FOR_CUSTOMER_SEGMENTATION_AND_HYPER-PERSONALIZATION_IN_E-COMMERCE)
43. **"Data-Driven Customer Segmentation: Advancing Precision Marketing through Analytics and ML."** *ResearchGate.* [link](https://www.researchgate.net/publication/384839238_Data-Driven_Customer_Segmentation_Advancing_Precision_Marketing_through_Analytics_and_Machine_Learning_Techniques)
44. **"Customer segmentation in digital marketing using a Q-learning differential evolution algorithm integrated with K-Means."** *PMC.* [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11805403/)
45. **"Leveraging AI for predictive customer churn modeling in telecommunications: a framework for enhanced CRM."** *Scientific Reports (Nature).* [Nature](https://www.nature.com/articles/s41598-025-30108-z)
46. **"Enhancing customer retention with ML: comparative analysis of ensemble models for churn prediction."** *ScienceDirect.* — RF/XGBoost (your exact churn models). [link](https://www.sciencedirect.com/science/article/pii/S2667096825000138)
47. **"An autonomous mixed-data oversampling method for AIoT-based churn recognition and personalized recommendations using behavioral segmentation."** *PMC.* — segmentation + churn + recommendation together. [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10773761/)

> **To chase down:** several sources describe an **"RFMOC / RFMOCD"** model that extends RFM
> to do segmentation **+ CLV + churn (logistic regression) + targeted-strategy ranking** —
> essentially this project's exact task combination. Search *"RFMOCD customer segmentation
> churn lifetime value"* on Google Scholar; if it checks out it may be your strongest single
> comparison paper.

---

## Part 3 — Positioning your contribution

Most of the closest related work stops at **segmentation + churn (sometimes + CLV)**. Few
papers continue the chain into an **auditable, cluster-driven notification/campaign engine**
*and* a **Monte Carlo ROI simulation that quantifies uplift over an untargeted (static)
baseline**. That end-to-end chain is the defensible novelty of this project.

Suggested framing for the Related Work conclusion:

> *"Prior work [37–41] establishes RFM-based clustering and churn/CLV modelling on retail
> data, but typically stops short of (a) converting validated segments into a transparent,
> rule-based campaign-action engine and (b) quantifying the ROI uplift of that targeting over
> untargeted marketing. This project addresses both, closing the loop from raw transactions
> to an auditable, financially-evaluated marketing plan."*
