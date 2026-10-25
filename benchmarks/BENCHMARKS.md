# Pipeline Empirical Benchmarks & Systems Evaluation

This document details the experimental findings, systems scalability metrics, and algorithm comparison results produced by the PySpark skill mining pipeline across multiple dataset scales and domain configurations.

---

## 1. Systems Scalability: 1% to 100% (1.28M Jobs)

The complete pipeline was evaluated across three dataset fractions on local standalone Spark (4GB heap, 16 shuffle partitions):
- **1% sample:** 12,092 jobs
- **50% sample:** 641,956 jobs
- **100% full dataset:** 1,283,940 jobs

### ⏱️ Wall-Clock Execution Timing per Stage (Seconds)

| Stage | 1% Sample (12k jobs) | 50% Sample (642k jobs) | 100% Full (1.28M jobs) | Scaling Behavior |
| :--- | :---: | :---: | :---: | :--- |
| **Ingest, Clean & Parquet Write** | 9.1s | 22.3s | 36.9s | Sub-linear ($4.0\times$ time for $100\times$ data) |
| **EDA & Distribution Stats** | 0.9s | 3.2s | 4.0s | Near-instant due to columnar Parquet projection |
| **FP-Growth Association Mining** | 1.2s | 17.4s | 31.7s | Highly efficient prefix-tree construction |
| **K-Means Clustering + PCA** | 4.2s | 11.9s | 20.9s | Bounded feature dimensionality (`vocab=300, pca=20`) |
| **Multi-Class Classification** | 5.0s | 37.1s | 63.7s | Linear with respect to row count |
| **Total End-to-End Runtime** | **20.4s** | **91.9s** | **157.2s** | **$100\times$ data scale in only $7.7\times$ execution time** |

> **Key Systems Takeaway:** Ingesting 1.28 million semi-structured records, extracting vocabulary, building frequent itemsets, performing PCA, clustering into 8 profiles, and training two classifiers completes in under **2 minutes and 38 seconds**.

---

## 2. Data Cleaning & Representation Impact

A major milestone between Phase 1 and Phase 2 was identifying that raw LinkedIn postings contain non-skill noise tokens (benefits, physical requirements, generic diplomas, and ubiquitous soft skills).

| Metric | v1 Noisy Baseline (10% sample) | v2 Cleaned Pipeline (100% Full Dataset) | Impact & Rationale |
| :--- | :---: | :---: | :--- |
| **Distinct Skill Tokens** | 448,244 | 82,818 | Pruned 400k+ rare typos and non-skill strings via `--min-skill-freq 20` |
| **Noise Stoplist** | Inactive (`--keep-noise`) | Active (`NOISE` stoplist) | Removed benefits (dental, PTO, 401k) and physical tags (lifting, standing) |
| **Clustering Silhouette** | **0.0791** | **0.4217** | **$+433\%$ improvement** via PCA projection ($k=20$) prior to K-Means |
| **Top Cluster Profiles** | Dominated by "communication", "401k" | Distinctive role skills (e.g. AWS, Kubernetes, React, Python) | Meaningful skill salience scoring ($share^2 / global\_frequency$) |

---

## 3. Tech-Domain Specialization vs. All-Industries

Postings across all industries are heavily weighted towards healthcare (240k jobs), sales (73k), and generic roles. When analyzing tech roles, applying `--tech-only` isolates software, data/ML, and DevOps postings:

| Pipeline Setting | Dataset Scope | Analyzed Jobs | Association Rules | Recommender Output for `["python", "sql"]` |
| :--- | :--- | :---: | :---: | :--- |
| **All-Industries** (`--keep-noise=false`) | All 10 role categories | 1,283,940 | 31 | No trigger at support=0.01 (tech pairs diluted by non-tech) |
| **Tech-Only** (`--tech-only`) | Software, Data/ML, DevOps | 18,725 | 1,641 | **1. `java`** (conf: 0.47, lift: 1.9)<br>**2. `aws`** (conf: 0.42, lift: 1.9) |

---

## 4. Machine Learning Classification Performance

Predicting weak-labeled role categories across held-out test splits:

| Setting | Classifier Model | Accuracy | Weighted F1 | Test Rows |
| :--- | :--- | :---: | :---: | :---: |
| **Full Dataset (100%)** | Logistic Regression | **83.57%** | **0.8215** | 88,602 |
| **Full Dataset (100%)** | Random Forest (`trees=80, depth=20`) | 77.89% | 0.7431 | 88,602 |
| **Tech Domain (50%)** | Logistic Regression | **84.74%** | **0.8415** | 3,677 |
| **Tech Domain (50%)** | Random Forest (`trees=80, depth=20`) | 82.70% | 0.8081 | 3,677 |
