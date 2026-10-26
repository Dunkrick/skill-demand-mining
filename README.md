# Job Market Skill-Demand Mining with PySpark

> **Course:** Analytics and Systems of Big Data (ASBD) - Sem 7  
> **Institution:** IIITDM Kancheepuram  
> **Contributors:** Rithwick, Rahul, Naresh, Sachin  
> **Repository:** [Dunkrick/skill-demand-mining](https://github.com/Dunkrick/skill-demand-mining.git)

---

## Overview

Job postings contain high-dimensional insights into labor market demands, but skill requirements are frequently buried within unstructured, semi-structured text. This project implements a distributed Big Data mining and machine learning pipeline built on **Apache Spark (PySpark)** over **1.3M+ LinkedIn job postings**.

By leveraging Spark SQL, DataFrames, and Spark MLlib, the pipeline:

- Ingests and normalizes raw job-posting text using broadcast synonym mappings and stoplists.
- Extracts co-occurring technical and business skills via distributed **FP-Growth** association rule mining.
- Profiles occupational archetypes using **PCA dimensionality reduction** and **K-Means clustering**.
- Evaluates **Logistic Regression**, **Random Forest**, and **Bernoulli Naive Bayes** to predict role categories from skill incidence vectors.
- Powers a rule-based **skill-gap recommender engine** that recommends next-step skills based on rule confidence and lift.

---

## Objectives

- **[O1] Distributed Ingestion & Normalization:** Ingest raw CSV data, normalize skill entities, filter noise tokens, and persist clean data to columnar Apache Parquet storage.
- **[O2] Frequent Skillset & Association Mining:** Apply distributed FP-Growth to extract co-occurring skill patterns ranked by Support, Confidence, and Lift.
- **[O3] Role Profiling & Clustering:** Vectorize multi-label skill baskets, apply PCA dimensionality reduction, and group roles using K-Means clustering evaluated via Silhouette score.
- **[O4] Multi-Class Role Classification:** Benchmark baseline classifiers (Logistic Regression, Random Forest, Naive Bayes) against a majority-class baseline to predict role categories from skills.
- **[O5] Skill-Gap Recommender Engine:** Recommend high-affinity next skills for candidate skill sets using antecedent rule matching.
- **[O6] Systems Scalability:** Measure execution runtime and output stability across dataset scales from 1% to 100% in local mode.

---

## System Architecture

The pipeline processes data through six modular stages designed for shuffle minimization and predicate pushdown:

1. **Distributed Ingestion & Cleaning:** Reads CSVs with quote-escaping, explodes skill tokens, joins with a broadcast synonym dictionary, filters out non-skill noise (benefits, physical requirements, generic soft skills), applies frequency thresholds, and derives weak role labels.
2. **Columnar Parquet Storage:** Writes Snappy-compressed columnar partitions to avoid redundant parsing across downstream stages.
3. **Exploratory Data Analysis (EDA):** Aggregates and exports skill frequencies, role distributions, and geographical counts to CSV.
4. **Frequent Pattern Mining:** Constructs prefix-trees with FP-Growth to identify itemsets and association rules ranked by lift and confidence.
5. **Clustering & Classification:** Vectorizes token incidence, reduces dimensionality via PCA to 20 components, clusters roles with K-Means, and trains supervised role classifiers.
6. **Skill-Gap Recommendation:** Matches candidate skills against antecedent rule combinations and ranks recommendations by $\text{Confidence} \times \text{Lift}$.

For detailed architectural flowcharts, stage breakdowns, and distributed systems rationale, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Dataset

- **Primary Source:** Kaggle — [1.3M LinkedIn Jobs & Skills (2024)](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024) (`asaniczka`)
  - `job_skills.csv`: `job_link`, `job_skills` (comma-delimited text).
  - `job_postings.csv`: `job_link`, `job_title`, `job_level`, `search_country`, company, location.
  - `job_summary.csv`: Full posting descriptions (_excluded to maximize ingestion throughput_).
- **Scope & Characteristics:**
  - Reflects a single scrape window snapshot; analyses emphasize role and regional patterns rather than multi-year temporal trends.
  - Role labels are weak labels generated through regex keyword rules against job titles rather than manual ground truth annotations.
  - Spans cross-industry sectors (healthcare, sales, software, operations); domain filtering (`--tech-only`) isolates software, data/ML, and DevOps subsets.

---

## Installation & Setup

### Prerequisites

- **Python:** 3.9+ (tested on Python 3.10 to 3.13)
- **Java:** OpenJDK 11, 17, or 21 (required by Apache Spark)

### Setup Steps

```bash
# Clone the repository
git clone https://github.com/Dunkrick/skill-demand-mining.git
cd skill-demand-mining

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Pipeline Execution

### 1. Fast Smoke Test (1% Sample)

```bash
bash scripts/phase1_setup_and_smoke.sh
```

### 2. Tech-Only Focus Run (Clean Career Trajectory Rules)

```bash
python job_skills_pipeline.py \
    --postings data/job_postings.csv \
    --skills data/job_skills.csv \
    --out out_tech/ \
    --sample 0.5 \
    --tech-only \
    --min-support 0.02 \
    --demo-skills "python,sql"
```

### 3. Full Dataset Execution (1.28M Jobs in Local Mode)

```bash
python job_skills_pipeline.py \
    --postings data/job_postings.csv \
    --skills data/job_skills.csv \
    --out out_scale_1.0/ \
    --sample 1.0 \
    --min-support 0.01 \
    --demo-skills "python,sql"
```

### CLI Parameters Reference

| Argument           | Default        | Description                                                       |
| :----------------- | :------------- | :---------------------------------------------------------------- |
| `--postings`       | _Required_     | Path to `job_postings.csv`                                        |
| `--skills`         | _Required_     | Path to `job_skills.csv`                                          |
| `--out`            | `out`          | Directory for writing Parquet, CSVs, and JSON metrics             |
| `--sample`         | `0.1`          | Fraction of dataset to process (`1.0` for full dataset)           |
| `--min-support`    | `0.01`         | Minimum support threshold for FP-Growth itemsets                  |
| `--min-conf`       | `0.3`          | Minimum confidence threshold for association rules                |
| `--k`              | `8`            | Number of clusters for K-Means                                    |
| `--vocab`          | `300`          | Vocabulary size for CountVectorizer feature extraction            |
| `--pca`            | `20`           | PCA dimensionality reduction components before KMeans (`0` = off) |
| `--min-skill-freq` | `20`           | Drop rare skills appearing in fewer than $N$ job postings         |
| `--keep-noise`     | `False`        | Keep benefits (PTO, 401k), physical reqs, and generic soft skills |
| `--tech-only`      | `False`        | Restrict dataset to `software_eng`, `data_ml`, and `devops_cloud` |
| `--demo-skills`    | `"python,sql"` | Comma-separated skills to evaluate in the recommender engine      |

---

## Scalability & Empirical Benchmark Summary

Evaluated across **1.28M+ real-world LinkedIn postings** in local mode (4GB heap, 16 shuffle partitions):

| Stage                       | 1% Sample (12k jobs) | 10% Sample (127k jobs) | 50% Sample (642k jobs) | 100% Full (1.28M jobs) | Scaling Behavior                                        |
| :-------------------------- | :------------------: | :--------------------: | :--------------------: | :--------------------: | :------------------------------------------------------ |
| **Ingest, Clean & Parquet** |         9.1s         |         14.1s          |         22.3s          |         36.9s          | Sub-linear ($4.0\times$ time for $100\times$ data)      |
| **EDA & Summary Stats**     |         0.9s         |          1.9s          |          3.2s          |          4.0s          | Near-instant columnar projection                        |
| **FP-Growth Mining**        |         1.2s         |          4.8s          |         17.4s          |         31.7s          | Memory-efficient FP-Tree construction                   |
| **K-Means + PCA**           |         4.2s         |          6.6s          |         11.9s          |         20.9s          | Bounded feature dimensionality (`vocab=300, pca=20`)    |
| **Classification**          |         5.0s         |         12.5s          |         37.1s          |         63.7s          | Sub-linear overall (12.7x time for 106x data)           |
| **Total Runtime**           |      **20.4s**       |       **39.9s**        |       **91.9s**        |       **157.2s**       | **$100\times$ data in only $7.7\times$ execution time** |

For detailed comparative evaluations, noise reduction impact, silhouette improvements, and classifier metrics, see [`benchmarks/BENCHMARKS.md`](benchmarks/BENCHMARKS.md) and [`benchmarks/scalability_summary.json`](benchmarks/scalability_summary.json).

---

## Output Artifacts & Demonstration

Execution outputs are exported to the specified `--out` directory:

1. **`rules_top.csv`**: Association rules ranked by Lift and Confidence.
2. **`cluster_profiles.json`**: Role clusters detailing cluster sizes, characteristic skills (salience scored), and top titles.
3. **`metrics.json`**: Execution summary including stage timings, dataset counts, silhouette evaluation, classifier metrics (accuracy, weighted F1, weighted precision, weighted recall, majority baseline), and demo recommendation output.

### Recommender Demo Example

Given input skills `["python", "sql"]` evaluated on tech-domain rules (`out_tech/`):

```json
{
  "input": ["python", "sql"],
  "output": [
    {
      "skill": "java",
      "why": "python + sql -> java (conf 0.47, lift 1.9)"
    },
    {
      "skill": "aws",
      "why": "python + sql -> aws (conf 0.42, lift 1.9)"
    }
  ]
}
```

---

## Repository Structure

```text
.
├── PROJECT_PLAN.md          # Project roadmap, syllabus alignment, and presentation outline
├── README.md                # Main documentation and execution guide
├── requirements.txt         # Core dependencies (PySpark, PyArrow, Pandas)
├── job_skills_pipeline.py   # Complete end-to-end PySpark pipeline script
├── docs/
│   └── ARCHITECTURE.md      # Detailed system architecture and algorithmic rationale
├── benchmarks/
│   ├── BENCHMARKS.md        # Comprehensive multi-scale benchmark report
│   └── scalability_summary.json # Machine-readable scalability and model metrics
├── scripts/
│   ├── phase1_setup_and_smoke.sh  # Automated setup and 1% smoke test
│   └── run_experiments.sh         # Batch suite (Tech-Only, All-Industries, Scalability)
├── data/                    # Dataset directory (untracked / gitignored)
└── out*/                    # Experiment output directories (gitignored)
```

---

## Project Roadmap

The current repository reflects the complete Mid-Semester prototype. Planned final-semester extensions include:

- Interactive Streamlit dashboard for real-time skill-gap exploration.
- Multi-snapshot temporal trend analysis to capture skill demand shifts.
- Spark Structured Streaming simulator for real-time job posting streams.
- Cluster deployment benchmarks on multi-node infrastructure.
- Hand-annotated ground-truth validation set for classifier calibration.

See [`PROJECT_PLAN.md`](PROJECT_PLAN.md) for detailed course milestones and evaluation notes.

---

## Contributors & Course Details

- **Course:** Analytics and Systems of Big Data (ASBD)
- **Contributors:** Rithwick, Rahul, Naresh, Sachin
- **Institution:** IIITDM Kancheepuram
- **Milestone:** Mid-Semester Evaluation (October 2026)
