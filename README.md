# Job Market Skill-Demand Mining with PySpark

> **Course:** Analytics and Systems of Big Data (ASBD) — Semester 7  
> **Milestone:** Mid-Semester Evaluation & Prototype  
> **Repository:** [Dunkrick/ASBD-Project](https://github.com/Dunkrick/ASBD-Project.git)

---

## 📌 1. Project Overview & Motivation

Online job postings contain rich insights into the modern labor market, but skill requirements are typically buried in free-form, unstructured text. Job seekers, students, and curriculum designers struggle to identify:

- Which technical and soft skills actually co-occur in industry demands.
- How distinct job roles overlap in required skill profiles.
- What specific skills bridge the gap from a candidate's current profile to target roles.

This project implements a scalable, distributed Big Data mining and machine learning pipeline using **Apache Spark (PySpark)** over **1.3M+ LinkedIn job postings**. By leveraging Spark SQL, DataFrames, and Spark MLlib, the pipeline processes large-scale semi-structured posting data to discover association patterns, cluster roles, classify job categories, and power an intelligent skill-gap recommender.

---

## 🎯 2. Objectives (Mid-Semester Scope)

- **[O1] Distributed Ingestion & Normalization Pipeline:** Ingest raw CSV data, normalize and canonicalize skill entities, eliminate single-skill noise, and persist intermediate data to columnar Parquet storage.
- **[O2] Frequent Skillset & Association Rule Mining:** Apply distributed **FP-Growth** to extract co-occurring skill patterns, ranked by **Support**, **Confidence**, and **Lift**.
- **[O3] Role Skill Profiling & Clustering:** Vectorize multi-label skill sets and group job titles using **K-Means clustering** evaluated via Silhouette score.
- **[O4] Multi-Class Role Classification:** Benchmark baseline classifiers (**Logistic Regression vs. Random Forest**) to predict high-level role categories from skill profiles.
- **[O5] Skill-Gap Recommender Engine:** Implement a prototype recommendation algorithm that identifies next-step skills to learn based on candidate input skills and high-confidence association rules.

---

## 🏗️ 3. System Architecture & Pipeline Flow

The pipeline is designed with Big Data systems principles to optimize memory usage, avoid expensive shuffles, and support scalable execution.

```
+-------------------------------------------------------------+
|                     Raw Datasets (CSV)                      |
|         job_postings.csv    +    job_skills.csv             |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|        Stage 1: Distributed Ingestion & Preprocessing       |
|  - Spark CSV Reader with escaping                           |
|  - Explode comma-separated skill strings                    |
|  - Whitespace trimming & length filtering [2, 40] chars     |
|  - Broadcast Join with Synonym Table (avoids full shuffle)  |
|  - GroupBy + collect_set() (deduplication & uniqueness)     |
|  - Filter jobs with < 2 skills (pruning zero-cooccurrence)  |
|  - Weak-label role tagging via regex rule engine            |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|        Stage 2: Optimized Storage (Apache Parquet)          |
|  - Snappy compressed, columnar partition storage            |
|  - Eliminates repeated CSV parsing for downstream analytics |
+-------------------------------------------------------------+
                              |
       +----------------------+----------------------+
       |                      |                      |
       v                      v                      v
+--------------+      +--------------+      +------------------+
| Stage 3: EDA |      | Stage 4: FPM |      | Stage 5: MLlib   |
| - Top skills |      | - FP-Growth  |      | - CountVectorize |
| - Role dist  |      | - Freq Sets  |      | - Normalizer(L2) |
| - Job counts |      | - Rules by   |      | - K-Means Clust. |
| - Geographies|      |   Lift/Conf  |      | - LR vs RF Clf   |
+--------------+      +--------------+      +------------------+
       |                      |                      |
       +----------------------+----------------------+
                              |
                              v
+-------------------------------------------------------------+
|          Stage 6: Skill-Gap Recommender & Artifacts         |
|  - Antecedent rule matching: {skills} -> recommendation     |
|  - Structured outputs: CSVs, cluster_profiles.json, metrics |
+-------------------------------------------------------------+
```

---

## 💡 4. Big Data Systems & Algorithmic Choices

| Component                 | Technical Choice               | Systems & Algorithmic Rationale                                                                                                                                                        |
| :------------------------ | :----------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Storage Format**        | Apache Parquet                 | Columnar, compressed format supporting pushdown predicates and selective column projection; speeds up downstream tasks by orders of magnitude over raw CSV.                            |
| **Deduplication**         | Spark `collect_set`            | FP-Growth expects unique transaction items per basket. Deduplicates synonyms and repeated postings natively within partition aggregations.                                             |
| **Synonym Normalization** | Broadcast Join (`F.broadcast`) | Synonym mapping is small. Broadcasting avoids expensive shuffle hash joins across cluster worker nodes.                                                                                |
| **Pattern Mining**        | FP-Growth (over Apriori)       | FP-Growth constructs a compact FP-Tree in two dataset passes without explicit candidate generation, scaling to millions of transactions where Apriori suffers combinatorial explosion. |
| **Feature Extraction**    | Binary `CountVectorizer`       | Fixed vocabulary size (`vocabSize=300`, `minDF=20`) to bound feature space dimensionality, filter rare token noise, and create binary transaction incidence vectors.                   |
| **Vector Normalization**  | L2 `Normalizer`                | Projects vectors onto a unit hypersphere, making standard Euclidean K-Means behave effectively like Cosine Similarity for sparse text data.                                            |
| **Classification**        | Spark ML Pipeline              | Evaluates linear decision boundaries (Logistic Regression) against tree-based ensembles (Random Forest) across an 80/20 train/test split.                                              |

---

## 📊 5. Dataset Information

- **Primary Source:** Kaggle — [1.3M LinkedIn Jobs & Skills (2024)](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024) (asaniczka)
  - `job_skills.csv`: Contains `job_link` and comma-delimited `job_skills`.
  - `job_postings.csv`: Contains metadata including `job_link`, `job_title`, `job_level`, `search_country`, `company_name`, etc.
  - `job_summary.csv`: Full job description text (_excluded to optimize ingestion throughput and avoid unnecessary overhead_).
- **Fallback Source:** Kaggle — [LinkedIn Job Postings (2023-2024)](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) (arshkon).

### Dataset Considerations & Limits

- **Snapshot Nature:** Data reflects a single scrape collection window; longitudinal trend analysis is limited, so spatial (country) and hierarchy (seniority level) comparisons are highlighted.
- **Weak Labels for Classification:** High-level role categories (`data_ml`, `software_eng`, `devops_cloud`, `healthcare`, etc.) are synthesized using keyword regex rules against job titles rather than manual ground truth labels.
- **Cross-Domain Scope:** Postings span diverse sectors (tech, sales, logistics, healthcare). For domain-specific investigations, subset filtering is easily applied.

---

## 📁 6. Repository Structure

```text
.
├── PROJECT_PLAN.md          # Comprehensive roadmap, course alignment & timeline
├── README.md                # Project documentation & execution guide
├── requirements.txt         # Core dependencies (PySpark, PyArrow, Pandas)
├── job_skills_pipeline.py   # Complete end-to-end PySpark pipeline script
├── data/                    # Dataset directory (untracked / gitignored)
│   ├── job_postings.csv
│   └── job_skills.csv
└── out/                     # Generated outputs & evaluation artifacts
    ├── jobs_parquet/        # Cleaned, columnar dataset
    ├── eda_top_skills.csv
    ├── eda_skills_per_job.csv
    ├── eda_role_counts.csv
    ├── eda_top_titles.csv
    ├── eda_by_country.csv
    ├── rules_top.csv
    ├── frequent_itemsets_top.csv
    ├── cluster_profiles.json
    └── metrics.json
```

---

## 🚀 7. Installation & Setup

### Prerequisites

- **Python:** 3.9+ (3.10 / 3.11 recommended)
- **Java:** Java 11, 17, or 21 (required by Apache Spark)
  ```bash
  java -version
  ```

### Step 1: Clone the Repository

```bash
git clone https://github.com/Dunkrick/ASBD-Project.git
cd ASBD-Project
```

### Step 2: Environment Setup

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 💻 8. Running the Pipeline

The pipeline provides configurable CLI flags for memory control, dataset sampling, and hyperparameter tuning.

### 1. Smoke Test (Fast Verification on 1% Sample)

```bash
python job_skills_pipeline.py \
    --postings data/job_postings.csv \
    --skills data/job_skills.csv \
    --out out/ \
    --sample 0.01
```

### 2. Mid-Semester Benchmark Run (10% Sample with Demo Query)

```bash
python job_skills_pipeline.py \
    --postings data/job_postings.csv \
    --skills data/job_skills.csv \
    --out out/ \
    --sample 0.1 \
    --min-support 0.01 \
    --min-conf 0.3 \
    --k 8 \
    --demo-skills "python,sql"
```

### 3. Full Dataset Execution (Production / Cluster Run)

```bash
export SPARK_DRIVER_MEMORY=8g
python job_skills_pipeline.py \
    --postings data/job_postings.csv \
    --skills data/job_skills.csv \
    --out out/ \
    --sample 1.0 \
    --min-support 0.005 \
    --min-conf 0.25
```

### CLI Parameters Reference

| Argument        | Default    | Description                                             |
| :-------------- | :--------- | :------------------------------------------------------ |
| `--postings`    | _Required_ | Path to `job_postings.csv`                              |
| `--skills`      | _Required_ | Path to `job_skills.csv`                                |
| `--out`         | `out`      | Directory for writing Parquet, CSVs, and JSON metrics   |
| `--sample`      | `0.1`      | Fraction of dataset to process (`1.0` for full dataset) |
| `--min-support` | `0.01`     | Minimum support threshold for FP-Growth itemsets        |
| `--min-conf`    | `0.3`      | Minimum confidence threshold for association rules      |
| `--k`           | `8`        | Number of clusters for K-Means                          |
| `--vocab`       | `300`      | Vocabulary size for CountVectorizer feature extraction  |
| `--demo-skills` | `"python"` | Comma-separated skills to test the recommender engine   |

---

## 📈 9. Output Artifacts & Demonstration

Upon completion, all metrics and artifacts are exported to the `--out` folder:

1. **`rules_top.csv`**: Association rules ranked by Lift and Confidence (e.g., `{docker, kubernetes} -> {aws}`).
2. **`cluster_profiles.json`**: Distinct role clusters containing cluster size, top 10 characteristic skills, and top frequent job titles.
3. **`metrics.json`**: Comprehensive execution summary:
   - Total processed jobs and distinct skill counts.
   - Frequent itemset count and generated association rule count.
   - K-Means silhouette evaluation score.
   - Classifier performance (Accuracy and Weighted F1 for Logistic Regression and Random Forest).
   - Recommender output response for `--demo-skills`.

### Recommender Demo Example

Given input skills `["python", "sql"]`, the engine evaluates all active rules where the antecedent is satisfied:

```json
{
  "input": ["python", "sql"],
  "output": [
    {
      "skill": "spark",
      "why": "python + sql -> spark (conf 0.42, lift 3.8)"
    },
    {
      "skill": "tableau",
      "why": "python + sql -> tableau (conf 0.38, lift 2.9)"
    }
  ]
}
```

---

## 🔮 10. Mid-Sem Progress vs. Final-Semester Scope

```
[ Mid-Semester (Current) ]                  [ Final-Semester Plan ]
├── Data Ingestion & Broadcast Cleaning      ├── Interactive Streamlit Dashboard
├── Parquet Storage Migration                ├── Multi-snapshot Temporal Trend Analysis
├── EDA & Distribution Reporting             ├── Spark Structured Streaming Simulator
├── FP-Growth Rule Mining                    ├── Distributed Cluster Scaling Benchmarks
├── K-Means Skill Clustering                 ├── Hand-Labeled Ground-Truth Validation Set
└── Rule-Based Skill Gap Prototype           └── Deployed REST API for Recommendations
```

---

## 👥 Contributors & Course Details

- **Course:** Analytics and Systems of Big Data (ASBD)
- **Institution:** IIITDM Kancheepuram
- **Current Milestone:** Mid-Semester Review (October 2026)
