# Pipeline Architecture and Systems Design

This document details the end-to-end architecture, distributed execution flow, and algorithmic design choices implemented in the PySpark job skill-demand mining pipeline.

---

## 1. End-to-End System Flow

The pipeline ingests raw, semi-structured job postings and skill text, executes distributed normalization and filtering, writes intermediate columnar Parquet partitions, and branches into parallel analytical and machine learning stages.

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
| - Geographies|      |   Lift/Conf  |      | - LR/RF/NB Clf   |
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

## 2. Systems and Algorithmic Design Choices

| Component | Technical Choice | Rationale |
| :--- | :--- | :--- |
| **Storage Format** | Apache Parquet (Snappy) | Columnar, compressed format supporting pushdown predicates and selective column projection; reduces disk I/O and speeds up subsequent analytical queries. |
| **Deduplication** | Spark `collect_set` | FP-Growth requires distinct transaction items per basket. Deduplicates synonyms and repeated postings natively within partition aggregations. |
| **Synonym Normalization** | Broadcast Hash Join (`F.broadcast`) | The canonical synonym mapping dictionary is small (~60 entries). Broadcasting sends the lookup table to all executors, avoiding expensive shuffle hash joins across cluster nodes. |
| **Pattern Mining** | Distributed FP-Growth | FP-Growth constructs a compact prefix tree in two data passes without explicit candidate generation, scaling to millions of transactions where Apriori suffers combinatorial explosion. |
| **Feature Extraction** | Binary `CountVectorizer` | Fixed vocabulary size (`vocabSize=300`, `minDF=20`) bounds the feature dimensionality, filters rare token noise, and outputs binary incidence vectors for clustering and classification. |
| **Vector Normalization** | L2 `Normalizer` | Projects sparse occurrence vectors onto a unit hypersphere, allowing Euclidean distance in K-Means to behave equivalently to Cosine Similarity. |
| **Clustering** | PCA + K-Means | PCA dimensionality reduction (to 20 components) alleviates sparsity and curtails the curse of dimensionality prior to K-Means partition assignment. |
| **Supervised Classification** | Spark ML Pipeline | Evaluates linear boundaries (Logistic Regression) against tree-based ensembles (Random Forest) and generative probabilistic baselines (Bernoulli Naive Bayes) across an 80/20 train/test split. |

---

## 3. Detailed Pipeline Stages

### Stage 1: Ingestion and Normalization
1. Reads `job_postings.csv` and `job_skills.csv` using Spark's CSV parser with quote escaping enabled.
2. Extracts and trims individual skill strings by splitting comma-separated text and exploding into individual rows.
3. Maps raw synonyms to canonical identifiers (e.g., `js` -> `javascript`, `postgres` -> `postgresql`) via a broadcast DataFrame join.
4. Drops noise tokens (benefits, physical requirements, generic credentials, and ubiquitous soft skills) when `--keep-noise` is false.
5. Filters out rare long-tail tokens using frequency thresholding (`--min-skill-freq`).
6. Re-aggregates distinct skill tokens per job posting using `collect_set`, discarding postings with fewer than two skills.
7. Assigns weak role labels using title-matching regular expressions (`ROLE_RULES`) across 9 broad categories.

### Stage 2: Parquet Persistence
- Persists intermediate clean data to columnar Parquet format on disk.
- Downstream stages read exclusively from this Parquet store, ensuring that data cleaning, tokenization, and joins execute exactly once.

### Stage 3: Exploratory Data Analysis (EDA)
- Computes distribution summaries: top skills overall, skills-per-job statistics, role category distributions, and job counts by geography.
- Exports summaries to CSV files for tabular and visual inspection.

### Stage 4: Frequent Pattern Mining (FP-Growth)
- Runs PySpark MLlib's distributed `FPGrowth` model over the tokenized baskets.
- Extracts frequent itemsets above `--min-support` (default 0.01).
- Generates association rules filtered by `--min-conf` (default 0.3) and computes Support, Confidence, and Lift metrics.

### Stage 5: Unsupervised Clustering & Supervised Classification
- **Clustering**: Fits a binary `CountVectorizer`, normalizes via L2 `Normalizer`, projects to 20 dimensions via `PCA`, and clusters records into $K$ profiles via `KMeans`. Computes cluster Silhouette scores and identifies top salient skills per cluster using frequency-ratio scoring ($share^2 / global\_freq$).
- **Classification**: Trains Logistic Regression, Random Forest, and Bernoulli Naive Bayes models to predict role categories from skill incidence vectors. Computes majority-class test baseline, accuracy, weighted F1, weighted precision, and weighted recall.

### Stage 6: Skill-Gap Recommender
- Evaluates candidate input skills against active association rules where the antecedent is a subset of input skills and the consequent is unobserved.
- Ranks candidate recommendations by Rule Score ($\text{Confidence} \times \text{Lift}$).
