# Job Market Skill-Demand Mining with Spark
**Course:** Analytics and Systems of Big Data | **Stage:** Mid-semester demo

---

## 1. Problem statement
Job postings list skills as free text. Students and job-seekers cannot easily tell which skills go together, which roles really overlap, or what to learn next. This project builds a Spark pipeline over 1M+ LinkedIn job postings to:

1. **Discover skill co-occurrence rules** (FP-Growth association rule mining), e.g. `{python, sql} -> {spark}`.
2. **Cluster roles by skill profile**, so job titles with similar skill needs group together (K-Means).
3. **Predict role category from skills** (Logistic Regression and Random Forest).
4. **Recommend skill gaps**: given a user's skills, suggest the next skills to learn, ranked by rule confidence x lift.

## 2. Objectives (put these on a slide)
- O1: Build a scalable ingest -> clean -> Parquet pipeline in PySpark.
- O2: Mine frequent skill sets and association rules; report support, confidence, lift.
- O3: Find role clusters and describe each by top skills and titles.
- O4: Train and compare two classifiers; report accuracy and weighted F1.
- O5: Deliver a skill-gap recommender demo.

## 3. Dataset
**Primary:** Kaggle, *"1.3M LinkedIn Jobs & Skills (2024)"* (author: asaniczka).
- `job_skills.csv`: `job_link`, `job_skills` (comma-separated string)
- `job_postings.csv`: `job_link`, `job_title`, `job_level`, `search_country`, company, location, etc.
- `job_summary.csv`: long descriptions (**not needed**, skip it)

**Fallback:** Kaggle *"LinkedIn Job Postings (2023-2024)"* (author: arshkon). Same idea, different schema, so edit the CONFIG block in the script.

> **Schema note:** Column names have been verified against the real Kaggle dataset (`asaniczka/1-3m-linkedin-jobs-and-skills-2024`) and confirmed in the pipeline CONFIG block.

**Known limits (say these in the presentation, it shows maturity):**
- The data is a snapshot (mostly one scrape window), so skill-trend-over-time is weak. Compare by country and job level instead.
- It covers all industries, not only tech. We use that on purpose: role categories include healthcare, sales, etc. For a tech-only view, filter `role in (data_ml, software_eng, devops_cloud)`.
- Role labels are weak labels derived from title keywords (regex rules), not human-annotated. State this in the evaluation.
- No salary column, so no salary prediction.

## 4. Architecture
```
Kaggle CSVs
   |
   v
[Spark read] -- job_skills.csv + job_postings.csv, join on job_link
   |
   v
[Clean/normalize] -- lowercase, trim, split, explode, synonym map (js->javascript),
   |                  length filter, collect_set per job, drop jobs with <2 skills
   v
[Parquet store] -- columnar, compressed, re-read for all downstream stages
   |
   +--> [EDA] top skills, skills/job, roles, titles, countries -> CSVs
   +--> [FP-Growth] frequent itemsets + association rules -> rules_top.csv
   +--> [CountVectorizer -> Normalizer -> KMeans] role clusters -> cluster_profiles.json
   +--> [CountVectorizer -> LR / RF] role classifier -> metrics.json
   |
   v
[Recommender] rules -> "what to learn next" (demo)
```

## 5. Method details
| Stage | Choice | Why |
|---|---|---|
| Storage | Parquet | Columnar and compressed; far faster than re-parsing CSV |
| Dedup per job | `collect_set` | FP-Growth requires unique items per transaction |
| Drop <2 skills | filter | Single-skill rows carry no co-occurrence signal |
| FP-Growth | `minSupport=0.01`, `minConfidence=0.3` | Tune: lower support = more rules but slower. Report the trade-off |
| Rule ranking | lift, then confidence | Lift > 1 means the pair co-occurs more than chance; ignores generically popular skills |
| Vectorization | binary CountVectorizer, vocab 300, minDF 20 | Bounded dimensionality, drops rare noise |
| Clustering | L2-normalize + KMeans, k=8 | Normalizing makes Euclidean close to cosine; evaluate with silhouette; try k = 5, 8, 12 |
| Classifier | LR vs Random Forest | Simple baseline vs non-linear model |
| Split | 80/20, seed 42 | Reproducible |

**Why FP-Growth, not Apriori:** FP-Growth builds a compact prefix tree and avoids candidate generation, so it scales to millions of transactions. Apriori re-scans the data per itemset size.

## 6. Evaluation
- **Rules:** count of itemsets and rules, top rules by lift, a sanity check that results make domain sense (e.g. `{aws, docker} -> {kubernetes}`).
- **Clustering:** silhouette score, plus manual check that top skills and titles per cluster are coherent.
- **Classifier:** accuracy and weighted F1 on the held-out 20%.
  - **Honest caveat:** labels come from title regex and skills are strongly tied to titles, so scores may look very high. Say so. A good talking point is the label-leakage question and how you would validate it (hand-label 200 rows).
- **Scalability:** run with `--sample 0.01`, `0.1`, `1.0` and record runtime per stage in a table. This is your "systems" evidence.

## 7. Mid-sem presentation (8 slides)
1. **Title + team**
2. **Problem and motivation**: why skill co-occurrence matters
3. **Objectives** (O1-O5)
4. **Dataset**: source, size, schema, limits
5. **Architecture diagram** (section 4)
6. **Pipeline and preprocessing**: cleaning steps with before/after example (`"JS, Python "` -> `["javascript","python"]`)
7. **Preliminary results**: top skills chart, skills-per-job histogram, first 10 rules, one cluster profile
8. **Remaining work and timeline**: classifier tuning, dashboard, scalability benchmark, final report

**Mid-sem minimum:** slides 1-7 with real numbers from the script. Everything else is final-sem scope.

## 8. Timeline (tight)
| When | Do |
|---|---|
| **Day 0 (today)** | Install Java + PySpark, download dataset, verify column names, run script with `--sample 0.01` |
| **Day 1** | Run `--sample 0.1`; inspect EDA; extend `SYNONYMS` with duplicates found in top skills; tune minSupport; collect screenshots/tables |
| **Day 2** | Build slides from `out/` files; rehearse; write 1-page report |
| **Day 3** | Buffer + optional Streamlit demo for the recommender |

## 9. Risks and fallbacks
| Risk | Fix |
|---|---|
| Out of memory | `--sample 0.05`, set `SPARK_DRIVER_MEMORY=6g`, raise `--min-support` to 0.02 |
| Column names differ | Edit CONFIG block only |
| FP-Growth too slow | Raise `--min-support`; sample down |
| `toPandas` errors | `pip install "pandas<3" pyarrow` |
| Rules full of generic soft skills ("communication") | Rank by lift; optionally add a stoplist of soft skills |
| Classifier accuracy suspiciously high | Explain weak labels (section 6); present as pipeline validation, not a headline claim |
| Kaggle download blocked | Use the fallback dataset |

## 10. Interview talking points
- Why Parquet over CSV; why `collect_set`; why FP-Growth over Apriori.
- What support, confidence and lift mean, with a concrete example from your own rules.
- How you chose `minSupport` and what happened to runtime and rule count when you changed it.
- Why normalize before KMeans.
- Weak-label caveat and how you would fix it.
- Broadcast join for the synonym table (small table, avoids a shuffle).
- How this would scale: more partitions, a cluster instead of `local[*]`, Delta/Parquet partitioning by country.

## 11. Final-sem extensions (do not start these now)
Streamlit dashboard, skill trend analysis with a second snapshot, Spark Structured Streaming simulation of new postings, deployed recommender API, hand-labeled validation set.

## 12. Run it
```bash
pip install -r requirements.txt
python job_skills_pipeline.py --postings data/job_postings.csv --skills data/job_skills.csv \
    --out out/ --sample 0.01          # smoke test
python job_skills_pipeline.py --postings data/job_postings.csv --skills data/job_skills.csv \
    --out out/ --sample 0.1 --demo-skills "python,sql"
```
**Outputs in `out/`:** `jobs_parquet/`, `eda_*.csv`, `rules_top.csv`, `frequent_itemsets_top.csv`, `cluster_profiles.json`, `metrics.json` (includes the demo recommendation).

*Status: The pipeline has been validated and run end-to-end on the real Kaggle data across 1% to 100% of the dataset (1.28M postings), with empirical scaling benchmarks and metrics recorded in `benchmarks/`.*
