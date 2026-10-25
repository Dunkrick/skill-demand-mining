"""
Job Market Skill-Demand Mining - PySpark pipeline
Course: Analytics and Systems of Big Data

Run (local):
    python job_skills_pipeline.py \
        --postings data/job_postings.csv \
        --skills   data/job_skills.csv \
        --out      out/ \
        --sample   0.1

Stages: load -> clean/normalize skills -> Parquet -> EDA -> FP-Growth rules
        -> KMeans role clusters -> role classifier -> skill-gap recommender demo.

If your CSV column names differ, edit the CONFIG block only.
"""
import argparse
import json
import os

import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import CountVectorizer, Normalizer, StringIndexer
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SparkSession, functions as F

# ----------------------------- CONFIG ---------------------------------------
JOB_KEY = "job_link"          # join key present in both files
SKILLS_COL = "job_skills"     # comma-separated skills string
TITLE_COL = "job_title"
LEVEL_COL = "job_level"       # optional
COUNTRY_COL = "search_country"  # optional

# raw skill string -> canonical skill (extend as you discover duplicates in EDA)
SYNONYMS = {
    "js": "javascript", "node": "node.js", "nodejs": "node.js", "reactjs": "react",
    "react.js": "react", "ml": "machine learning", "ai": "artificial intelligence",
    "postgres": "postgresql", "powerbi": "power bi", "ms excel": "excel",
    "microsoft excel": "excel", "k8s": "kubernetes", "gcp": "google cloud",
    "amazon web services": "aws", "c sharp": "c#", "golang": "go",
    "communications": "communication", "teamwork": "team collaboration",
}

# ordered rules: first match wins. Used as weak labels for the classifier.
ROLE_RULES = [
    ("data_ml", r"data scien|machine learning|\bml\b|\bai\b|data analy|analytics|data engineer|business intelligence"),
    ("devops_cloud", r"devops|site reliability|\bsre\b|cloud|infrastructure|platform engineer"),
    ("software_eng", r"software|developer|programmer|full[- ]?stack|front[- ]?end|back[- ]?end|web dev|qa engineer|sdet"),
    ("product_design", r"product manager|product owner|ux|ui designer|graphic design|product design"),
    ("healthcare", r"nurse|\brn\b|physician|clinical|medical|therap|pharmac|dental|caregiver"),
    ("sales_marketing", r"sales|marketing|account executive|business development|brand|seo"),
    ("finance_accounting", r"account(ant|ing)|finance|financial|auditor|payroll|bookkeep|analyst.*(credit|risk)"),
    ("ops_logistics", r"driver|warehouse|logistics|supply chain|operations|delivery|forklift"),
    ("hr_admin", r"recruit|human resources|\bhr\b|administrative|office manager|receptionist"),
]
# ----------------------------------------------------------------------------


def get_spark(app="JobSkillMining"):
    return (
        SparkSession.builder.appName(app)
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "4g"))
        .getOrCreate()
    )


def role_category_col(title_col):
    expr = F.lit("other")
    for name, pattern in reversed(ROLE_RULES):
        expr = F.when(F.lower(title_col).rlike(pattern), F.lit(name)).otherwise(expr)
    return expr


def load_and_clean(spark, postings_path, skills_path, sample, out):
    skills_raw = spark.read.csv(skills_path, header=True, escape='"')
    postings = spark.read.csv(postings_path, header=True, escape='"', multiLine=False)

    for opt in (LEVEL_COL, COUNTRY_COL):  # optional columns -> null if absent
        if opt not in postings.columns:
            postings = postings.withColumn(opt, F.lit(None).cast("string"))
    postings = postings.select(JOB_KEY, TITLE_COL, LEVEL_COL, COUNTRY_COL).dropDuplicates([JOB_KEY])
    skills_raw = skills_raw.select(JOB_KEY, SKILLS_COL).dropna()

    if sample < 1.0:
        skills_raw = skills_raw.sample(fraction=sample, seed=42)

    # explode -> normalize -> re-collect (robust under ANSI mode, no UDF)
    exploded = (
        skills_raw.select(JOB_KEY, F.explode(F.split(F.lower(F.col(SKILLS_COL)), ",")).alias("skill"))
        .withColumn("skill", F.trim(F.regexp_replace("skill", r"\s+", " ")))
        .filter((F.length("skill") >= 2) & (F.length("skill") <= 40))
    )
    syn = spark.createDataFrame(list(SYNONYMS.items()), ["raw", "canon"])
    exploded = (
        exploded.join(F.broadcast(syn), exploded.skill == syn.raw, "left")
        .withColumn("skill", F.coalesce("canon", "skill"))
        .select(JOB_KEY, "skill")
    )

    jobs = (
        exploded.groupBy(JOB_KEY).agg(F.array_sort(F.collect_set("skill")).alias("items"))
        .filter(F.size("items") >= 2)  # single-skill jobs carry no co-occurrence info
        .join(postings, JOB_KEY, "inner")
        .withColumn("role", role_category_col(F.col(TITLE_COL)))
    )
    jobs.write.mode("overwrite").parquet(os.path.join(out, "jobs_parquet"))
    return spark.read.parquet(os.path.join(out, "jobs_parquet"))  # re-read = columnar speed


def run_eda(jobs, out, metrics):
    n = jobs.count()
    metrics["n_jobs"] = n
    exploded = jobs.select(JOB_KEY, F.explode("items").alias("skill"))
    metrics["n_distinct_skills"] = exploded.select("skill").distinct().count()

    top = exploded.groupBy("skill").count().orderBy(F.desc("count")).limit(50).toPandas()
    top["pct_of_jobs"] = (top["count"] / n * 100).round(2)
    top.to_csv(os.path.join(out, "eda_top_skills.csv"), index=False)

    dist = jobs.select(F.size("items").alias("n_skills")).groupBy("n_skills").count().orderBy("n_skills")
    dist.toPandas().to_csv(os.path.join(out, "eda_skills_per_job.csv"), index=False)

    jobs.groupBy("role").count().orderBy(F.desc("count")).toPandas().to_csv(
        os.path.join(out, "eda_role_counts.csv"), index=False)
    jobs.groupBy(TITLE_COL).count().orderBy(F.desc("count")).limit(30).toPandas().to_csv(
        os.path.join(out, "eda_top_titles.csv"), index=False)
    if COUNTRY_COL in jobs.columns:
        jobs.groupBy(COUNTRY_COL).count().orderBy(F.desc("count")).limit(20).toPandas().to_csv(
            os.path.join(out, "eda_by_country.csv"), index=False)


def mine_rules(jobs, out, metrics, min_support, min_conf):
    fp = FPGrowth(itemsCol="items", minSupport=min_support, minConfidence=min_conf)
    model = fp.fit(jobs)
    freq = model.freqItemsets
    rules = model.associationRules
    metrics["n_frequent_itemsets"] = freq.count()
    metrics["n_rules"] = rules.count()
    metrics["min_support"], metrics["min_confidence"] = min_support, min_conf

    rules_pd = (
        rules.orderBy(F.desc("lift"), F.desc("confidence")).limit(5000).toPandas()
    )
    rules_pd["antecedent"] = rules_pd["antecedent"].apply(list)
    rules_pd["consequent"] = rules_pd["consequent"].apply(list)
    save = rules_pd.copy()
    save["antecedent"] = save["antecedent"].apply("|".join)
    save["consequent"] = save["consequent"].apply("|".join)
    save.to_csv(os.path.join(out, "rules_top.csv"), index=False)
    freq.orderBy(F.desc("freq")).limit(100).withColumn("items", F.concat_ws("|", "items")).toPandas().to_csv(
        os.path.join(out, "frequent_itemsets_top.csv"), index=False)
    return rules_pd


def cluster_roles(jobs, out, metrics, k, vocab):
    cv = CountVectorizer(inputCol="items", outputCol="tf", vocabSize=vocab, minDF=20, binary=True)
    cv_model = cv.fit(jobs)
    feats = cv_model.transform(jobs)
    feats = Normalizer(inputCol="tf", outputCol="features").transform(feats)

    km = KMeans(k=k, seed=42, featuresCol="features", predictionCol="cluster", maxIter=30)
    km_model = km.fit(feats)
    pred = km_model.transform(feats)
    metrics["kmeans_k"] = k
    metrics["kmeans_silhouette"] = round(ClusteringEvaluator(
        featuresCol="features", predictionCol="cluster").evaluate(pred), 4)

    vocab_list = cv_model.vocabulary
    profiles = {}
    for cid, center in enumerate(km_model.clusterCenters()):
        top_idx = center.argsort()[::-1][:10]
        profiles[str(cid)] = {"top_skills": [vocab_list[i] for i in top_idx]}
    titles = (
        pred.groupBy("cluster", TITLE_COL).count()
        .withColumn("rk", F.row_number().over(
            __import__("pyspark.sql.window", fromlist=["Window"]).Window.partitionBy("cluster").orderBy(F.desc("count"))))
        .filter("rk <= 5").collect()
    )
    for r in titles:
        profiles[str(r["cluster"])].setdefault("top_titles", []).append(r[TITLE_COL])
    sizes = {str(r["cluster"]): r["count"] for r in pred.groupBy("cluster").count().collect()}
    for cid, s in sizes.items():
        profiles[cid]["size"] = s
    with open(os.path.join(out, "cluster_profiles.json"), "w") as f:
        json.dump(profiles, f, indent=2)


def classify_roles(jobs, metrics, vocab):
    data = jobs.filter(F.col("role") != "other")
    train, test = data.randomSplit([0.8, 0.2], seed=42)
    cv = CountVectorizer(inputCol="items", outputCol="features", vocabSize=vocab, minDF=20, binary=True)
    idx = StringIndexer(inputCol="role", outputCol="label", handleInvalid="skip")

    results = {}
    for name, clf in {
        "logreg": LogisticRegression(maxIter=40, regParam=0.01),
        "random_forest": RandomForestClassifier(numTrees=60, maxDepth=12, seed=42),
    }.items():
        model = Pipeline(stages=[cv, idx, clf]).fit(train)
        pred = model.transform(test)
        acc = MulticlassClassificationEvaluator(metricName="accuracy").evaluate(pred)
        f1 = MulticlassClassificationEvaluator(metricName="f1").evaluate(pred)
        results[name] = {"accuracy": round(acc, 4), "weighted_f1": round(f1, 4)}
    metrics["classifier"] = results
    metrics["classifier_train_rows"] = train.count()
    metrics["classifier_test_rows"] = test.count()


def recommend(user_skills, rules_pd, top_n=10):
    """Skill-gap recommender: fire rules whose antecedent is covered by user's skills."""
    have = {s.strip().lower() for s in user_skills}
    scores = {}
    for _, r in rules_pd.iterrows():
        ant, con = set(r["antecedent"]), r["consequent"][0]
        if ant <= have and con not in have:
            s = r["confidence"] * r["lift"]
            if s > scores.get(con, (0, None))[0]:
                scores[con] = (s, f"{' + '.join(sorted(ant))} -> {con} (conf {r['confidence']:.2f}, lift {r['lift']:.1f})")
    ranked = sorted(scores.items(), key=lambda kv: -kv[1][0])[:top_n]
    return [{"skill": k, "why": v[1]} for k, v in ranked]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--postings", required=True)
    ap.add_argument("--skills", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--sample", type=float, default=0.1, help="fraction of jobs, 1.0 = all")
    ap.add_argument("--min-support", type=float, default=0.01)
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=300)
    ap.add_argument("--demo-skills", default="python")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")
    metrics = {}

    jobs = load_and_clean(spark, args.postings, args.skills, args.sample, args.out).cache()
    run_eda(jobs, args.out, metrics)
    rules_pd = mine_rules(jobs, args.out, metrics, args.min_support, args.min_conf)
    cluster_roles(jobs, args.out, metrics, args.k, args.vocab)
    classify_roles(jobs, metrics, args.vocab)

    demo = [s for s in args.demo_skills.split(",") if s]
    metrics["demo_recommendation"] = {"input": demo, "output": recommend(demo, rules_pd)}
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    spark.stop()


if __name__ == "__main__":
    main()
