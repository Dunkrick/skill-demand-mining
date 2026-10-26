#!/usr/bin/env bash
# Run benchmark experimental suites:
#  1. Tech-only domain pipeline (demonstrating clean high-lift rules)
#  2. All-industries pipeline (broad view)
#  3. Scalability suite across sample sizes (1%, 10%, 50%, 100%)
set -euo pipefail

cd "$(dirname "$0")/.."

POSTINGS="data/job_postings.csv"
SKILLS="data/job_skills.csv"

echo "=== 1. Tech-Only Focus Run (sample=0.5) ==="
python3 job_skills_pipeline.py \
    --postings "$POSTINGS" \
    --skills "$SKILLS" \
    --out out_tech/ \
    --sample 0.5 \
    --tech-only \
    --min-support 0.02 \
    --demo-skills "python,sql"

echo "=== 2. All-Industries Run (sample=0.1) ==="
python3 job_skills_pipeline.py \
    --postings "$POSTINGS" \
    --skills "$SKILLS" \
    --out out_all/ \
    --sample 0.1 \
    --demo-skills "python,sql"

echo "=== 3. Scalability Suite ==="
for S in 0.01 0.1 0.5 1.0; do
    echo "Running scale test at sample=${S}..."
    python3 job_skills_pipeline.py \
        --postings "$POSTINGS" \
        --skills "$SKILLS" \
        --out "out_scale_${S}/" \
        --sample "$S" \
        --demo-skills "python,sql"
done

echo "=== All benchmark runs completed successfully! ==="
