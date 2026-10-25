#!/usr/bin/env bash
# Phase 1: Environment Setup, Kaggle Download & Schema Check
set -euo pipefail

echo "=== 1. Checking Java Version ==="
java -version

echo "=== 2. Installing Python Requirements ==="
pip install -r requirements.txt
pip install "pandas<3" pyarrow kaggle

echo "=== 3. Downloading Kaggle Dataset (if not already in data/) ==="
mkdir -p data
if [ ! -f "data/job_skills.csv" ]; then
    echo "Downloading asaniczka/1-3m-linkedin-jobs-and-skills-2024..."
    kaggle datasets download -d asaniczka/1-3m-linkedin-jobs-and-skills-2024 -f job_skills.csv -p data/ --unzip
    kaggle datasets download -d asaniczka/1-3m-linkedin-jobs-and-skills-2024 -f linkedin_job_postings.csv -p data/ --unzip
    # Rename or symlink linkedin_job_postings.csv -> job_postings.csv so CLI commands work unchanged
    if [ -f "data/linkedin_job_postings.csv" ] && [ ! -f "data/job_postings.csv" ]; then
        ln -s linkedin_job_postings.csv data/job_postings.csv
        echo "Linked data/linkedin_job_postings.csv -> data/job_postings.csv"
    fi
fi

echo "=== 4. Schema & 5-Row Preview: data/job_skills.csv ==="
head -n 6 data/job_skills.csv

echo "=== 5. Schema & 5-Row Preview: data/job_postings.csv ==="
head -n 6 data/job_postings.csv

echo "=== 6. Running Smoke Test (--sample 0.01) ==="
python3 job_skills_pipeline.py \
    --postings data/job_postings.csv \
    --skills   data/job_skills.csv \
    --out      out_smoke/ \
    --sample   0.01

echo "=== Smoke Test Complete! Outputs in out_smoke/ ==="
ls -lh out_smoke/