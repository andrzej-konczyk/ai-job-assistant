"""Skill vocabulary, keyword extraction, semantic skill-gap detection, job ranking."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer, util

from config import DEFAULT_THRESHOLD, DEFAULT_TOP_N, JOBS_CSV, MODEL_NAME
from cv_parser import extract_cv_text

SKILLS_VOCAB: list[str] = [
    "Python", "SQL", "R", "Scala", "Java", "Bash",
    "Spark", "Kafka", "Airflow", "dbt", "Hadoop", "Flink",
    "Snowflake", "BigQuery", "Redshift", "PostgreSQL", "MySQL", "MongoDB",
    "Tableau", "Power BI", "Looker", "Metabase", "Grafana",
    "TensorFlow", "PyTorch", "Scikit-learn", "XGBoost", "Keras",
    "HuggingFace", "spaCy", "NLTK", "OpenCV",
    "Docker", "Kubernetes", "Terraform", "Ansible", "CI/CD",
    "AWS", "GCP", "Azure", "AWS SageMaker", "MLflow", "Kubeflow",
    "Excel", "Statistics", "A/B Testing", "Machine Learning", "Deep Learning",
    "NLP", "Computer Vision", "Feature Engineering", "Data Modeling",
    "Data Governance", "GDPR", "ETL", "ELT", "Data Warehousing",
    "DAX", "Collibra", "Git", "Agile", "Stakeholder Management",
    "Time Series", "Financial Modeling", "Research",
]


def extract_skills_from_text(text: str) -> list[str]:
    """Return skills from SKILLS_VOCAB found (case-insensitive) in text."""
    text_lower = text.lower()
    found: list[str] = []
    for skill in SKILLS_VOCAB:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, text_lower):
            found.append(skill)
    return found


def find_missing_skills(
    cv_text: str,
    job_skills: list[str],
    model: SentenceTransformer,
    threshold: float,
) -> list[str]:
    """
    For each required job skill, check if anything in the CV is semantically close.
    If not → missing.
    """
    cv_sentences = [s.strip() for s in re.split(r"[.\n]", cv_text) if len(s.strip()) > 5]
    if not cv_sentences:
        return list(job_skills)
    cv_embeddings = model.encode(cv_sentences, convert_to_tensor=True)
    skill_embeddings = model.encode(job_skills, convert_to_tensor=True)
    similarities = util.cos_sim(skill_embeddings, cv_embeddings)
    max_scores = similarities.max(dim=1).values.cpu().numpy()
    return [skill for skill, score in zip(job_skills, max_scores) if score < threshold]


def load_jobs_dataframe(csv_path: Path | None = None) -> pd.DataFrame:
    path = Path(csv_path) if csv_path is not None else JOBS_CSV
    if not path.is_file():
        raise FileNotFoundError(
            f"Jobs file not found: {path}. Add jobs.csv next to the app or set --jobs."
        )
    df = pd.read_csv(path)
    df["skills_list"] = df["required_skills"].apply(
        lambda s: [skill.strip() for skill in str(s).split(",")]
    )
    df["full_text"] = (
        df["title"].astype(str)
        + ". "
        + df["description"].astype(str)
        + " Skills: "
        + df["required_skills"].astype(str)
    )
    return df


def rank_jobs_by_cv(
    df: pd.DataFrame,
    cv_text: str,
    model: SentenceTransformer,
    top_n: int,
) -> pd.DataFrame:
    cv_embedding = model.encode(cv_text, convert_to_tensor=True)
    job_embeddings = model.encode(df["full_text"].tolist(), convert_to_tensor=True)
    out = df.copy()
    out["similarity"] = util.cos_sim(cv_embedding, job_embeddings)[0].cpu().numpy()
    return out.sort_values("similarity", ascending=False).head(top_n).reset_index(drop=True)


def run_cli(cv_path: str, *, jobs_csv: Path | None = None, top_n: int = DEFAULT_TOP_N) -> None:
    cv_text = extract_cv_text(cv_path)
    print(f"Loading model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    df = load_jobs_dataframe(jobs_csv)
    top_jobs = rank_jobs_by_cv(df, cv_text, model, top_n)

    print(f"\nTop {top_n} job matches with skill gap analysis:\n")
    print("=" * 60)
    for _, row in top_jobs.iterrows():
        missing = find_missing_skills(cv_text, row["skills_list"], model, DEFAULT_THRESHOLD)
        print(f"\nJob:  {row['title']}  (similarity: {row['similarity']:.2f})")
        if missing:
            print("Missing skills:")
            for skill in missing:
                print(f"  - {skill}")
        else:
            print("  ✓ All required skills matched")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match CV to jobs and list skill gaps.")
    parser.add_argument("cv_pdf", help="Path to CV PDF (no default).")
    parser.add_argument("--jobs", type=Path, default=None, help="Jobs CSV (default: bundled jobs.csv).")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    args = parser.parse_args()
    run_cli(args.cv_pdf, jobs_csv=args.jobs, top_n=args.top_n)
