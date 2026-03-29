"""Print CV improvement recommendations from skill gaps (CLI only — no hardcoded CV)."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from sentence_transformers import SentenceTransformer

from config import DEFAULT_THRESHOLD, DEFAULT_TOP_N, MODEL_NAME
from cv_parser import extract_cv_text
from matching import find_missing_skills, load_jobs_dataframe, rank_jobs_by_cv

LEARNING_RESOURCES: dict[str, str] = {
    "Docker": "Learn Docker → docker.com/get-started (containers & images basics)",
    "Kubernetes": "Learn Kubernetes → kubernetes.io/docs/tutorials/kubernetes-basics",
    "Airflow": "Learn Apache Airflow → airflow.apache.org/docs (DAGs & scheduling)",
    "dbt": "Learn dbt → courses.getdbt.com (free fundamentals course)",
    "Spark": "Learn Apache Spark → spark.apache.org/docs/latest (PySpark quickstart)",
    "Kafka": "Learn Kafka → kafka.apache.org/quickstart",
    "Snowflake": "Learn Snowflake → quickstarts.snowflake.com",
    "Terraform": "Learn Terraform → developer.hashicorp.com/terraform/tutorials",
    "MLflow": "Learn MLflow → mlflow.org/docs/latest/tutorials-and-examples",
    "Power BI": "Learn Power BI → learn.microsoft.com/en-us/training/powerplatform/power-bi",
    "Tableau": "Learn Tableau → help.tableau.com (Free Creator trial available)",
    "AWS SageMaker": "Learn SageMaker → docs.aws.amazon.com/sagemaker (Studio lab is free)",
    "A/B Testing": "Learn A/B Testing → 'Trustworthy Online Controlled Experiments' (book)",
    "Deep Learning": "Learn Deep Learning → fast.ai (Practical Deep Learning, free)",
    "TensorFlow": "Learn TensorFlow → tensorflow.org/tutorials",
    "PyTorch": "Learn PyTorch → pytorch.org/tutorials",
    "HuggingFace": "Learn HuggingFace → huggingface.co/learn/nlp-course",
    "Data Governance": "Learn Data Governance → DAMA DMBOK (book) or Collibra University",
    "CI/CD": "Learn CI/CD → GitHub Actions quickstart or GitLab CI/CD docs",
    "Feature Engineering": "Learn Feature Engineering → 'Feature Engineering for ML' (book by Alice Zheng)",
    "Statistics": "Learn Statistics → Khan Academy Statistics or StatQuest (YouTube)",
    "Data Modeling": "Learn Data Modeling → 'The Data Warehouse Toolkit' by Kimball (book)",
    "R": "Learn R → r4ds.had.co.nz (R for Data Science, free online)",
    "Collibra": "Learn Collibra → university.collibra.com (free courses available)",
    "GDPR": "Learn GDPR basics → gdpr.eu/what-is-gdpr",
}

CV_SECTION_TIPS: dict[str, str] = {
    "Docker": "Add a bullet under Projects: 'Containerized pipelines using Docker'",
    "Airflow": "Add to Experience: 'Orchestrated workflows with Apache Airflow DAGs'",
    "dbt": "Add to Skills section and mention any dbt models you've built",
    "Kubernetes": "If used at work, add to cloud/infrastructure bullet points",
    "Snowflake": "Add to Skills and mention schema design or cost optimization work",
    "A/B Testing": "Add to Experience: 'Designed and analyzed A/B experiments for [feature]'",
    "Deep Learning": "Add a Projects section with any personal DL project (Kaggle counts)",
    "Data Governance": "Add to Experience if relevant: 'Established data quality checks and ownership'",
    "CI/CD": "Add to Experience: 'Set up CI/CD for ML pipelines using GitHub Actions'",
}


def run_cli(cv_pdf: str, *, jobs_csv: Path | None = None, top_n: int = DEFAULT_TOP_N) -> None:
    cv_text = extract_cv_text(cv_pdf)
    print(f"Loading model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    df = load_jobs_dataframe(jobs_csv)
    top_jobs = rank_jobs_by_cv(df, cv_text, model, top_n)

    skill_gap: dict[str, list[str]] = {}
    for _, row in top_jobs.iterrows():
        missing = find_missing_skills(cv_text, row["skills_list"], model, DEFAULT_THRESHOLD)
        for skill in missing:
            skill_gap.setdefault(skill, []).append(row["title"])

    skill_frequency = Counter({skill: len(jobs) for skill, jobs in skill_gap.items()})

    print("\n" + "=" * 60)
    print("  CV IMPROVEMENT RECOMMENDATIONS")
    print("=" * 60)

    if not skill_frequency:
        print("\n✓ No major skill gaps found across top job matches.")
    else:
        print(
            f"\nFound {len(skill_frequency)} missing skills across your top {len(top_jobs)} job matches.\n"
        )
        for skill, count in skill_frequency.most_common():
            jobs_needing_it = ", ".join(skill_gap[skill])
            print(f"▸ {skill}  (required by {count} job{'s' if count > 1 else ''}: {jobs_needing_it})")
            resource = LEARNING_RESOURCES.get(skill)
            if resource:
                print(f"  Learn:   {resource}")
            cv_tip = CV_SECTION_TIPS.get(skill)
            if cv_tip:
                print(f"  CV tip:  {cv_tip}")
            print()

    print("-" * 60)
    print("PRIORITY ACTION PLAN:")
    for i, (skill, count) in enumerate(skill_frequency.most_common(3), 1):
        print(
            f"  {i}. Add '{skill}' — unlocks {count} more job match{'es' if count > 1 else ''}"
        )
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CV improvement tips from job skill gaps.")
    parser.add_argument("cv_pdf", help="Path to CV PDF.")
    parser.add_argument("--jobs", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    args = parser.parse_args()
    run_cli(args.cv_pdf, jobs_csv=args.jobs, top_n=args.top_n)
