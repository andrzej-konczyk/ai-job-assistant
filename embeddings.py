"""Embed CV vs job descriptions and rank matches (CLI or importable helpers)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer, util

from config import DEFAULT_TOP_N, JOBS_CSV, MODEL_NAME
from cv_parser import extract_cv_text
from matching import load_jobs_dataframe


def rank_jobs_embedding(
    cv_text: str,
    model: SentenceTransformer,
    df: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:
    cv_embedding = model.encode(cv_text, convert_to_tensor=True)
    job_embeddings = model.encode(df["full_text"].tolist(), convert_to_tensor=True)
    scores = util.cos_sim(cv_embedding, job_embeddings)[0]
    out = df.copy()
    out["similarity"] = scores.cpu().numpy()
    return (
        out[["title", "similarity"]]
        .sort_values("similarity", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def run_cli(cv_pdf: str, *, jobs_csv: Path | None = None, top_n: int = DEFAULT_TOP_N) -> None:
    cv_text = extract_cv_text(cv_pdf)
    print(f"Loading model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    print("Embedding CV...")
    df = load_jobs_dataframe(jobs_csv)
    print("Embedding job descriptions...")
    results = rank_jobs_embedding(cv_text, model, df, top_n)
    results.insert(0, "rank", range(1, len(results) + 1))

    print(f"\nTop {top_n} job matches for your CV:\n")
    for _, row in results.iterrows():
        print(f"{int(row['rank'])}. {row['title']:<35} → {row['similarity']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank jobs by embedding similarity to CV.")
    parser.add_argument("cv_pdf", help="Path to CV PDF.")
    parser.add_argument("--jobs", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    args = parser.parse_args()
    run_cli(args.cv_pdf, jobs_csv=args.jobs, top_n=args.top_n)
