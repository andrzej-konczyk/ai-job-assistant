"""Unit tests for matching.py"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from matching import extract_skills_from_text, find_missing_skills


# ── extract_skills_from_text ──────────────────────────────────────────────────

class TestExtractSkillsFromText:
    def test_finds_exact_skill(self):
        assert "Python" in extract_skills_from_text("I have 5 years of Python experience.")

    def test_case_insensitive(self):
        assert "SQL" in extract_skills_from_text("proficient in sql and data modeling")

    def test_finds_multiple_skills(self):
        text = "Worked with Python, SQL, Docker, and Airflow."
        result = extract_skills_from_text(text)
        assert "Python" in result
        assert "SQL" in result
        assert "Docker" in result
        assert "Airflow" in result

    def test_no_false_positives(self):
        """Skills not mentioned should not appear in results."""
        result = extract_skills_from_text("I enjoy cooking and gardening.")
        assert result == []

    def test_word_boundary_respected(self):
        """'R' should not match inside 'orchestrator' or 'Docker'."""
        result = extract_skills_from_text("I use Docker and orchestrator tools.")
        assert "R" not in result

    def test_multiword_skill(self):
        assert "Machine Learning" in extract_skills_from_text(
            "Experience in machine learning and statistics."
        )

    def test_empty_text(self):
        assert extract_skills_from_text("") == []

    def test_returns_list(self):
        result = extract_skills_from_text("Python SQL")
        assert isinstance(result, list)


# ── find_missing_skills ───────────────────────────────────────────────────────

def _mock_model(cv_max_scores: list[float]):
    """
    Return a mock SentenceTransformer whose cos_sim returns controllable scores.

    cv_max_scores[i] is the max similarity that job skill i gets against any
    CV sentence. Values >= threshold → matched; < threshold → missing.
    """
    n_skills = len(cv_max_scores)
    n_sentences = 3  # arbitrary number of CV sentences

    # encode() returns a tensor of shape (n, dim) — content doesn't matter
    # because cos_sim is also mocked via the util patch
    model = MagicMock()
    model.encode.return_value = torch.zeros(max(n_skills, n_sentences), 16)

    # Build a (n_skills × n_sentences) similarity matrix where each row's max
    # equals the desired score for that skill
    sim_matrix = torch.zeros(n_skills, n_sentences)
    for i, score in enumerate(cv_max_scores):
        sim_matrix[i, 0] = score  # put the score in the first column

    return model, sim_matrix


@patch("matching.util.cos_sim")
def test_all_skills_present(mock_cos_sim):
    """If all similarity scores are above threshold, nothing should be missing."""
    scores = [0.80, 0.75, 0.90]
    model, sim_matrix = _mock_model(scores)
    mock_cos_sim.return_value = sim_matrix

    missing = find_missing_skills(
        cv_text="Python SQL Docker experience.",
        job_skills=["Python", "SQL", "Docker"],
        model=model,
        threshold=0.45,
    )
    assert missing == []


@patch("matching.util.cos_sim")
def test_all_skills_missing(mock_cos_sim):
    """If all scores are below threshold, all skills should be flagged."""
    scores = [0.10, 0.20, 0.15]
    model, sim_matrix = _mock_model(scores)
    mock_cos_sim.return_value = sim_matrix

    missing = find_missing_skills(
        cv_text="I enjoy cooking and gardening.",
        job_skills=["Kubernetes", "Kafka", "Terraform"],
        model=model,
        threshold=0.45,
    )
    assert set(missing) == {"Kubernetes", "Kafka", "Terraform"}


@patch("matching.util.cos_sim")
def test_partial_match(mock_cos_sim):
    """Only skills below threshold should appear as missing."""
    # Python → 0.80 (present), Docker → 0.20 (missing), Airflow → 0.30 (missing)
    scores = [0.80, 0.20, 0.30]
    model, sim_matrix = _mock_model(scores)
    mock_cos_sim.return_value = sim_matrix

    missing = find_missing_skills(
        cv_text="Python experience.",
        job_skills=["Python", "Docker", "Airflow"],
        model=model,
        threshold=0.45,
    )
    assert "Python" not in missing
    assert "Docker" in missing
    assert "Airflow" in missing


@patch("matching.util.cos_sim")
def test_threshold_boundary(mock_cos_sim):
    """Score exactly equal to threshold should be treated as matched (not missing)."""
    scores = [0.45]
    model, sim_matrix = _mock_model(scores)
    mock_cos_sim.return_value = sim_matrix

    missing = find_missing_skills(
        cv_text="Some text.",
        job_skills=["SQL"],
        model=model,
        threshold=0.45,
    )
    assert "SQL" not in missing


def test_empty_cv_text_returns_all_skills():
    """When CV text is empty, every job skill should be flagged as missing."""
    model = MagicMock()
    missing = find_missing_skills(
        cv_text="",
        job_skills=["Python", "SQL"],
        model=model,
        threshold=0.45,
    )
    assert set(missing) == {"Python", "SQL"}
    model.encode.assert_not_called()
