"""LLM-powered CV analysis, classification, and career coaching via Claude API."""
from __future__ import annotations

import anthropic

CLAUDE_MODEL = "claude-opus-4-6"

CAREER_DOMAINS = [
    "Data & Analytics",
    "Software Engineering",
    "Marketing",
    "Sales",
    "HR & People",
    "Finance",
    "Product",
    "Operations",
    "Other",
]

# ── Internal helper ───────────────────────────────────────────────────────────

def _call_claude(system: str, user: str, api_key: str, max_tokens: int = 1024) -> str:
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text.strip()


# ── Public functions ──────────────────────────────────────────────────────────

def ask_about_cv(question: str, cv_text: str, api_key: str) -> str:
    """Answer a freeform question about the CV."""
    system = (
        "You are a professional career coach and CV expert. "
        "Answer questions about the CV using bullet points and short paragraphs — never long walls of text. "
        "Base every answer strictly on the CV content. "
        "If information is missing from the CV, say so explicitly."
    )
    return _call_claude(system, f"<cv>\n{cv_text}\n</cv>\n\nQuestion: {question}", api_key)


def classify_cv(cv_text: str, api_key: str) -> str:
    """
    Classify the CV into one career domain using Claude.
    Falls back to keyword matching if the API call fails.
    """
    domains_list = "\n".join(f"- {d}" for d in CAREER_DOMAINS)
    system = "You are a career domain classifier. Return ONLY the domain name from the list — nothing else."
    prompt = (
        f"Classify this person's career domain.\n\n"
        f"Choose EXACTLY one:\n{domains_list}\n\n"
        f"<cv>\n{cv_text[:3000]}\n</cv>\n\n"
        f"Domain:"
    )
    try:
        result = _call_claude(system, prompt, api_key, max_tokens=20)
        for domain in CAREER_DOMAINS:
            if domain.lower() in result.lower():
                return domain
        return "Other"
    except Exception:
        return _classify_cv_fallback(cv_text)


def _classify_cv_fallback(cv_text: str) -> str:
    text = cv_text.lower()
    scores: dict[str, int] = {
        "Data & Analytics":     sum(1 for kw in ["data", "sql", "python", "analytics", "machine learning", "spark", "tableau", "bi "] if kw in text),
        "Software Engineering": sum(1 for kw in ["software", "developer", "backend", "frontend", "api", "react", "java", "kubernetes"] if kw in text),
        "Marketing":            sum(1 for kw in ["marketing", "seo", "content", "campaign", "brand", "social media", "growth"] if kw in text),
        "Sales":                sum(1 for kw in ["sales", "revenue", "quota", "account", "crm", "salesforce", "business development"] if kw in text),
        "HR & People":          sum(1 for kw in ["human resources", "hr", "recruiting", "talent", "people", "onboarding"] if kw in text),
        "Finance":              sum(1 for kw in ["finance", "financial", "accounting", "budget", "forecasting", "cfa", "fp&a"] if kw in text),
        "Product":              sum(1 for kw in ["product", "roadmap", "agile", "scrum", "user research", "ux"] if kw in text),
        "Operations":           sum(1 for kw in ["operations", "process", "logistics", "supply chain", "project management"] if kw in text),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Other"


def extract_skills_with_llm(cv_text: str, api_key: str) -> list[str]:
    """
    Extract skills from CV text using Claude.
    Returns a deduplicated list of skills.
    Falls back to regex extraction if the API call fails.
    """
    system = (
        "You are a CV parser. Extract all professional skills from the CV. "
        "Return ONLY a comma-separated list — no numbering, no explanations, no headers. "
        "Include: programming languages, frameworks, tools, methodologies, certifications, domain expertise."
    )
    prompt = f"<cv>\n{cv_text[:4000]}\n</cv>\n\nSkills (comma-separated list only):"
    try:
        raw = _call_claude(system, prompt, api_key, max_tokens=400)
        return [s.strip() for s in raw.split(",") if s.strip()]
    except Exception:
        from matching import extract_skills_from_text
        return extract_skills_from_text(cv_text)


def generate_career_path(
    cv_text: str,
    missing_skills: list[str],
    category: str,
    api_key: str,
) -> str:
    """
    Generate a step-by-step career development plan using Claude.
    Falls back to a template-based plan if the API call fails.
    """
    system = (
        "You are a senior career coach. Write a clear, actionable 5-step career plan. "
        "Format: numbered steps, each with a bold title on its own line, "
        "then 1-2 sentences of action, then one concrete resource or milestone in italics. "
        "No filler, no long intros. Total response under 400 words."
    )
    skills_str = ", ".join(missing_skills[:8]) if missing_skills else "none identified"
    prompt = (
        f"Career domain: {category}\n"
        f"Skills to develop: {skills_str}\n\n"
        f"<cv>\n{cv_text[:2000]}\n</cv>\n\n"
        f"Write a 5-step plan to help this person advance in {category}."
    )
    try:
        return _call_claude(system, prompt, api_key, max_tokens=600)
    except Exception:
        return _career_path_fallback(missing_skills, category)


def _career_path_fallback(missing_skills: list[str], category: str) -> str:
    lines = [f"**Step 1 — Audit your current position**\nReview your {category} skills against current job postings. _Milestone: list your top 3 gaps._\n"]
    for i, skill in enumerate(missing_skills[:3], 2):
        lines.append(f"**Step {i} — Learn {skill}**\nComplete an online course or hands-on project focused on {skill}. _Resource: search '{skill} tutorial' on YouTube or Coursera._\n")
    n = len(missing_skills[:3]) + 2
    lines.append(f"**Step {n} — Build a portfolio project**\nCreate one end-to-end project that showcases your new skills. _Milestone: publish on GitHub or a personal site._\n")
    lines.append(f"**Step {n + 1} — Apply strategically**\nTarget 10 {category} roles aligned with your upgraded profile. _Milestone: 3 interviews within 60 days._")
    return "\n".join(lines)


def generate_recommendations(
    cv_text: str,
    missing_skills: list[str],
    api_key: str,
) -> str:
    """
    Generate dynamic CV improvement recommendations using Claude.
    Falls back to template bullets if the API call fails.
    """
    system = (
        "You are a CV improvement expert. Give 5 specific, actionable recommendations. "
        "Use bullet points (• prefix). Each bullet: one clear action, why it matters, and how to do it. "
        "Be direct. No vague advice. Under 300 words."
    )
    skills_str = ", ".join(missing_skills[:8]) if missing_skills else "none"
    prompt = (
        f"Missing skills: {skills_str}\n\n"
        f"<cv>\n{cv_text[:2000]}\n</cv>\n\n"
        f"Give 5 bullet-point recommendations to strengthen this CV."
    )
    try:
        return _call_claude(system, prompt, api_key, max_tokens=500)
    except Exception:
        if not missing_skills:
            return (
                "• **Quantify your achievements** — add numbers, percentages, and impact to every bullet.\n"
                "• **Add a professional summary** — 3 sentences: role, years of experience, top value.\n"
                "• **Tailor keywords** to each job description to pass ATS screening.\n"
                "• **List certifications** prominently if you have any relevant ones.\n"
                "• **Include links** to GitHub, portfolio, or LinkedIn."
            )
        bullets = [f"• Learn **{s}** and add a concrete project to your CV that demonstrates it." for s in missing_skills[:5]]
        return "\n".join(bullets)
