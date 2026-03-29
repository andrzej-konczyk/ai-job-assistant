"""
AI Career Assistant — Streamlit app.

Run:  py app.py   or   streamlit run app.py
"""
from __future__ import annotations

import html
import subprocess
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path

# Bootstrap: allow `py app.py` to forward to streamlit
if __name__ == "__main__" and "streamlit" not in sys.modules:
    app_path = Path(__file__).resolve()
    raise SystemExit(subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(app_path), *sys.argv[1:]]
    ))

import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util

from config import DEFAULT_THRESHOLD, MODEL_NAME, DEFAULT_TOP_N as CONFIG_TOP_N
from cv_parser import extract_cv_text
from cv_qa import (
    CAREER_DOMAINS,
    ask_about_cv,
    classify_cv,
    extract_skills_with_llm,
    extract_job_skills,
    analyze_job_fit,
    generate_career_path,
    generate_recommendations,
)
from job_fetcher import fetch_jobs
from matching import extract_skills_from_text, find_missing_skills

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Career Assistant", page_icon="🚀", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Hero ── */
  .hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e40af 100%);
    border-radius: 14px; padding: 30px 36px; color: white; margin-bottom: 24px;
  }
  .hero h1 { margin: 0; font-size: 2rem; letter-spacing: -.03em; }
  .hero p  { margin: 6px 0 0; opacity: .8; font-size: 1rem; }

  /* ── Domain badge ── */
  .domain-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: #ede9fe; color: #4c1d95;
    border-radius: 999px; padding: 5px 16px;
    font-size: .85rem; font-weight: 700; margin-bottom: 16px;
  }

  /* ── Match cards ── */
  .match-card {
    border-radius: 12px; padding: 18px 20px; margin-bottom: 14px;
    border-left: 6px solid #94a3b8; background: #f8fafc; color: #0f172a;
  }
  .match-best { border-left-color: #16a34a; background: #f0fdf4; }
  .match-good { border-left-color: #d97706; background: #fffbeb; }
  .match-low  { border-left-color: #dc2626; background: #fef2f2; }
  .match-title { font-size: 1.1rem; font-weight: 800; color: #020617 !important; }
  .match-score { font-size: 1.5rem; font-weight: 800; color: #1d4ed8 !important; }

  /* ── Skill pills ── */
  .pill         { display:inline-block; border-radius:20px; padding:3px 12px;
                  font-size:.82rem; font-weight:600; margin:3px; }
  .pill-have    { background:#dcfce7; color:#15803d; }
  .pill-missing { background:#fee2e2; color:#b91c1c; }

  /* ── Skill gap cards ── */
  .gap-card {
    border-radius: 12px; padding: 18px 20px; margin-bottom: 14px;
    border-left: 6px solid #6366f1; background: #f1f5f9; color: #0f172a;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
  }
  .gap-title { font-size: 1.1rem; font-weight: 800; color: #020617 !important; }
  .gap-score { font-size: 1.35rem; font-weight: 800; color: #1d4ed8 !important; }
  .pill-label { font-size:.7rem; font-weight:700; letter-spacing:.07em;
                text-transform:uppercase; color:#64748b; margin:10px 0 4px; }

  /* ── Game Changer ── */
  .game-changer {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    border-radius: 14px; padding: 24px 28px; color: white; margin-bottom: 20px;
  }
  .game-changer h3 { margin: 0 0 6px; font-size: 1.2rem; }

  /* ── Reco cards ── */
  .reco-card {
    border-radius: 12px; padding: 16px 18px; margin-bottom: 12px;
    border-left: 5px solid #10b981; background: #f8fafc; color: #0f172a;
    box-shadow: 0 1px 2px rgba(0,0,0,.05);
  }
  .reco-skill { font-size: 1.05rem; font-weight: 800; color: #020617 !important; }
  .reco-jobs  { font-size: .88rem; color: #64748b !important; margin: 2px 0 8px; }
  .reco-line  { font-size: .92rem; color: #334155 !important; margin: 4px 0; }

  /* ── Career path ── */
  .path-step {
    border-radius: 12px; padding: 16px 20px; margin-bottom: 12px;
    background: #f8fafc; border-left: 5px solid #2563eb; color: #0f172a;
    box-shadow: 0 1px 2px rgba(0,0,0,.05);
  }
  .path-ai-box {
    border-radius: 12px; padding: 20px 22px; background: #faf5ff;
    border: 1px solid #e9d5ff; color: #1e1b4b !important;
    font-size: .95rem; line-height: 1.7; white-space: pre-wrap;
  }

  /* ── Section label ── */
  .section-label {
    font-size:.72rem; font-weight:700; letter-spacing:.08em;
    text-transform:uppercase; color:#64748b; margin-bottom:8px;
  }

  /* ── AI answer box ── */
  .ai-answer {
    border-radius: 12px; padding: 18px 20px; background: #f0f9ff;
    border-left: 5px solid #0ea5e9; color: #0c4a6e !important;
    font-size: .95rem; line-height: 1.7; white-space: pre-wrap;
  }
  .ai-answer strong { color: #0c4a6e !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
TOP_N     = CONFIG_TOP_N
THRESHOLD = DEFAULT_THRESHOLD

DOMAIN_ICONS = {
    "Data & Analytics":    "📊",
    "Software Engineering":"💻",
    "Marketing":           "📣",
    "Sales":               "💰",
    "HR & People":         "🤝",
    "Finance":             "📈",
    "Product":             "🎯",
    "Operations":          "⚙️",
    "Other":               "🔍",
}

LEARNING_RESOURCES: dict[str, str] = {
    "Docker":             "docker.com/get-started",
    "Kubernetes":         "kubernetes.io/docs/tutorials/kubernetes-basics",
    "Airflow":            "airflow.apache.org/docs",
    "dbt":                "courses.getdbt.com (free fundamentals)",
    "Spark":              "spark.apache.org/docs/latest",
    "Kafka":              "kafka.apache.org/quickstart",
    "Snowflake":          "quickstarts.snowflake.com",
    "Terraform":          "developer.hashicorp.com/terraform/tutorials",
    "MLflow":             "mlflow.org/docs",
    "Power BI":           "learn.microsoft.com/en-us/training/powerplatform/power-bi",
    "Tableau":            "help.tableau.com",
    "AWS SageMaker":      "docs.aws.amazon.com/sagemaker",
    "A/B Testing":        "'Trustworthy Online Controlled Experiments' (book)",
    "Deep Learning":      "fast.ai — free course",
    "TensorFlow":         "tensorflow.org/tutorials",
    "PyTorch":            "pytorch.org/tutorials",
    "HuggingFace":        "huggingface.co/learn/nlp-course",
    "Data Governance":    "DAMA DMBOK or Collibra University",
    "CI/CD":              "GitHub Actions quickstart",
    "Feature Engineering":"'Feature Engineering for ML' by Alice Zheng",
    "Statistics":         "Khan Academy or StatQuest (YouTube)",
    "Data Modeling":      "'The Data Warehouse Toolkit' by Kimball",
    "R":                  "r4ds.had.co.nz (free online)",
    "SEO":                "moz.com/beginners-guide-to-seo",
    "Google Ads":         "skillshop.google.com (free certification)",
    "Meta Ads":           "facebook.com/business/learn",
    "Google Analytics":   "analytics.google.com/analytics/academy",
    "HubSpot":            "academy.hubspot.com (free courses)",
    "Salesforce":         "trailhead.salesforce.com",
    "Excel":              "support.microsoft.com/excel",
    "Figma":              "help.figma.com/hc/en-us",
    "Agile":              "agilemanifesto.org + Scrum Guide (free PDF)",
}

CV_TIPS: dict[str, str] = {
    "Docker":         "Add: 'Containerized data pipelines using Docker'",
    "Airflow":        "Add: 'Orchestrated workflows with Apache Airflow DAGs'",
    "dbt":            "Mention dbt models you've built; add to Skills section",
    "Kubernetes":     "Add to cloud/infra bullet points",
    "SEO":            "Add: 'Improved organic traffic by X% through on-page SEO'",
    "Google Ads":     "Add: 'Managed Google Ads campaigns with €X monthly budget'",
    "A/B Testing":    "Add: 'Designed and evaluated A/B experiments for [feature]'",
    "Deep Learning":  "Add a Projects section — even a Kaggle entry counts",
    "CI/CD":          "Add: 'Set up CI/CD pipelines using GitHub Actions'",
    "Agile":          "Add: 'Worked in Agile/Scrum teams, ran sprint planning'",
    "Excel":          "Add: 'Built financial models / dashboards in Excel'",
}

# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def load_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)

@st.cache_data(show_spinner="Fetching live job listings…", ttl=3600)
def load_jobs() -> pd.DataFrame:
    return fetch_jobs()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚀 AI Career Assistant")
    st.markdown("Upload your CV — get ranked job matches, skill gaps, a career path, and AI coaching.")
    st.divider()

    uploaded_file = st.file_uploader("Upload CV (PDF)", type=["pdf"])
    st.divider()

    st.markdown("**Claude API key** _(optional — enables AI features)_")
    anthropic_key = st.text_input(
        "API key", type="password", placeholder="sk-ant-...",
        help="Powers CV classification, AI chat, career path, and smart recommendations. Never stored.",
        label_visibility="collapsed",
    )
    st.divider()

    top_n = st.slider("Top jobs to show", 3, 15, TOP_N)
    filter_domain = st.selectbox(
        "Filter by industry", ["All industries"] + CAREER_DOMAINS, index=0
    )
    threshold = st.slider(
        "Skill sensitivity", 0.25, 0.70, THRESHOLD, 0.05,
        help="Lower = more skills flagged as missing."
    )
    st.divider()
    st.caption(f"Embedding model: `{MODEL_NAME}`")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🚀 AI Career Assistant</h1>
  <p>Upload your CV → instant job matches, skill gap analysis, career path plan, and AI coaching — across all industries.</p>
</div>
""", unsafe_allow_html=True)

if uploaded_file is None:
    col1, col2, col3 = st.columns(3)
    col1.info("**Step 1** — Upload your CV (PDF) in the sidebar")
    col2.info("**Step 2** — Optionally add your Anthropic API key for AI features")
    col3.info("**Step 3** — Explore matches, gaps, and your career path")
    st.stop()

# ── Parse CV ──────────────────────────────────────────────────────────────────
with st.spinner("Parsing CV…"):
    cv_text = extract_cv_text(BytesIO(uploaded_file.getvalue()))

if not cv_text.strip():
    st.error("Could not extract text from the PDF. Make sure it is not a scanned image-only PDF.")
    st.stop()

# ── CV classification ─────────────────────────────────────────────────────────
has_key = bool(anthropic_key)

with st.spinner("Classifying CV domain…"):
    if has_key:
        cv_domain = classify_cv(cv_text, anthropic_key)
    else:
        from cv_qa import _classify_cv_fallback
        cv_domain = _classify_cv_fallback(cv_text)

domain_icon = DOMAIN_ICONS.get(cv_domain, "🔍")
st.markdown(
    f'<div class="domain-badge">{domain_icon} Detected domain: <strong>{cv_domain}</strong>'
    + ("&nbsp;&nbsp;·&nbsp;&nbsp;🤖 AI classified" if has_key else "&nbsp;&nbsp;·&nbsp;&nbsp;🔤 Keyword classified")
    + "</div>",
    unsafe_allow_html=True,
)

# ── Skill extraction ──────────────────────────────────────────────────────────
with st.spinner("Extracting CV skills…"):
    if has_key:
        cv_skills = extract_skills_with_llm(cv_text, anthropic_key)
    else:
        cv_skills = extract_skills_from_text(cv_text)

with st.expander(f"Extracted CV text  ·  {len(cv_skills)} skills detected", expanded=False):
    st.text(cv_text)
    if cv_skills:
        st.markdown("**Detected skills:** " + ", ".join(f"`{s}`" for s in cv_skills))

# ── Load jobs & compute matches ───────────────────────────────────────────────
model = load_model()
df    = load_jobs()

# Apply industry filter
if filter_domain != "All industries" and "category" in df.columns:
    df_filtered = df[df["category"] == filter_domain]
    df_work = df_filtered if not df_filtered.empty else df
else:
    df_work = df

with st.spinner("Computing similarity scores…"):
    job_emb  = model.encode(df_work["full_text"].tolist(), convert_to_tensor=True)
    cv_emb   = model.encode(cv_text, convert_to_tensor=True)
    df_work  = df_work.copy()
    df_work["similarity"] = util.cos_sim(cv_emb, job_emb)[0].cpu().numpy()
    top_jobs = df_work.sort_values("similarity", ascending=False).head(top_n).reset_index(drop=True)

# ── Skill gap analysis ────────────────────────────────────────────────────────
with st.spinner("Analysing skill gaps…"):
    gap_rows: list[dict] = []
    for _, row in top_jobs.iterrows():
        missing = find_missing_skills(cv_text, row["skills_list"], model, threshold)
        matched = [s for s in row["skills_list"] if s not in missing]
        gap_rows.append({
            "job":         row["title"],
            "similarity":  row["similarity"],
            "description": row["description"],
            "missing":     missing,
            "matched":     matched,
            "all_skills":  row["skills_list"],
            "category":    row.get("category", ""),
        })

# Aggregate across top jobs
skill_gap:  dict[str, list[str]] = {}
for entry in gap_rows:
    for skill in entry["missing"]:
        skill_gap.setdefault(skill, []).append(entry["job"])
skill_freq = Counter({s: len(j) for s, j in skill_gap.items()})

# ── Summary metrics ───────────────────────────────────────────────────────────
best_score = top_jobs.iloc[0]["similarity"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Best match",        f"{best_score:.0%}")
m2.metric("Jobs found",        len(df_work))
m3.metric("Skill gaps",        len(skill_freq))
m4.metric("Skills on CV",      len(cv_skills))
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Job Matches",
    "🔍 Skill Gaps",
    "🚀 Career Path",
    "💬 AI Chat",
    "📋 Analyze a Job",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Job Matches
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-label">Ranked by semantic similarity to your CV</div>', unsafe_allow_html=True)

    for i, entry in enumerate(gap_rows):
        score   = entry["similarity"]
        is_best = i == 0

        if score >= 0.72:
            cls, badge = "match-best", "🟢 Strong match"
        elif score >= 0.55:
            cls, badge = "match-good", "🟡 Good match"
        else:
            cls, badge = "match-low",  "🔴 Partial match"

        best_tag = (
            '<span style="background:#16a34a;color:#fff;border-radius:6px;'
            'padding:2px 9px;font-size:.78rem;margin-left:8px;">★ Best match</span>'
            if is_best else ""
        )
        cat_tag = ""
        if entry["category"]:
            cat_icon = DOMAIN_ICONS.get(entry["category"], "")
            cat_tag = (
                f'<span style="background:#e0f2fe;color:#0369a1;border-radius:6px;'
                f'padding:2px 9px;font-size:.78rem;margin-left:6px;">'
                f'{cat_icon} {html.escape(entry["category"])}</span>'
            )

        pills_have    = "".join(f'<span class="pill pill-have">✓ {html.escape(s)}</span>'    for s in entry["matched"])
        pills_missing = "".join(f'<span class="pill pill-missing">✗ {html.escape(s)}</span>' for s in entry["missing"])

        st.markdown(f"""
<div class="match-card {cls}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin-bottom:8px;">
    <span class="match-title">{i+1}. {html.escape(entry['job'])}{best_tag}{cat_tag}</span>
    <span class="match-score">{score:.0%}</span>
  </div>
  <div style="color:#64748b;font-size:.88rem;margin-bottom:10px;">{html.escape(entry['description'])}</div>
  <div style="margin-bottom:4px;font-size:.88rem;font-weight:600;">{badge}</div>
  <div>{pills_have}{pills_missing}</div>
</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Skill Gaps
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-label">Missing skills per job — sorted by match score</div>', unsafe_allow_html=True)

    for entry in gap_rows:
        score = entry["similarity"]
        m_pills = "".join(f'<span class="pill pill-missing">✗ {html.escape(s)}</span>' for s in entry["missing"])
        h_pills = "".join(f'<span class="pill pill-have">✓ {html.escape(s)}</span>'    for s in entry["matched"])

        missing_block = (
            f'<div class="pill-label">Missing</div><div>{m_pills}</div>' if entry["missing"]
            else '<div style="color:#15803d;font-weight:600;margin-top:6px;">✓ All skills matched</div>'
        )
        matched_block = f'<div class="pill-label">You have</div><div>{h_pills}</div>' if entry["matched"] else ""

        st.markdown(f"""
<div class="gap-card">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
    <span class="gap-title">{html.escape(entry['job'])}</span>
    <span class="gap-score">{score:.0%}</span>
  </div>
  {missing_block}
  {matched_block}
</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Career Path
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    all_missing = list(skill_freq.keys())

    # ── Game Changer ─────────────────────────────────────────────────────────
    if skill_freq:
        top_skill, top_count = skill_freq.most_common(1)[0]
        res = LEARNING_RESOURCES.get(top_skill, "Search on YouTube / Coursera")
        st.markdown(f"""
<div class="game-changer">
  <h3>⚡ Game Changer Skill</h3>
  <p style="font-size:1.35rem;font-weight:800;margin:4px 0;">{html.escape(top_skill)}</p>
  <p style="opacity:.9;margin:0;">Adding <strong>{html.escape(top_skill)}</strong> unlocks
  <strong>{top_count} of your top {top_n} target roles</strong>.<br>
  📚 Start here: {html.escape(res)}</p>
</div>""", unsafe_allow_html=True)

    # ── AI Career Path ────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Your personalised career development plan</div>', unsafe_allow_html=True)

    if has_key:
        if "career_path_cache" not in st.session_state:
            with st.spinner("Generating career path with Claude…"):
                st.session_state.career_path_cache = generate_career_path(
                    cv_text, all_missing, cv_domain, anthropic_key
                )
        st.markdown(
            f'<div class="path-ai-box">🤖 <strong>AI-generated career plan for {html.escape(cv_domain)}</strong><br><br>'
            + html.escape(st.session_state.career_path_cache)
            + "</div>",
            unsafe_allow_html=True,
        )
    else:
        from cv_qa import _career_path_fallback
        fallback = _career_path_fallback(all_missing[:5], cv_domain)
        st.info("Add your Anthropic API key in the sidebar for a personalised AI-generated plan.")
        for line in fallback.split("\n"):
            if line.strip():
                with st.container(border=True):
                    st.markdown(line)

    # ── Skill Recommendations ─────────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-label">Skill gap recommendations</div>', unsafe_allow_html=True)

    if has_key:
        if "reco_cache" not in st.session_state:
            with st.spinner("Generating recommendations with Claude…"):
                st.session_state.reco_cache = generate_recommendations(cv_text, all_missing, anthropic_key)
        reco_text = st.session_state.reco_cache
        st.markdown(
            f'<div class="ai-answer">🤖 <strong>AI recommendations</strong><br><br>{html.escape(reco_text)}</div>',
            unsafe_allow_html=True,
        )
    else:
        if not skill_freq:
            st.success("No major skill gaps found! Your CV is well-aligned with the top matches.")
        else:
            for skill, count in skill_freq.most_common():
                jobs_need = ", ".join(skill_gap[skill])
                res  = LEARNING_RESOURCES.get(skill, "")
                tip  = CV_TIPS.get(skill, "")
                st.markdown(f"""
<div class="reco-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;">
    <span class="reco-skill">{html.escape(skill)}</span>
    <span style="font-size:.95rem;font-weight:700;color:#047857;">{count}/{top_n} roles</span>
  </div>
  <div class="reco-jobs">Required by: {html.escape(jobs_need)}</div>
  {"<div class='reco-line'>📚 <strong>Learn:</strong> " + html.escape(res) + "</div>" if res else ""}
  {"<div class='reco-line'>📝 <strong>CV tip:</strong> " + html.escape(tip) + "</div>" if tip else ""}
</div>""", unsafe_allow_html=True)

    # ── Priority plan (always shown) ──────────────────────────────────────────
    if skill_freq:
        st.divider()
        st.markdown("**Priority action plan**")
        for i, (skill, count) in enumerate(skill_freq.most_common(3), 1):
            pct = min(100, int(count / top_n * 100))
            st.markdown(f"{i}. Add **`{skill}`** — needed by {count} of your top {top_n} matches")
            st.progress(pct)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — AI Chat
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    if not has_key:
        st.warning("Enter your Anthropic API key in the sidebar to use the AI chat.")
        st.info("The AI chat lets you ask anything about your CV: strengths, positioning, interview prep, salary negotiation, and more.")
        st.stop()

    st.markdown("Ask Claude anything about your CV — strengths, gaps, how to position yourself, interview prep, salary negotiation, and more.")

    # Initialise chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: list[dict[str, str]] = []

    # Render history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🙋" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

    # Suggested questions (only shown before first message)
    if not st.session_state.chat_history:
        st.markdown("**Try asking:**")
        suggestions = [
            "What are my strongest skills for a Data Engineering role?",
            "What is missing from my CV to be competitive in Product Management?",
            "How should I position myself for a career change to Marketing?",
            "Write a professional summary for my CV in 3 sentences.",
        ]
        cols = st.columns(2)
        for j, q in enumerate(suggestions):
            if cols[j % 2].button(q, key=f"sugg_{j}"):
                st.session_state.chat_history.append({"role": "user", "content": q})
                with st.spinner("Claude is thinking…"):
                    answer = ask_about_cv(q, cv_text, anthropic_key)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.rerun()

    # Chat input
    if prompt := st.chat_input("Ask about your CV…"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🙋"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking…"):
                try:
                    answer = ask_about_cv(prompt, cv_text, anthropic_key)
                except Exception as e:
                    answer = f"⚠️ API error: {e}"
            st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Analyze a Job
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown(
        "Paste any job offer below — from LinkedIn, a company website, or anywhere else. "
        "Get your **fit score**, **skill gap**, and **AI feedback** tailored to that specific role."
    )

    job_input = st.text_area(
        "Job offer text",
        height=260,
        placeholder=(
            "Paste the full job description here…\n\n"
            "e.g.:\n"
            "Senior Data Engineer — Acme Corp\n\n"
            "We are looking for a Data Engineer to build and maintain our data platform…\n"
            "Requirements:\n- 3+ years Python\n- Experience with Airflow, dbt, Snowflake\n- ..."
        ),
        key="job_offer_input",
    )

    col_btn, col_clear = st.columns([1, 5])
    run_analysis = col_btn.button("🔍 Analyze fit", type="primary", disabled=not job_input.strip())
    if col_clear.button("Clear", type="secondary") and "job_analysis" in st.session_state:
        del st.session_state["job_analysis"]
        st.rerun()

    # Run & cache analysis
    if run_analysis and job_input.strip():
        with st.spinner("Extracting skills from job offer…"):
            job_skills = extract_job_skills(job_input, anthropic_key)

        with st.spinner("Computing match score…"):
            job_emb_single = model.encode(job_input, convert_to_tensor=True)
            raw_score = float(util.cos_sim(cv_emb, job_emb_single).item())

        with st.spinner("Finding skill gaps…"):
            job_missing = find_missing_skills(cv_text, job_skills, model, threshold)
            job_matched = [s for s in job_skills if s not in job_missing]

        ai_feedback = None
        if has_key:
            with st.spinner("Generating AI feedback with Claude…"):
                try:
                    ai_feedback = analyze_job_fit(cv_text, job_input, job_missing, raw_score, anthropic_key)
                except Exception as e:
                    ai_feedback = f"⚠️ API error: {e}"

        st.session_state.job_analysis = {
            "score":     raw_score,
            "skills":    job_skills,
            "missing":   job_missing,
            "matched":   job_matched,
            "feedback":  ai_feedback,
        }

    # Render results
    if "job_analysis" in st.session_state:
        r = st.session_state.job_analysis
        score = r["score"]

        st.divider()

        # ── Score banner ──────────────────────────────────────────────────
        if score >= 0.72:
            color, label = "#16a34a", "Strong match"
        elif score >= 0.55:
            color, label = "#d97706", "Good match"
        else:
            color, label = "#dc2626", "Partial match"

        st.markdown(f"""
<div style="border-radius:14px;padding:20px 26px;background:{color}18;
     border-left:6px solid {color};margin-bottom:18px;color:#0f172a;">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <span style="font-size:1.05rem;font-weight:700;color:{color};">{label}</span>
    <span style="font-size:2.2rem;font-weight:900;color:{color};">{score:.0%}</span>
  </div>
  <div style="font-size:.88rem;color:#64748b;margin-top:4px;">
    Semantic similarity between your CV and this job offer
  </div>
</div>""", unsafe_allow_html=True)

        # ── Skill pills ───────────────────────────────────────────────────
        col_m, col_h = st.columns(2)

        with col_m:
            st.markdown("**Missing skills**")
            if r["missing"]:
                pills = "".join(
                    f'<span class="pill pill-missing">✗ {html.escape(s)}</span>'
                    for s in r["missing"]
                )
                st.markdown(pills, unsafe_allow_html=True)
            else:
                st.success("✓ No missing skills detected")

        with col_h:
            st.markdown("**Skills you already have**")
            if r["matched"]:
                pills = "".join(
                    f'<span class="pill pill-have">✓ {html.escape(s)}</span>'
                    for s in r["matched"]
                )
                st.markdown(pills, unsafe_allow_html=True)
            else:
                st.info("No skill matches detected")

        # ── AI Feedback ───────────────────────────────────────────────────
        st.divider()
        if r["feedback"]:
            st.markdown("**AI Feedback**")
            st.markdown(
                f'<div class="ai-answer">🤖 {html.escape(r["feedback"])}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Add your Anthropic API key in the sidebar for detailed AI feedback on this job.")
