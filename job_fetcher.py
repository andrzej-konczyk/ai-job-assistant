"""Fetch live job offers (JustJoin.it) or return a diverse industry-wide mock dataset."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_JUSTJOIN_URL = "https://justjoin.it/api/offers"
_REQUEST_TIMEOUT = 10

_ICON_TO_DOMAIN: dict[str, str] = {
    "data":                    "Data & Analytics",
    "data-engineer":           "Data & Analytics",
    "data-scientist":          "Data & Analytics",
    "data-analyst":            "Data & Analytics",
    "machine-learning":        "Data & Analytics",
    "artificial-intelligence": "Data & Analytics",
    "business-intelligence":   "Data & Analytics",
    "big-data":                "Data & Analytics",
    "analytics":               "Data & Analytics",
    "python":                  "Software Engineering",
    "javascript":              "Software Engineering",
    "typescript":              "Software Engineering",
    "java":                    "Software Engineering",
    "net":                     "Software Engineering",
    "go":                      "Software Engineering",
    "ruby":                    "Software Engineering",
    "scala":                   "Software Engineering",
    "mobile":                  "Software Engineering",
    "ios":                     "Software Engineering",
    "android":                 "Software Engineering",
    "devops":                  "Software Engineering",
    "cloud":                   "Software Engineering",
    "security":                "Software Engineering",
    "game":                    "Software Engineering",
    "embedded":                "Software Engineering",
    "testing":                 "Software Engineering",
    "hr":                      "HR & People",
    "marketing":               "Marketing",
    "sales":                   "Sales",
    "finance":                 "Finance",
    "product":                 "Product",
    "project-management":      "Operations",
    "other":                   "Other",
}


def _skills_to_str(skills: list[dict]) -> str:
    return ", ".join(s.get("name", "") for s in skills if s.get("name"))


def _normalise_justjoin(raw: list[dict]) -> pd.DataFrame:
    rows = []
    for offer in raw:
        icon = (offer.get("marker_icon") or "other").lower()
        domain = _ICON_TO_DOMAIN.get(icon, "Other")
        title = (offer.get("title") or "").strip()
        if not title:
            continue
        required_skills = _skills_to_str(offer.get("skills") or [])
        if not required_skills:
            continue
        description = (offer.get("body") or "").strip()
        if not description:
            parts = [p for p in [
                (offer.get("experience_level") or "").capitalize(),
                (offer.get("company_name") or "").strip(),
                offer.get("workplace_type", ""),
            ] if p]
            description = f"{title} role" + (f" at {', '.join(parts)}" if parts else "")
        rows.append({"title": title, "description": description,
                     "required_skills": required_skills, "category": domain})

    df = pd.DataFrame(rows, columns=["title", "description", "required_skills", "category"])
    df = df.drop_duplicates(subset=["title", "required_skills"])
    return _add_derived_columns(df)


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["skills_list"] = df["required_skills"].apply(
        lambda s: [x.strip() for x in str(s).split(",") if x.strip()]
    )
    df["full_text"] = (
        df["title"].astype(str) + ". "
        + df["description"].astype(str) + " Skills: "
        + df["required_skills"].astype(str)
    )
    return df.reset_index(drop=True)


def _mock_jobs() -> pd.DataFrame:
    """
    Curated multi-industry dataset used when the live API is unavailable.
    Covers: Data, Software, Marketing (SEO/Ads/Social), HR, Finance, Product,
            Sales, Customer Success, Legal, Office/Admin, and Operations.
    """
    rows = [
        # ── Data & Analytics ──────────────────────────────────────────────
        ("Data Engineer",
         "Build and maintain scalable data pipelines. Design Snowflake schemas and lead ETL migrations.",
         "Python, SQL, Apache Spark, Airflow, dbt, Snowflake, Git", "Data & Analytics"),
        ("Data Scientist",
         "Develop ML models and deploy them to production. Design and evaluate A/B experiments.",
         "Python, Machine Learning, Scikit-learn, TensorFlow, SQL, Statistics", "Data & Analytics"),
        ("Data Analyst",
         "Analyze business metrics and build executive dashboards. Support data-driven product decisions.",
         "SQL, Tableau, Excel, Python, Statistics, Power BI", "Data & Analytics"),
        ("ML Engineer",
         "Build and serve ML models at scale. Own CI/CD for model training and deployment pipelines.",
         "Python, MLflow, Docker, Kubernetes, AWS SageMaker, CI/CD", "Data & Analytics"),
        ("Analytics Engineer",
         "Own the analytics data layer. Build dbt models and partner with analysts and scientists.",
         "dbt, SQL, Snowflake, Python, Data Modeling, Git", "Data & Analytics"),

        # ── Software Engineering ──────────────────────────────────────────
        ("Backend Developer",
         "Design and build REST APIs for a high-traffic SaaS platform. Own service reliability.",
         "Python, FastAPI, PostgreSQL, Docker, Redis, AWS", "Software Engineering"),
        ("Full Stack Developer",
         "Build features across the whole stack. Work closely with design to ship great UX.",
         "React, TypeScript, Node.js, GraphQL, PostgreSQL, AWS", "Software Engineering"),
        ("DevOps Engineer",
         "Manage cloud infrastructure and internal developer platform. Reduce deployment friction.",
         "Kubernetes, Terraform, CI/CD, Docker, AWS, Bash", "Software Engineering"),
        ("Cloud Architect",
         "Define cloud strategy and architect multi-region highly available systems.",
         "AWS, Azure, Terraform, Microservices, Security, Networking", "Software Engineering"),
        ("QA Engineer",
         "Design and automate test suites for web and API services. Own the release quality gate.",
         "Pytest, Selenium, API Testing, CI/CD, Python, Postman", "Software Engineering"),

        # ── Marketing — SEO ───────────────────────────────────────────────
        ("SEO Specialist",
         "Drive organic search growth through technical SEO, content, and link-building strategies.",
         "SEO, Google Search Console, Ahrefs, Semrush, Content Writing, HTML, Analytics", "Marketing"),
        ("Senior SEO Manager",
         "Own the global organic search strategy. Lead a team of 3 SEO specialists.",
         "SEO, Technical SEO, Ahrefs, Google Analytics, Content Strategy, Link Building, Python", "Marketing"),
        ("Technical SEO Analyst",
         "Audit and fix crawlability, indexation, and Core Web Vitals issues at scale.",
         "Technical SEO, Screaming Frog, Python, Google Search Console, HTML, JavaScript", "Marketing"),

        # ── Marketing — Paid Ads ──────────────────────────────────────────
        ("Performance Marketing Manager",
         "Run paid acquisition campaigns across Google, Meta, and TikTok Ads. Own ROAS targets.",
         "Google Ads, Meta Ads, TikTok Ads, A/B Testing, Google Analytics, Excel, Attribution", "Marketing"),
        ("Google Ads Specialist",
         "Manage Search, Display, and Shopping campaigns. Optimize bids and creatives weekly.",
         "Google Ads, Google Analytics, Excel, Keyword Research, A/B Testing, Conversion Tracking", "Marketing"),
        ("Paid Social Specialist",
         "Plan and execute paid social campaigns on Meta, LinkedIn, and TikTok.",
         "Meta Ads, LinkedIn Ads, TikTok Ads, Copywriting, A/B Testing, Looker Studio", "Marketing"),
        ("Programmatic Advertising Analyst",
         "Manage DSP campaigns, analyze audience segments, and optimize CPM and CTR.",
         "DV360, The Trade Desk, Excel, SQL, Programmatic Buying, Attribution Modeling", "Marketing"),

        # ── Marketing — General / Content / Growth ────────────────────────
        ("Digital Marketing Manager",
         "Lead paid and organic growth campaigns across all channels. Manage a team of specialists.",
         "Google Analytics, SEO, PPC, Content Strategy, HubSpot, A/B Testing, Excel", "Marketing"),
        ("Growth Hacker",
         "Run experiments across the funnel to identify scalable growth levers.",
         "A/B Testing, SQL, Google Analytics, Email Marketing, CRO, Python", "Marketing"),
        ("Content Strategist",
         "Own the content calendar and brand voice. Produce long-form and social content.",
         "Content Writing, SEO, CMS, Brand Strategy, Social Media, Copywriting", "Marketing"),
        ("Email Marketing Specialist",
         "Manage lifecycle email campaigns. Build automation flows in Klaviyo and HubSpot.",
         "Email Marketing, Klaviyo, HubSpot, Copywriting, A/B Testing, HTML, Segmentation", "Marketing"),
        ("Social Media Manager",
         "Manage brand presence across Instagram, LinkedIn, and X. Create content and engage community.",
         "Social Media, Content Creation, Canva, Copywriting, Analytics, Community Management", "Marketing"),
        ("Marketing Analyst",
         "Measure campaign performance and build attribution models to optimize marketing spend.",
         "Google Analytics, SQL, Tableau, Excel, Attribution Modeling, Python", "Marketing"),

        # ── HR & People ───────────────────────────────────────────────────
        ("HR Business Partner",
         "Act as a strategic partner to engineering leadership on org design and performance.",
         "HR Strategy, Employee Relations, Performance Management, Communication, Coaching", "HR & People"),
        ("Talent Acquisition Specialist",
         "Source, interview, and close candidates for technical roles across Europe.",
         "Recruiting, Sourcing, LinkedIn Recruiter, Employer Branding, ATS, Interviewing", "HR & People"),
        ("People Analytics Manager",
         "Build the people analytics function. Deliver workforce insights to leadership.",
         "SQL, Python, HRIS, Tableau, Statistics, Data Visualization", "HR & People"),
        ("L&D Manager",
         "Design and run learning programs across the company. Own the LMS and vendor relationships.",
         "Training Design, LMS, Facilitation, Needs Analysis, Communication, Curriculum Design", "HR & People"),
        ("Compensation Analyst",
         "Benchmark roles, run annual review cycles, and advise HRBPs on comp strategy.",
         "Excel, Compensation Planning, Market Data, HR Systems, Statistics", "HR & People"),

        # ── Finance ───────────────────────────────────────────────────────
        ("Financial Analyst",
         "Build financial models and support strategic planning. Present to CFO and board.",
         "Excel, Financial Modeling, SQL, Power BI, Forecasting, FP&A", "Finance"),
        ("FP&A Manager",
         "Lead budgeting and forecasting for a €200M business. Build the annual operating plan.",
         "Excel, SAP, Financial Planning, Forecasting, Stakeholder Management, PowerPoint", "Finance"),
        ("Quantitative Analyst",
         "Develop statistical models for pricing and risk in a quant team.",
         "Python, R, Statistics, Time Series, Risk Management, Financial Modeling", "Finance"),
        ("Controller",
         "Own month-end close, statutory reporting, and internal controls for a PE-backed company.",
         "Accounting, IFRS, Excel, ERP, Financial Reporting, Audit", "Finance"),

        # ── Product ───────────────────────────────────────────────────────
        ("Product Manager",
         "Define the roadmap for a B2B SaaS product. Work with engineering, design, and customers.",
         "Agile, Roadmapping, Stakeholder Management, SQL, User Research, Product Strategy", "Product"),
        ("Product Analyst",
         "Instrument product analytics and evaluate experiments. Support data-driven product decisions.",
         "SQL, A/B Testing, Google Analytics, Python, Data Visualization, Amplitude", "Product"),
        ("UX Researcher",
         "Run user research studies and synthesize insights to inform product strategy.",
         "User Research, Usability Testing, Figma, Survey Design, Statistics, Interviewing", "Product"),
        ("Product Designer",
         "Design end-to-end product experiences from discovery to delivery.",
         "Figma, UX Design, Prototyping, User Research, Design Systems, Accessibility", "Product"),

        # ── Sales ─────────────────────────────────────────────────────────
        ("Account Executive",
         "Close net-new business across mid-market accounts in EMEA. Own full-cycle sales.",
         "B2B Sales, Salesforce, Negotiation, Discovery, Forecasting, Communication", "Sales"),
        ("Sales Development Representative",
         "Generate and qualify outbound pipeline for Account Executives.",
         "Outbound Sales, LinkedIn Sales Navigator, Salesloft, Cold Calling, Email Prospecting", "Sales"),
        ("Sales Manager",
         "Lead and coach a team of 6 AEs. Drive quarterly revenue targets and pipeline health.",
         "Sales Leadership, Salesforce, Coaching, Forecasting, Hiring, B2B Sales", "Sales"),
        ("Key Account Manager",
         "Manage and grow relationships with strategic enterprise clients.",
         "Account Management, Negotiation, CRM, Upselling, Executive Relationships, QBRs", "Sales"),

        # ── Customer Success ──────────────────────────────────────────────
        ("Customer Success Manager",
         "Own post-sales relationships with a portfolio of SaaS customers. Drive retention and expansion.",
         "Customer Success, CRM, Onboarding, Renewals, Communication, Product Adoption", "Sales"),
        ("Customer Support Specialist",
         "Handle tier-1 and tier-2 support tickets across chat, email, and phone.",
         "Customer Support, Zendesk, Communication, Troubleshooting, SLA Management", "Sales"),

        # ── Legal ─────────────────────────────────────────────────────────
        ("Legal Counsel",
         "Draft and negotiate commercial contracts. Advise on GDPR, IP, and employment matters.",
         "Contract Law, GDPR, Legal Research, Negotiation, Corporate Law, English, Communication", "Operations"),
        ("Compliance Officer",
         "Ensure the company meets regulatory obligations across EU jurisdictions.",
         "Compliance, GDPR, AML, Risk Assessment, Policy Writing, Stakeholder Management", "Operations"),

        # ── Office / Administrative ───────────────────────────────────────
        ("Office Manager",
         "Run day-to-day office operations for a 150-person HQ. Manage vendors and facilities.",
         "Office Management, Microsoft Office, Communication, Organization, Vendor Management, Excel", "Operations"),
        ("Executive Assistant",
         "Support the CEO and C-suite with calendar management, travel, and board meeting prep.",
         "Calendar Management, Microsoft Office, Communication, Discretion, Travel Coordination", "Operations"),
        ("Administrative Coordinator",
         "Coordinate internal events, office supplies, and HR admin tasks.",
         "Microsoft Office, Excel, Communication, Organization, Time Management", "Operations"),
        ("Receptionist",
         "Greet visitors, manage front desk, and handle incoming communications.",
         "Communication, Microsoft Office, Customer Service, Organization, Multitasking", "Operations"),

        # ── Operations / Project Management ───────────────────────────────
        ("Project Manager",
         "Lead cross-functional delivery of digital transformation projects. Own timelines and budgets.",
         "Project Management, Agile, Jira, Stakeholder Management, Risk Management, PMP", "Operations"),
        ("Operations Manager",
         "Drive operational efficiency across a 60-person team. Own process improvement initiatives.",
         "Operations Management, Excel, Process Improvement, Lean, Stakeholder Management", "Operations"),
        ("Business Analyst",
         "Gather requirements, document processes, and bridge business and IT teams.",
         "Business Analysis, SQL, Jira, Requirements Gathering, Process Mapping, Excel", "Operations"),
    ]

    df = pd.DataFrame(rows, columns=["title", "description", "required_skills", "category"])
    return _add_derived_columns(df)


def fetch_jobs() -> pd.DataFrame:
    """
    Fetch all live job offers from JustJoin.it.
    Falls back to a curated multi-industry mock dataset if the API is unavailable.

    Returns a DataFrame with columns:
        title, description, required_skills, category, skills_list, full_text
    """
    try:
        response = requests.get(
            _JUSTJOIN_URL,
            timeout=_REQUEST_TIMEOUT,
            headers={"User-Agent": "ai-job-assistant/1.0"},
        )
        response.raise_for_status()
        raw: list[dict] = response.json()
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"Unexpected API response: {type(raw)}")
        df = _normalise_justjoin(raw)
        if df.empty:
            raise ValueError("API returned no usable offers.")
        logger.info("Fetched %d offers from JustJoin.it", len(df))
        return df
    except Exception as exc:
        logger.warning("JustJoin.it unavailable (%s). Using mock dataset.", exc)
        return _mock_jobs()
