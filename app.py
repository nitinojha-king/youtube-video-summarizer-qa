import html
import os
import re
import tempfile
import time
from urllib.parse import parse_qs, urlparse

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from groq import Groq

import main
from main import (
    answer_question_with_docs,
    build_embeddings,
    build_llm,
    build_retriever,
    download_audio,
    summarize_transcript,
    transcribe_audio,
)

st.set_page_config(
    page_title="VidLens | Video intelligence",
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400;1,6..72,500&family=Inter:wght@400;500;600&display=swap');

:root {
    --paper: #f5f2ea;
    --card: #fdfbf6;
    --ink: #1b1917;
    --ink-soft: #403b34;
    --muted: #6d6659;
    --faint: #756d5f;
    --line: #ded8cb;
    --line-strong: #c6bead;
    --accent: #a8442b;
    --radius: 8px;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    -webkit-font-smoothing: antialiased;
}

.stApp {
    background: var(--paper);
    color: var(--ink);
}

[data-testid="stAppViewContainer"] {
    background: transparent;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="collapsedControl"],
[data-testid="stHeaderActionElements"],
#MainMenu,
footer {
    display: none !important;
}

.block-container {
    max-width: 1060px;
    padding: 1.7rem 2.4rem 6rem;
}

::selection {
    background: rgba(168, 68, 43, 0.16);
    color: var(--ink);
}

/* Masthead */

.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    padding-bottom: 0.95rem;
    border-bottom: 1px solid var(--line);
}

.brand-lockup {
    display: flex;
    align-items: center;
    gap: 11px;
}

.brand-mark {
    width: 27px;
    height: 27px;
    display: grid;
    place-items: center;
    border-radius: 4px;
    background: var(--ink);
}

.brand-mark svg {
    width: 16px;
    height: 16px;
}

.brand-name {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.08rem;
    font-weight: 500;
    letter-spacing: -0.01em;
    color: var(--ink);
}

.brand-name span {
    margin-left: 9px;
    color: var(--faint);
    font-family: 'Inter', sans-serif;
    font-size: 0.64rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

.top-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 0;
    border: 0;
    border-radius: 0;
    background: none;
    color: var(--faint);
    font-family: 'Inter', sans-serif;
    font-size: 0.64rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}

.badge-dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: var(--accent);
}

/* Hero */

.hero {
    max-width: 780px;
    margin: 0;
    padding: 3.4rem 0 0;
    text-align: left;
}

.hero.compact {
    max-width: 800px;
    padding: 2.3rem 0 0;
}

.eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    color: var(--faint);
    font-size: 0.64rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}

.eyebrow-dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: var(--accent);
}

.hero h1 {
    margin: 1.15rem 0 0.95rem;
    padding: 0;
    font-family: 'Newsreader', Georgia, serif;
    font-size: clamp(2.5rem, 5.4vw, 4.05rem);
    font-weight: 400;
    line-height: 1.05;
    letter-spacing: -0.022em;
    color: var(--ink);
}

.hero.compact h1 {
    margin: 1rem 0 0.8rem;
    font-size: clamp(2rem, 3.9vw, 3rem);
    line-height: 1.1;
}

.hero h1 .accent {
    font-style: italic;
    color: var(--accent);
    white-space: nowrap;
}

.hero > p {
    max-width: 58ch;
    margin: 0;
    color: var(--muted);
    font-size: 0.98rem;
    line-height: 1.75;
}

.hero.compact > p {
    font-size: 0.94rem;
}

.hero-tags {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0;
    margin-top: 1.4rem;
}

.hero-tag {
    padding: 0;
    border: 0;
    border-radius: 0;
    background: none;
    color: var(--faint);
    font-size: 0.73rem;
    font-weight: 400;
    letter-spacing: 0.01em;
}

.hero-tag:not(:last-child)::after {
    content: "/";
    margin: 0 0.65rem;
    color: var(--line-strong);
}

/* Analyze form */

[data-testid="stForm"] {
    margin-top: 2.6rem;
    padding: 1.4rem 0 0;
    border: 0;
    border-top: 1px solid var(--line);
    border-radius: 0;
    background: transparent;
    box-shadow: none;
    backdrop-filter: none;
}

[data-testid="stForm"] [data-testid="stHorizontalBlock"] {
    align-items: flex-end;
}

[data-testid="stForm"] [data-testid="stVerticalBlock"] {
    gap: 0.3rem;
}

.field-label {
    margin: 0 0 0.45rem 0.1rem;
    color: var(--faint);
    font-size: 0.63rem;
    font-weight: 500;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}

[data-baseweb="input"],
[data-testid="stTextInputRootElement"] {
    border: 1px solid var(--line-strong) !important;
    border-radius: 6px !important;
    background: var(--card) !important;
    transition: border-color 0.15s ease;
}

[data-baseweb="input"]:focus-within,
[data-testid="stTextInputRootElement"]:focus-within {
    border-color: var(--ink) !important;
    background: var(--card) !important;
    box-shadow: none !important;
}

[data-baseweb="base-input"] {
    background: transparent !important;
}


[data-testid="stTextInput"] input {
    min-height: 48px;
    padding: 0 14px;
    border: 0 !important;
    background: transparent !important;
    color: var(--ink) !important;
    font-family: 'Inter', sans-serif;
    font-size: 0.94rem;
    caret-color: var(--accent);
}

[data-testid="stTextInput"] input::placeholder {
    color: #8c8373;
    opacity: 1;
}

[data-testid="stTextInput"] label,
[data-testid="InputInstructions"] {
    display: none;
}

button[kind="primaryFormSubmit"],
[data-testid="stFormSubmitButton"] button {
    min-height: 48px;
    border: 0 !important;
    border-radius: 6px !important;
    background: var(--ink) !important;
    color: #fbf9f3 !important;
    font-family: 'Inter', sans-serif;
    font-size: 0.84rem;
    font-weight: 500;
    letter-spacing: 0.01em;
    box-shadow: none !important;
    transition: background 0.15s ease;
}

button[kind="primaryFormSubmit"]:hover,
[data-testid="stFormSubmitButton"] button:hover {
    background: #37322b !important;
    color: #fbf9f3 !important;
}

button[kind="primaryFormSubmit"]:focus,
[data-testid="stFormSubmitButton"] button:focus {
    box-shadow: 0 0 0 2px var(--paper), 0 0 0 3px var(--ink) !important;
}

.form-note {
    margin: 0.85rem 0 0;
    text-align: left;
    color: var(--faint);
    font-size: 0.72rem;
    line-height: 1.6;
}

/* Empty-state features */

.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 2.4rem;
    margin-top: 3.6rem;
}

.feature-card {
    min-height: 0;
    padding: 1.05rem 0 0;
    border: 0;
    border-top: 1px solid var(--line-strong);
    border-radius: 0;
    background: none;
    box-shadow: none;
    transition: border-color 0.2s ease;
}

.feature-card:hover {
    transform: none;
    border-top-color: var(--ink);
    background: none;
}

.feature-index {
    display: block;
    margin-bottom: 0.6rem;
    color: var(--accent);
    font-family: 'Newsreader', Georgia, serif;
    font-size: 0.88rem;
    font-style: italic;
    font-weight: 400;
    letter-spacing: 0;
}

.feature-card h3 {
    margin: 0 0 0.4rem;
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.16rem;
    font-weight: 500;
    letter-spacing: -0.01em;
    color: var(--ink);
}

.feature-card p {
    margin: 0;
    color: var(--muted);
    font-size: 0.82rem;
    line-height: 1.68;
}

/* Metrics */

[data-testid="stMetric"] {
    min-height: 0;
    padding: 0.1rem 0 0.1rem 1rem;
    border: 0;
    border-left: 1px solid var(--line-strong);
    border-radius: 0;
    background: none;
    box-shadow: none;
}

[data-testid="stMetricLabel"] {
    color: var(--faint);
    font-size: 0.62rem;
    font-weight: 500;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}

[data-testid="stMetricValue"] {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.5rem;
    font-weight: 400;
    letter-spacing: -0.01em;
    color: var(--ink);
}

/* Bordered containers */

[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"] {
    border: 1px solid var(--line) !important;
    border-radius: var(--radius) !important;
    background: var(--card) !important;
    box-shadow: none !important;
    padding: 1.3rem 1.4rem;
    backdrop-filter: none;
}

[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"],
[data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"] {
    gap: 0.7rem;
}

[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMarkdownContainer"] p,
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMarkdownContainer"] li,
[data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"] [data-testid="stMarkdownContainer"] p,
[data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"] [data-testid="stMarkdownContainer"] li {
    color: var(--ink-soft);
    font-size: 0.865rem;
    line-height: 1.72;
}

[data-testid="stMarkdownContainer"] li {
    margin-bottom: 0.35rem;
}

[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMarkdownContainer"] strong,
[data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"] [data-testid="stMarkdownContainer"] strong {
    color: var(--ink);
    font-weight: 600;
}

[data-testid="stMarkdownContainer"] ol,
[data-testid="stMarkdownContainer"] ul {
    padding-left: 1.15rem;
}

[data-testid="stMarkdownContainer"] h3 {
    margin: 1.15rem 0 0.45rem;
    padding: 0;
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.08rem;
    font-weight: 500;
    letter-spacing: -0.005em;
    color: var(--ink);
}

[data-testid="stMarkdownContainer"] h3:first-child {
    margin-top: 0;
}

.kicker {
    color: var(--faint);
    font-size: 0.62rem;
    font-weight: 500;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}

.card-title {
    margin: 0.45rem 0 0.85rem;
    padding: 0;
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.32rem;
    font-weight: 500;
    letter-spacing: -0.01em;
    color: var(--ink);
}

/* Video */

.video-shell {
    overflow: hidden;
    border: 1px solid var(--line);
    border-radius: 6px;
    background: #ece7dc;
    box-shadow: none;
}

.source-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--faint);
    font-size: 0.73rem;
    line-height: 1.5;
}

.source-meta a {
    margin-left: auto;
    color: var(--ink);
    font-weight: 500;
    text-decoration: underline;
    text-underline-offset: 3px;
    text-decoration-color: var(--line-strong);
    border-bottom: 0;
}

.source-meta a:hover {
    text-decoration-color: var(--ink);
}

.live-dot {
    width: 5px;
    height: 5px;
    flex: 0 0 auto;
    border-radius: 50%;
    background: var(--accent);
}

.source-placeholder {
    display: flex;
    align-items: center;
    gap: 14px;
    min-height: 200px;
    padding: 1.6rem;
    border: 1px dashed var(--line-strong);
    border-radius: 6px;
    background: var(--paper);
}

.source-glyph {
    width: 38px;
    height: 38px;
    display: grid;
    place-items: center;
    flex: 0 0 auto;
    border-radius: 50%;
    background: var(--ink);
    color: var(--card);
}

.source-glyph svg {
    width: 15px;
    height: 15px;
    margin-left: 2px;
}

.source-placeholder strong {
    display: block;
    margin-bottom: 3px;
    color: var(--ink);
    font-family: 'Newsreader', Georgia, serif;
    font-size: 0.98rem;
    font-weight: 500;
}

.source-placeholder span {
    color: var(--muted);
    font-size: 0.75rem;
    line-height: 1.55;
}

/* Secondary buttons */

.stDownloadButton > button,
[data-testid="stDownloadButton"] button,
.stButton > button {
    min-height: 38px;
    border: 1px solid var(--line-strong) !important;
    border-radius: 6px !important;
    background: transparent !important;
    color: var(--ink) !important;
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    font-weight: 500;
    box-shadow: none !important;
    transition: background 0.15s ease, border-color 0.15s ease;
}

.stDownloadButton > button:hover,
[data-testid="stDownloadButton"] button:hover,
.stButton > button:hover {
    border-color: var(--ink) !important;
    background: rgba(27, 25, 23, 0.04) !important;
    color: var(--ink) !important;
}

.stDownloadButton > button:focus,
[data-testid="stDownloadButton"] button:focus,
.stButton > button:focus {
    box-shadow: 0 0 0 2px var(--paper), 0 0 0 3px var(--ink) !important;
}

/* Expanders and transcript */

[data-testid="stExpander"] {
    border: 1px solid var(--line);
    border-radius: 6px;
    background: transparent;
    overflow: hidden;
}

[data-testid="stExpander"] details {
    border: 0;
    background: transparent;
}

[data-testid="stExpander"] summary {
    padding: 0.85rem 1rem;
    color: var(--ink-soft);
    font-size: 0.82rem;
    font-weight: 500;
}

[data-testid="stExpander"] summary:hover {
    color: var(--ink);
}

[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    padding: 0 1rem 1rem;
}

.transcript-body {
    max-height: 360px;
    overflow-y: auto;
    padding: 1.05rem 1.2rem;
    border: 1px solid var(--line);
    border-radius: 4px;
    background: var(--card);
    color: var(--ink-soft);
    font-size: 0.82rem;
    line-height: 1.85;
    white-space: pre-wrap;
}

.transcript-body::-webkit-scrollbar {
    width: 8px;
}

.transcript-body::-webkit-scrollbar-track {
    background: transparent;
}

.transcript-body::-webkit-scrollbar-thumb {
    border-radius: 999px;
    background: var(--line-strong);
}

/* Section heading */

.section-heading {
    display: flex;
    flex-direction: column;
    justify-content: center;
    min-height: 54px;
}

.section-heading h2 {
    margin: 0.35rem 0 0;
    padding: 0;
    font-family: 'Newsreader', Georgia, serif;
    font-size: clamp(1.65rem, 2.8vw, 2.15rem);
    font-weight: 400;
    letter-spacing: -0.015em;
    color: var(--ink);
}

.section-heading p {
    margin: 0.4rem 0 0;
    color: var(--muted);
    font-size: 0.82rem;
    line-height: 1.6;
}

/* Chat */

.chat-empty {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    margin: 0.7rem 0 1rem;
    padding: 0 0 0 0.95rem;
    border: 0;
    border-left: 2px solid var(--line-strong);
    border-radius: 0;
    background: none;
}

.chat-empty-mark {
    width: 5px;
    height: 5px;
    flex: 0 0 auto;
    margin-top: 8px;
    border-radius: 50%;
    background: var(--accent);
}

.chat-empty strong {
    display: block;
    margin-bottom: 3px;
    color: var(--ink);
    font-size: 0.83rem;
    font-weight: 600;
}

.chat-empty p {
    margin: 0;
    color: var(--muted);
    font-size: 0.77rem;
    line-height: 1.6;
}

[data-testid="stChatMessage"] {
    margin-bottom: 0.4rem;
    padding: 0.8rem 1rem;
    border: 1px solid var(--line);
    border-radius: 6px;
    background: var(--card);
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]),
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    border-color: #e4dccb;
    background: #efe9dc;
}

[data-testid="stChatMessageContent"] p {
    color: var(--ink-soft);
    font-size: 0.875rem;
    line-height: 1.72;
}

[data-testid="stChatInput"] {
    border: 1px solid var(--line-strong);
    border-radius: 8px;
    background: var(--card);
    box-shadow: 0 6px 20px rgba(27, 25, 23, 0.05);
    backdrop-filter: none;
}

[data-testid="stChatInput"] textarea {
    color: var(--ink) !important;
    font-family: 'Inter', sans-serif;
    font-size: 0.88rem;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #8c8373 !important;
}

[data-testid="stChatInput"] button {
    color: var(--muted);
}

[data-testid="stChatInput"] button:hover {
    color: var(--ink);
}

/* Sources */

.source-snippet {
    margin-bottom: 0.6rem;
    padding: 0.75rem 0.95rem;
    border-left: 2px solid var(--accent);
    border-radius: 0;
    background: var(--paper);
}

.source-snippet strong {
    display: block;
    margin-bottom: 0.3rem;
    color: var(--accent);
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}

.source-snippet p {
    margin: 0;
    color: var(--ink-soft);
    font-size: 0.78rem;
    line-height: 1.7;
    white-space: pre-wrap;
}

/* Status + progress */

[data-testid="stStatus"] {
    border: 1px solid var(--line);
    border-radius: 6px;
    background: var(--card);
}

[data-testid="stProgress"] div[role="progressbar"] > div > div {
    background: var(--ink) !important;
}

/* Footer */

.app-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
    margin-top: 3.4rem;
    padding-top: 1.1rem;
    border-top: 1px solid var(--line);
    color: var(--faint);
    font-size: 0.7rem;
    line-height: 1.7;
}

.app-footer b {
    color: var(--ink-soft);
    font-weight: 500;
}

@media (max-width: 900px) {
    .block-container {
        padding: 1.2rem 1.1rem 6rem;
    }

    .topbar {
        margin-bottom: 0;
    }

    .brand-name span {
        display: none;
    }

    .top-pill {
        display: none;
    }

    .hero {
        padding-top: 2.4rem;
    }

    .hero h1 {
        font-size: clamp(2.15rem, 9vw, 3rem);
    }

    .hero.compact h1 {
        font-size: clamp(1.9rem, 7.5vw, 2.5rem);
    }

    .feature-grid {
        grid-template-columns: 1fr;
        gap: 1.7rem;
        margin-top: 2.6rem;
    }

    [data-testid="stForm"] {
        padding-top: 1.2rem;
    }

    [data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"] {
        padding: 1.05rem;
    }

    .app-footer {
        align-items: flex-start;
        flex-direction: column;
    }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation: none !important;
        transition: none !important;
    }
}
</style>
"""

EMPTY_FEATURES = """
<div class="feature-grid">
    <article class="feature-card">
        <span class="feature-index">01</span>
        <h3>Read less, know more</h3>
        <p>A structured brief with the overview, key points, and takeaways pulled from the full recording.</p>
    </article>
    <article class="feature-card">
        <span class="feature-index">02</span>
        <h3>Signal, not noise</h3>
        <p>Filler, sponsor breaks, and repetition are stripped away so only the useful parts remain.</p>
    </article>
    <article class="feature-card">
        <span class="feature-index">03</span>
        <h3>Ask the source</h3>
        <p>Ask anything and get a complete-sentence answer with the transcript excerpts behind it.</p>
    </article>
</div>
"""

SUGGESTIONS = (
    "What is the main takeaway?",
    "What are the most important facts?",
    "What did the speaker leave unresolved?",
)


def get_services():
    load_dotenv(os.path.join(main.BASE_DIR, ".env"))
    groq_api_key = os.getenv("GROQ_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not groq_api_key or not openrouter_api_key:
        raise RuntimeError(
            "Missing GROQ_API_KEY or OPENROUTER_API_KEY. Add them to the .env file."
        )
    groq_client = Groq(
        api_key=groq_api_key,
        timeout=300.0,
        max_retries=3,
    )
    return (
        groq_client,
        build_llm(openrouter_api_key),
        build_embeddings(openrouter_api_key),
    )


def get_cached_services():
    return _cached_services()


@st.cache_resource(show_spinner=False)
def _cached_services():
    return get_services()


def init_state() -> None:
    defaults = {
        "analysis": None,
        "messages": [],
        "pending_question": None,
        "form_revision": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def normalize_video_url(value: str) -> str | None:
    value = value.strip()
    if not value:
        return None
    if not re.match(r"^https?://", value, re.IGNORECASE):
        value = f"https://{value}"
    parsed = urlparse(value)
    if parsed.scheme.lower() not in {"http", "https"}:
        return None
    if not parsed.netloc or "." not in parsed.netloc:
        return None
    return value


def youtube_video_id(url: str) -> str | None:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    candidate = None
    if host == "youtu.be" or host.endswith(".youtu.be"):
        candidate = parsed.path.strip("/").split("/")[0]
    elif "youtube.com" in host:
        path = parsed.path.strip("/")
        if path.startswith(("shorts/", "embed/", "live/")):
            candidate = path.split("/")[1]
        else:
            candidate = parse_qs(parsed.query).get("v", [None])[0]
    if candidate and re.fullmatch(r"[A-Za-z0-9_-]{6,20}", candidate):
        return candidate
    return None


def source_host(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def render_topbar() -> None:
    st.markdown(
        """
        <div class="topbar">
            <div class="brand-lockup">
                <div class="brand-mark">
                    <svg viewBox="0 0 24 24" fill="none">
                        <circle cx="12" cy="12" r="9.25" stroke="white" stroke-opacity="0.36" stroke-width="1.5"/>
                        <path d="M9.4 7.6v8.8L16.2 12 9.4 7.6Z" fill="white"/>
                    </svg>
                </div>
                <div class="brand-name">VidLens<span>Video intelligence</span></div>
            </div>
            <div class="top-pill"><span class="badge-dot"></span>Whisper + Nemotron</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(compact: bool) -> None:
    if compact:
        st.markdown(
            """
            <section class="hero compact">
                <div class="eyebrow"><span class="eyebrow-dot"></span>Analysis complete</div>
                <h1>Your video, <span class="accent">made legible.</span></h1>
                <p>Read the brief, explore the transcript, or ask a grounded question below.</p>
            </section>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        """
        <section class="hero">
            <div class="eyebrow"><span class="eyebrow-dot"></span>Video intelligence</div>
            <h1>Turn any video into <span class="accent">clear insight.</span></h1>
            <p>Paste a link. VidLens listens to every word, builds a structured brief, and answers your questions using the transcript as its only source.</p>
            <div class="hero-tags">
                <span class="hero-tag">YouTube &amp; Vimeo</span>
                <span class="hero-tag">Structured briefs</span>
                <span class="hero-tag">Transcript-grounded Q&amp;A</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_analyze_form(has_analysis: bool) -> None:
    revision = st.session_state.form_revision
    with st.form("video_form", clear_on_submit=False):
        url_col, action_col = st.columns([4.6, 1.4], gap="small", vertical_alignment="bottom")
        with url_col:
            field_label = "New video link" if has_analysis else "Video link"
            st.markdown(
                f'<div class="field-label">{field_label}</div>',
                unsafe_allow_html=True,
            )
            value = st.text_input(
                "Video link",
                key=f"video_url_{revision}",
                placeholder="Paste a YouTube, Shorts, Vimeo, or other video link",
                label_visibility="collapsed",
            )
        with action_col:
            submitted = st.form_submit_button(
                "Analyze video",
                type="primary",
                use_container_width=True,
            )

    st.markdown(
        '<div class="form-note">Public videos supported by yt-dlp. Longer recordings can take a few minutes.</div>',
        unsafe_allow_html=True,
    )

    if submitted:
        url = normalize_video_url(value)
        if not url:
            st.error("Enter a full video URL, such as https://www.youtube.com/watch?v=...")
            return
        analyze_video(url)


def analyze_video(url: str) -> None:
    try:
        groq_client, llm, embeddings = get_cached_services()
    except Exception as exc:
        st.error(str(exc))
        return

    started = time.perf_counter()
    try:
        with st.status("Analyzing your video", expanded=True) as status:
            progress = st.progress(0.04)

            status.write("Fetching the best available audio track...")
            with tempfile.TemporaryDirectory(prefix="video_agent_") as workdir:
                audio_file = download_audio(url, workdir)
                progress.progress(0.18)
                status.write("Audio ready. Transcribing with Whisper...")
                transcript = transcribe_audio(audio_file, groq_client)

            progress.progress(0.4)
            status.write(f"Transcript ready: {len(transcript):,} characters.")

            def on_progress(message: str) -> None:
                status.write(message)
                if message.startswith("Summarizing transcript"):
                    progress.progress(0.5)
                elif message.startswith("Summarizing section"):
                    match = re.search(r"(\d+)/(\d+)", message)
                    if match:
                        done = int(match.group(1))
                        total = max(1, int(match.group(2)))
                        progress.progress(0.48 + 0.34 * done / total)
                elif message.startswith("Combining section summaries"):
                    progress.progress(0.84)

            summary = summarize_transcript(
                transcript,
                llm,
                progress_callback=on_progress,
            )
            progress.progress(0.9)
            status.write("Building the semantic search index...")

            retriever = None
            index_error = None
            try:
                retriever = build_retriever(transcript, embeddings)
            except Exception as exc:
                index_error = str(exc)
                status.write("Search index unavailable. The brief and transcript are still ready.")

            progress.progress(1.0)
            elapsed = time.perf_counter() - started
            status.update(
                label=f"Analysis complete in {elapsed:.1f} seconds",
                state="complete",
                expanded=False,
            )
    except Exception as exc:
        st.error(f"Could not analyze that video: {exc}")
        return

    st.session_state.analysis = {
        "url": url,
        "summary": summary,
        "transcript": transcript,
        "retriever": retriever,
        "index_error": index_error,
        "elapsed": elapsed,
    }
    st.session_state.messages = []
    st.session_state.pending_question = None
    st.session_state.form_revision += 1
    st.toast("Your video brief is ready.")
    st.rerun()


def youtube_embed(url: str) -> str | None:
    video_id = youtube_video_id(url)
    if not video_id:
        return None
    return f"https://www.youtube-nocookie.com/embed/{video_id}"


def render_video(url: str) -> None:
    embed_url = youtube_embed(url)
    safe_url = html.escape(url, quote=True)
    if embed_url:
        components.html(
            f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    html, body {{
                        margin: 0;
                        height: 100%;
                        background: #f5f2ea;
                    }}
                    .frame {{
                        width: 100%;
                        height: 100%;
                        overflow: hidden;
                        border-radius: 5px;
                        background: #f5f2ea;
                    }}
                    iframe {{
                        display: block;
                        width: 100%;
                        height: 100%;
                        border: 0;
                    }}
                </style>
            </head>
            <body>
                <div class="frame">
                    <iframe
                        src="{embed_url}"
                        title="Source video"
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                        referrerpolicy="strict-origin-when-cross-origin"
                        allowfullscreen>
                    </iframe>
                </div>
            </body>
            </html>
            """,
            height=300,
        )
    else:
        st.markdown(
            f"""
            <div class="source-placeholder">
                <div class="source-glyph">
                    <svg viewBox="0 0 24 24" fill="currentColor">
                        <path d="M8 5.5v13l10-6.5L8 5.5Z"/>
                    </svg>
                </div>
                <div>
                    <strong>{html.escape(source_host(url))}</strong>
                    <span>Inline preview is available for YouTube links. The original source is still indexed and ready for questions.</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="source-meta">
            <span class="live-dot"></span>
            <span>{html.escape(source_host(url))}</span>
            <a href="{safe_url}" target="_blank" rel="noopener noreferrer">Open original</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_summary(summary: str) -> str:
    formatted = re.sub(
        r"(?im)^\s*(?:\d+[.)]\s*)?(Overview|Key points|Takeaways)\s*:?\s*",
        lambda match: f"### {match.group(1).title()}\n\n",
        summary.strip(),
    )
    return re.sub(r"\n{3,}", "\n\n", formatted).strip()


def scroll_to_chat() -> None:
    components.html(
        """
        <script>
            const target = window.parent.document.getElementById("chat-section");
            if (target) {
                target.scrollIntoView({behavior: "smooth", block: "start"});
            }
        </script>
        """,
        height=0,
    )


def render_sources(snippets: list[str]) -> None:
    if not snippets:
        return
    with st.expander(f"View transcript sources ({len(snippets)})"):
        for index, snippet in enumerate(snippets, start=1):
            body = snippet if len(snippet) <= 900 else f"{snippet[:900].rstrip()}..."
            st.markdown(
                f"""
                <div class="source-snippet">
                    <strong>Excerpt {index}</strong>
                    <p>{html.escape(body)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_analysis(analysis: dict) -> None:
    url = analysis["url"]
    summary = analysis["summary"]
    transcript = analysis["transcript"]
    retriever = analysis["retriever"]
    index_error = analysis["index_error"]
    words = len(transcript.split())
    read_time = max(1, round(words / 220))

    metric_cols = st.columns(3, gap="small")
    with metric_cols[0]:
        st.metric("Transcript words", f"{words:,}")
    with metric_cols[1]:
        st.metric("Read time", f"{read_time} min")
    with metric_cols[2]:
        st.metric("Analyzed in", f"{max(1, round(analysis['elapsed']))} sec")

    left, right = st.columns([0.94, 1.06], gap="large")
    with left:
        with st.container(border=True):
            st.markdown(
                '<div class="kicker">Source video</div><h3 class="card-title">Watch the original</h3>',
                unsafe_allow_html=True,
            )
            render_video(url)

    with right:
        with st.container(border=True):
            st.markdown(
                '<div class="kicker">Structured brief</div><h3 class="card-title">The short version</h3>',
                unsafe_allow_html=True,
            )
            st.markdown(format_summary(summary))
            st.download_button(
                "Download brief",
                data=summary,
                file_name="video-brief.md",
                mime="text/markdown",
                use_container_width=True,
            )

    with st.expander("Full transcript"):
        st.markdown(
            f'<div class="transcript-body">{html.escape(transcript)}</div>',
            unsafe_allow_html=True,
        )
        st.download_button(
            "Download transcript",
            data=transcript,
            file_name="video-transcript.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.markdown('<div id="chat-section" style="height: 1.9rem;"></div>', unsafe_allow_html=True)
    heading_col, action_col = st.columns([4.3, 1], gap="small", vertical_alignment="center")
    with heading_col:
        st.markdown(
            """
            <div class="section-heading">
                <div class="kicker">Transcript Q&amp;A</div>
                <h2>Ask the video</h2>
                <p>Answers stay grounded in what was actually said in the recording.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with action_col:
        if st.session_state.messages:
            if st.button("Clear chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    if retriever is None:
        st.warning(
            f"Question answering is unavailable for this video. {index_error or ''}".strip()
        )
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_sources(message.get("sources", []))

    prompt = st.chat_input("Ask anything about this video...")
    pending = st.session_state.pop("pending_question", None)
    question = prompt or pending

    if not question:
        if not st.session_state.messages:
            st.markdown(
                """
                <div class="chat-empty">
                    <span class="chat-empty-mark"></span>
                    <div>
                        <strong>Grounded in the transcript.</strong>
                        <p>Every answer is generated only from this video, and you can inspect the excerpts behind it.</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            suggestion_cols = st.columns(3, gap="small")
            for index, suggestion in enumerate(SUGGESTIONS):
                with suggestion_cols[index]:
                    if st.button(suggestion, key=f"suggestion_{index}", use_container_width=True):
                        st.session_state.pending_question = suggestion
                        st.rerun()
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            _, llm, _ = get_cached_services()
            with st.spinner("Searching the transcript..."):
                answer, docs = answer_question_with_docs(question, retriever, llm)
            snippets = [doc.page_content for doc in docs]
            st.markdown(answer)
            render_sources(snippets)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": snippets,
                }
            )
        except Exception as exc:
            st.error(f"Could not answer that question: {exc}")

    scroll_to_chat()


def render_footer() -> None:
    st.markdown(
        """
        <div class="app-footer">
            <span><b>VidLens</b> / Video intelligence without the noise.</span>
            <span>Whisper transcription &nbsp;·&nbsp; Nemotron summaries &nbsp;·&nbsp; Transcript-grounded retrieval</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


init_state()
st.markdown(APP_CSS, unsafe_allow_html=True)
render_topbar()

analysis = st.session_state.analysis
render_hero(compact=analysis is not None)
render_analyze_form(has_analysis=analysis is not None)

if analysis is None:
    st.markdown(EMPTY_FEATURES, unsafe_allow_html=True)
else:
    render_analysis(analysis)

render_footer()
