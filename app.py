import hashlib
import re

import streamlit as st
from rag.qa_chain import ask
from voice.speech_to_text import transcribe
from voice.text_to_speech import speak_to_bytes

# ── Query validator ────────────────────────────────────────────────────────
_FARMING_KEYWORDS = re.compile(
    r"crop|plant|seed|soil|fertilizer|fertilis|pesticide|pest|insect|worm|disease|"
    r"irrigat|water|rain|drought|harvest|yield|paddy|rice|wheat|maize|corn|cotton|"
    r"soybean|tomato|onion|potato|sugarcane|groundnut|pulses|vegetable|fruit|farm|"
    r"agricultur|kisan|pm.kisan|subsidy|scheme|government|organic|compost|manure|"
    r"spray|fungicide|herbicide|weed|nitrogen|phosphorus|potassium|npk|urea|"
    r"sowing|planting|pruning|tilling|tillage|blight|rot|mildew|aphid|thrip|"
    r"nematode|borers|armyworm|locust|cultivation|horticulture|floriculture",
    re.IGNORECASE,
)
_MIN_WORDS = 2
_MAX_WORDS = 300
_GIBBERISH = re.compile(r"^[^a-zA-Z0-9\s]{3,}$|^(.)\1{4,}$")


def _is_valid_query(text: str) -> tuple[bool, str]:
    """Return (is_valid, reason). Checks length, gibberish, and topic relevance."""
    text = text.strip()
    if not text:
        return False, "empty"
    words = text.split()
    if len(words) < _MIN_WORDS:
        return False, "too short"
    if len(words) > _MAX_WORDS:
        return False, "too long"
    if _GIBBERISH.search(text):
        return False, "gibberish"
    if not _FARMING_KEYWORDS.search(text):
        return False, "off-topic"
    return True, "ok"


_INVALID_RESPONSES = {
    "empty": "⚠️ Please type a question before sending.",
    "too short": "⚠️ Your query is too short. Please ask a complete farming-related question.",
    "too long": "⚠️ Your message is too long. Please shorten your question.",
    "gibberish": "❌ That doesn't look like a valid question. Please ask something about farming, crops, or agriculture.",
    "off-topic": "❌ **Invalid query.** I can only answer questions related to farming, crops, pests, fertilizers, irrigation, or government agricultural schemes like PM-KISAN. Please rephrase your question.",
}

st.set_page_config(page_title="Farmer Advisory Assistant", page_icon="🌾", layout="wide", initial_sidebar_state="collapsed")


def _seed_welcome_message():
    return [
        {
            "role": "assistant",
            "content": (
                "👋 Welcome! Ask me about crops, pests, fertilizer schedules, irrigation, "
                "or government schemes like PM-KISAN."
            ),
            "audio": None,
        }
    ]


if "messages" not in st.session_state:
    st.session_state.messages = _seed_welcome_message()

if "last_mic_hash" not in st.session_state:
    st.session_state.last_mic_hash = None

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "enable_tts" not in st.session_state:
    st.session_state.enable_tts = False

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Light"

_THEMES = {
    "Light": {
        "app_bg": "#f6f8fb",
        "topbar_bg": "linear-gradient(135deg, #14532d 0%, #166534 55%, #15803d 100%)",
        "topbar_text": "#f0fdf4",
        "topbar_subtext": "#dcfce7",
        "chat_bg": "#ffffff",
        "composer_bg": "#ffffff",
        "composer_text": "#1f2937",
        "composer_placeholder": "#6b7280",
        "composer_border": "#d1d5db",
    },
    "Dark": {
        "app_bg": "#0b1220",
        "topbar_bg": "linear-gradient(135deg, #0f3d2b 0%, #0f5132 55%, #0f766e 100%)",
        "topbar_text": "#ecfeff",
        "topbar_subtext": "#d1fae5",
        "chat_bg": "#121212",
        "composer_bg": "#1e1e1e",
        "composer_text": "#ffffff",
        "composer_placeholder": "#9aa0a6",
        "composer_border": "#2a2a2a",
    },
}

current_theme = _THEMES["Light"]
css = """
<style>
    :root {
        --chat-input-width: min(1100px, calc(100vw - 1.2rem));
        --composer-bottom: 0.65rem;
        --mic-size: 2.2rem;
        --green-600: #16a34a;
        --green-700: #15803d;
        --green-50: #f0fdf4;
        --radius-pill: 999px;
        --radius-card: 16px;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
        box-shadow: none !important;
        block-size: 0 !important;
    }

    [data-testid="stSidebarCollapsedControl"],
    button[title="Open sidebar"],
    button[title="Close sidebar"],
    button[aria-label="Open sidebar"],
    button[aria-label="Close sidebar"] {
        z-index: 1200 !important;
        position: fixed !important;
        inset-block-start: 0.55rem !important;
        inset-inline-start: 0.55rem !important;
        inline-size: 2.4rem !important;
        block-size: 2.4rem !important;
        min-inline-size: 44px !important;
        min-block-size: 44px !important;
        border-radius: 8px !important;
        background: #166534 !important;
        border: 1px solid rgba(255,255,255,0.22) !important;
        color: #ffffff !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        opacity: 1 !important;
        visibility: visible !important;
        pointer-events: auto !important;
    }

    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="stSidebarCollapsedControl"] svg,
    button[title="Open sidebar"] svg,
    button[title="Close sidebar"] svg {
        color: #ffffff !important;
        fill: #ffffff !important;
        stroke: #ffffff !important;
        opacity: 1 !important;
    }

    html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] {
        block-size: 100%;
    }
    [data-testid="stAppViewContainer"] { background: __APP_BG__; }

    .main .block-container {
        max-inline-size: 1120px;
        padding-block-start: 5.2rem;
        padding-block-end: 7rem;
        padding-inline: 1.2rem;
    }

    .chat-topbar {
        position: fixed;
        inset-block-start: 0;
        inset-inline-start: 2.8rem;
        inset-inline-end: 0;
        z-index: 999;
        background: __TOPBAR_BG__;
        padding: 0.82rem 1.5rem;
        color: __TOPBAR_TEXT__;
        box-shadow: 0 2px 16px rgba(15,61,43,0.28);
        display: flex;
        align-items: center;
        gap: 0.85rem;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        transition: inset-inline-start 0.3s ease, padding 0.25s ease;
    }

    [data-testid="stApp"]:has([data-testid="stSidebar"][aria-expanded="true"]) .chat-topbar {
        inset-inline-start: var(--sidebar-width, 336px);
        padding: 0.65rem 1.1rem;
    }
    [data-testid="stApp"]:has([data-testid="stSidebar"][aria-expanded="true"]) .chat-topbar h2 { font-size: 0.95rem; }
    [data-testid="stApp"]:has([data-testid="stSidebar"][aria-expanded="true"]) .chat-topbar .topbar-icon { font-size: 1.45rem; }
    [data-testid="stApp"]:has([data-testid="stSidebar"][aria-expanded="true"]) .chat-topbar p { font-size: 0.76rem; }

    .chat-topbar .topbar-icon { font-size: 2rem; line-height: 1; flex-shrink: 0; }
    .chat-topbar .topbar-text { flex: 1; min-inline-size: 0; }
    .chat-topbar h2 {
        margin: 0;
        font-size: 1.18rem;
        font-weight: 700;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .chat-topbar p {
        margin: 0.15rem 0 0 0;
        color: __TOPBAR_SUBTEXT__;
        font-size: 0.84rem;
        opacity: 0.92;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .topbar-status {
        display: flex;
        align-items: center;
        gap: 0.38rem;
        font-size: 0.78rem;
        color: __TOPBAR_SUBTEXT__;
        background: rgba(255,255,255,0.12);
        border-radius: var(--radius-pill);
        padding: 0.28rem 0.72rem;
        font-weight: 500;
        flex-shrink: 0;
        white-space: nowrap;
    }
    .topbar-status::before {
        content: "";
        display: inline-block;
        inline-size: 0.52rem;
        block-size: 0.52rem;
        border-radius: 50%;
        background: #4ade80;
        animation: pulse-dot 2s infinite;
    }
    @keyframes pulse-dot { 0%,100% { opacity:1; } 50% { opacity:0.4; } }

    [data-testid="stChatMessage"] {
        padding: 0.5rem 0.6rem !important;
        border-radius: var(--radius-card) !important;
        margin-block-end: 0.55rem !important;
        border: 1px solid transparent !important;
        transition: background 0.15s;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: #f0fdf4 !important;
        border-color: #bbf7d0 !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background: #ffffff !important;
        border-color: #e5e7eb !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    [data-testid="stChatMessageAvatarUser"] {
        background: linear-gradient(135deg, #16a34a, #15803d) !important;
        border-radius: 50% !important;
        color: #fff !important;
    }
    [data-testid="stChatMessageAvatarAssistant"] {
        background: linear-gradient(135deg, #065f46, #047857) !important;
        border-radius: 50% !important;
        color: #fff !important;
    }
    [data-testid="stChatMessage"] p { font-size: 0.97rem; line-height: 1.65; color: #111827; }

    div[data-testid="stBottomBlockContainer"] {
        padding-inline: 0.55rem;
        padding-block-end: var(--composer-bottom);
        background: linear-gradient(to top, rgba(246,248,251,0.98) 55%, transparent 100%);
    }

    div[data-testid="stChatInput"] {
        inline-size: 100%;
        max-inline-size: var(--chat-input-width);
        margin-inline: auto;
        padding-inline-start: 0.6rem;
    }
    div[data-testid="stChatInput"] > div {
        border-radius: 30px !important;
        border: 1.5px solid #d1fae5 !important;
        background: #ffffff !important;
        box-shadow: 0 4px 24px rgba(22,163,74,0.10), 0 1px 4px rgba(0,0,0,0.07) !important;
        padding-block: 0.35rem;
        transition: border-color 0.18s, box-shadow 0.18s;
    }
    div[data-testid="stChatInput"] > div:focus-within {
        border-color: #86efac !important;
        box-shadow: 0 0 0 3px rgba(34,197,94,0.13), 0 4px 20px rgba(22,163,74,0.12) !important;
    }
    div[data-testid="stChatInput"] textarea {
        color: #111827 !important;
        font-size: 1rem !important;
        padding-inline-start: 0.4rem !important;
        padding-block: 0.28rem !important;
        background: transparent !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder { color: #9ca3af !important; opacity: 1; }
    div[data-testid="stChatInput"] button {
        border-radius: var(--radius-pill) !important;
        inline-size: 2.6rem !important;
        block-size: 2.6rem !important;
        min-inline-size: 44px !important;
        min-block-size: 44px !important;
        background: linear-gradient(135deg, #16a34a, #15803d) !important;
        border: none !important;
        box-shadow: 0 2px 8px rgba(22,163,74,0.28) !important;
        transition: transform 0.12s, box-shadow 0.12s;
    }
    div[data-testid="stChatInput"] button:hover { transform: scale(1.06); }
    div[data-testid="stChatInput"] button svg { display: none !important; }
    div[data-testid="stChatInput"] button::before { content: ""; color: #fff; font-size: 1rem; }

    [data-testid="stSidebar"] {
        background: #f8fafc !important;
        border-inline-end: 1px solid #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        font-weight: 700;
        font-size: 1.05rem;
        color: #14532d;
        border-block-end: 2px solid #bbf7d0;
        padding-block-end: 0.35rem;
        margin-block-end: 0.6rem;
    }
    [data-testid="stSidebar"] .stMarkdown h4 {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #6b7280;
        margin-block-end: 0.4rem;
        margin-block-start: 1rem;
    }
    [data-testid="stSidebar"] button[kind="secondary"] {
        border-radius: 10px !important;
        border: 1px solid #dcfce7 !important;
        background: #f0fdf4 !important;
        color: #166534 !important;
        font-size: 0.84rem !important;
        font-weight: 500 !important;
        text-align: start !important;
        transition: background 0.14s, border-color 0.14s;
        padding-block: 0.6rem !important;
        min-block-size: 44px !important;
    }
    [data-testid="stSidebar"] button[kind="secondary"]:hover {
        background: #dcfce7 !important;
        border-color: #86efac !important;
    }
    [data-testid="stSidebar"] button[kind="primary"],
    [data-testid="stSidebar"] button[data-testid="baseButton-secondary"]:last-of-type {
        border-radius: 10px !important;
        background: linear-gradient(135deg, #16a34a, #15803d) !important;
        color: #fff !important;
        border: none !important;
        font-weight: 600 !important;
        min-block-size: 44px !important;
    }

    .sidebar-mic {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 12px;
        padding: 0.65rem 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .sidebar-mic div[data-testid="stAudioInput"] label { display: none !important; }
    .sidebar-mic button {
        inline-size: var(--mic-size) !important;
        block-size: var(--mic-size) !important;
        min-inline-size: 44px !important;
        min-block-size: 44px !important;
        padding: 0 !important;
        border-radius: var(--radius-pill) !important;
        border: 1px solid #86efac !important;
        background: linear-gradient(135deg,#dcfce7,#bbf7d0) !important;
    }
    .sidebar-mic button svg { display: none !important; }
    .sidebar-mic button::before { content: ""; font-size: 1.05rem; line-height: 1; }

    .stSpinner > div { border-block-start-color: var(--green-600) !important; }

    @media (max-width: 900px) {
        .main .block-container {
            padding-block-start: 4.8rem;
            padding-block-end: 7rem;
            padding-inline: 0.9rem;
        }
        .chat-topbar h2 { font-size: 1.05rem; }
        .chat-topbar p  { font-size: 0.78rem; }
    }

    @media (max-width: 600px) {
        :root {
            --chat-input-width: calc(100vw - 0.5rem);
            --composer-bottom: env(safe-area-inset-bottom, 0.5rem);
            --mic-size: 2.75rem;
        }
        .chat-topbar {
            inset-inline-start: 0 !important;
            padding: 0.6rem 0.75rem 0.6rem 3.4rem;
            gap: 0.5rem;
        }
        .chat-topbar .topbar-icon { font-size: 1.5rem; }
        .chat-topbar h2 { font-size: 0.92rem; }
        .chat-topbar p  { display: none; }
        .topbar-status  { display: none; }
        .main .block-container {
            max-inline-size: 100%;
            padding-block-start: 4rem;
            padding-block-end: 5.5rem;
            padding-inline: 0.4rem;
        }
        [data-testid="stChatMessage"] {
            padding: 0.45rem 0.5rem !important;
            border-radius: 12px !important;
            margin-block-end: 0.4rem !important;
        }
        [data-testid="stChatMessage"] p { font-size: 0.93rem; line-height: 1.6; }
        div[data-testid="stBottomBlockContainer"] {
            padding-inline: 0.3rem;
            padding-block-end: max(var(--composer-bottom), 0.4rem);
        }
        div[data-testid="stChatInput"] { padding-inline-start: 0.2rem; max-inline-size: 100%; }
        div[data-testid="stChatInput"] textarea { font-size: 0.97rem !important; }
        [data-testid="stSidebar"] {
            min-inline-size: 100vw !important;
            max-inline-size: 100vw !important;
        }
        [data-testid="stSidebar"] button[kind="secondary"] {
            font-size: 0.9rem !important;
            padding-block: 0.7rem !important;
        }
    }

    @media (max-width: 380px) {
        .chat-topbar h2 { font-size: 0.82rem; }
        [data-testid="stChatMessage"] p { font-size: 0.88rem; }
        div[data-testid="stChatInput"] textarea { font-size: 0.92rem !important; }
    }
</style>
"""

css = css.replace("__APP_BG__", current_theme["app_bg"])
css = css.replace("__TOPBAR_BG__", current_theme["topbar_bg"])
css = css.replace("__TOPBAR_TEXT__", current_theme["topbar_text"])
css = css.replace("__TOPBAR_SUBTEXT__", current_theme["topbar_subtext"])

st.markdown(css, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Farmer Assistant")
    st.caption("Chat-style advisory with voice input")

    st.session_state.enable_tts = st.toggle(
        "Read answers aloud",
        value=st.session_state.enable_tts,
    )

    st.markdown("#### Quick prompts")
    quick_questions = [
        "How to control fall armyworm in maize?",
        "Best fertilizer schedule for paddy",
        "How often should I irrigate tomato in summer?",
        "PM-KISAN eligibility and required documents",
    ]

    for idx, item in enumerate(quick_questions, start=1):
        if st.button(item, key=f"quick_{idx}", use_container_width=True):
            st.session_state.pending_prompt = item
            st.rerun()

    st.markdown("#### Voice query")
    st.markdown("<div class='sidebar-mic'>", unsafe_allow_html=True)
    mic_audio = st.audio_input(" ", key="mic_input")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("New chat", use_container_width=True):
        st.session_state.messages = _seed_welcome_message()
        st.session_state.pending_prompt = None
        st.session_state.last_mic_hash = None
        st.rerun()

st.markdown(
    """
    <div class="chat-topbar">
        <span class="topbar-icon"></span>
        <div class="topbar-text">
            <h2>AI Farmer Advisory Chat</h2>
            <p>Ask about crops, pests, fertilizer, irrigation &amp; govt. schemes</p>
        </div>
        <span class="topbar-status">Online</span>
    </div>
    """,
    unsafe_allow_html=True,
)

chat_container = st.container(border=False)
with chat_container:
    st.markdown("<div class='chat-scroll'>", unsafe_allow_html=True)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("audio"):
                st.audio(message["audio"], format="audio/mp3")
    st.markdown("</div>", unsafe_allow_html=True)

if mic_audio is not None:
    audio_bytes = mic_audio.getvalue()
    audio_hash = hashlib.md5(audio_bytes).hexdigest()
    if st.session_state.last_mic_hash != audio_hash:
        with st.spinner("Transcribing..."):
            try:
                transcribed = transcribe(mic_audio)
                st.session_state.last_mic_hash = audio_hash
                if transcribed:
                    st.session_state.pending_prompt = transcribed
                    st.rerun()
                else:
                    st.warning("Could not detect speech. Try again.")
            except Exception as exc:
                st.session_state.last_mic_hash = audio_hash
                st.error(f"Voice transcription failed: {exc}")

# st.chat_input is Enter-to-send by default and includes a built-in send button.
typed_prompt = st.chat_input("Ask your farming question...")

prompt = typed_prompt or st.session_state.pending_prompt
if prompt:
    st.session_state.pending_prompt = None

    st.session_state.messages.append({"role": "user", "content": prompt, "audio": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        valid, reason = _is_valid_query(prompt)
        if not valid:
            answer = _INVALID_RESPONSES.get(reason, _INVALID_RESPONSES["off-topic"])
        else:
            with st.spinner("Preparing advisory..."):
                answer = ask(prompt)

        final_answer = answer

        st.markdown(final_answer)

        answer_audio = None
        if st.session_state.enable_tts and final_answer:
            try:
                answer_audio = speak_to_bytes(final_answer, lang="en")
                st.audio(answer_audio, format="audio/mp3")
            except Exception as exc:
                st.warning(f"Could not generate voice output: {exc}")

    st.session_state.messages.append(
        {"role": "assistant", "content": final_answer, "audio": answer_audio}
    )
    st.rerun()
