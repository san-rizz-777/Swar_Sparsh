"""
Swar-Sparsh Pro (स्वर-स्पर्श) — Contextual Assistive Communication Platform
==========================================================================

An event-driven, vernacular AAC (Augmentative and Alternative Communication)
system for individuals with severe motor / speech impairments (ALS, Cerebral
Palsy, post-stroke paralysis/aphasia).

Pipeline:
    Input (switch keypress / micro-utterance)
        -> Context Engine (time of day, urgency, patient state)
        -> Sarvam Saaras  (saaras:v4)        : speech-to-text for micro-utterances
        -> Sarvam LLM     (sarvam-30b)       : expand 1-word intent into a polite,
                                               first-person, context-aware sentence
        -> Sarvam Bulbul  (bulbul:v3)         : natural Indic speech synthesis (WAV)
        -> Caregiver interface               : auto-playing audio + persistent log

Run locally:
    export SARVAM_API_KEY="your-key"        # or paste it in the sidebar
    streamlit run app.py

Dependencies: streamlit, requests  (see requirements.txt)
"""

import base64
import csv
import datetime
import io
import math
import os
import struct
import time
import wave

import requests
import streamlit as st

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="Swar-Sparsh Pro — स्वर-स्पर्श",
    page_icon="🎧",
    layout="wide",
)

API_BASE = "https://api.sarvam.ai"

STT_MODEL = "saaras:v4"       # Sarvam Saaras speech-to-text
LLM_MODEL = "sarvam-30b"      # Sarvam LLM (contextual expansion)
TTS_MODEL = "bulbul:v3"       # Sarvam Bulbul text-to-speech

LANGUAGES = {
    "Hindi (हिन्दी)": "hi-IN",
    "Tamil (தமிழ்)": "ta-IN",
    "Telugu (తెలుగు)": "te-IN",
    "Bengali (বাংলা)": "bn-IN",
    "Kannada (ಕನ್ನಡ)": "kn-IN",
    "Marathi (मराठी)": "mr-IN",
    "Gujarati (ગુજરાતી)": "gu-IN",
    "English (India)": "en-IN",
}

# Bulbul speaker catalogue (name -> voice persona)
SPEAKERS = ["Anushka", "Abhilash", "Manisha", "Vidya", "Arya", "Karun", "Hitesh"]

# Default switch-intent map (KEY 1-3). Fully editable from the sidebar.
DEFAULT_INTENTS = {
    "KEY 1": "पानी",        # water
    "KEY 2": "बाथरूम",     # bathroom
    "KEY 3": "मदद",        # help / emergency
}

URGENCY_LEVELS = ["Normal", "Urgent", "EMERGENCY"]

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------

if "log" not in st.session_state:
    st.session_state["log"] = []
if "last_audio" not in st.session_state:
    st.session_state["last_audio"] = None
if "last_sentence" not in st.session_state:
    st.session_state["last_sentence"] = ""


# ----------------------------------------------------------------------------
# Helpers — API key & context engine
# ----------------------------------------------------------------------------

def get_api_key() -> str:
    """Resolve the Sarvam API key: sidebar input > st.secrets > environment."""
    key = st.session_state.get("api_key", "") or ""
    if key:
        return key.strip()
    try:
        key = st.secrets.get("SARVAM_API_KEY", "")
        if key:
            return key.strip()
    except Exception:
        pass
    return os.environ.get("SARVAM_API_KEY", "").strip()


def time_context() -> str:
    """Context Engine: derive a natural-language time-of-day descriptor."""
    h = datetime.datetime.now().hour
    if 5 <= h < 12:
        return "early morning (before noon)"
    if 12 <= h < 17:
        return "afternoon (post-lunch)"
    if 17 <= h < 21:
        return "evening"
    return "night (late hours)"


# ----------------------------------------------------------------------------
# Pipeline stage 1 — Sarvam Saaras (speech-to-text)
# ----------------------------------------------------------------------------

def saaras_transcribe(audio_bytes: bytes, language_code: str, api_key: str) -> str:
    """Transcribe a short, heavily accented micro-utterance via Saaras."""
    files = {"file": ("utterance.wav", audio_bytes, "audio/wav")}
    data = {
        "model": STT_MODEL,
        "language_code": language_code,
        "with_pipeline": "false",
    }
    r = requests.post(
        f"{API_BASE}/speech-to-text",
        headers={"api-subscription-key": api_key},
        files=files,
        data=data,
        timeout=60,
    )
    r.raise_for_status()
    res = r.json()
    # Saaras returns a list of candidate transcripts
    transcripts = res.get("transcripts") or []
    if transcripts:
        return transcripts[0].strip()
    return (res.get("transcript") or "").strip()


# ----------------------------------------------------------------------------
# Pipeline stage 2 — Sarvam LLM (contextual expansion)
# ----------------------------------------------------------------------------

def expand_intent(fragment: str, language_code: str, language_name: str,
                  urgency: str, api_key: str) -> str:
    """Expand a 1-word / fragment intent into one natural first-person
    sentence in the target Indic language, conditioned on context."""
    system_prompt = (
        "You are Swar-Sparsh Pro, an assistive communication assistant for a person "
        "with a severe speech and motor impairment (e.g. ALS, cerebral palsy, "
        "post-stroke paralysis). The user can only produce a single word or a very "
        "short fragment. Your job is to convert it into exactly ONE natural, polite, "
        "first-person sentence in {lang} that the user would say to a caregiver or "
        "family member.\n"
        "Context you MUST respect:\n"
        "- Time of day: {tod}\n"
        "- Urgency level: {urgency}\n"
        "Rules:\n"
        "1. Output ONLY the final sentence. No quotes, no explanations, no romanisation.\n"
        "2. Keep it short (max 20 words), warm and natural.\n"
        "3. If urgency is EMERGENCY, begin with an urgent appeal for immediate help.\n"
        "4. If urgency is Urgent, convey the need firmly but politely.\n"
        "5. If the fragment is ambiguous, choose the most likely everyday need."
    ).format(lang=language_name, tod=time_context(), urgency=urgency)

    r = requests.post(
        f"{API_BASE}/v1/chat/completions",
        headers={
            "api-subscription-key": api_key,
            "Authorization": f"Bearer {api_key}",
        },
        json={
            "model": st.session_state.get("llm_model", LLM_MODEL),
            "temperature": 0.4,
            "max_tokens": 400,
            # sarvam-105b is a reasoning model and defaults to
            # reasoning_effort="medium". Reasoning tokens are billed against
            # max_tokens, so with a small budget the model can burn the
            # whole thing on hidden reasoning and return content=None,
            # which is exactly what caused the "'NoneType' object has no
            # attribute 'strip'" crash. Explicitly disabling reasoning
            # (per Sarvam docs: "Can be disabled by explicitly setting to
            # None") guarantees a populated `content` field.
            "reasoning_effort": None,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": fragment},
            ],
        },
        timeout=60,
    )
    if not r.ok:
        raise RuntimeError(f"Sarvam LLM returned HTTP {r.status_code}: {r.text[:300]}")

    res = r.json()
    choices = res.get("choices") or []
    if not choices:
        raise RuntimeError(f"Sarvam LLM returned no choices: {res}")

    message = choices[0].get("message") or {}
    content = message.get("content")

    if not content:
        # Defensive fallback: some reasoning-capable models can still put
        # text in reasoning_content, or return an empty string on a
        # finish_reason of "length". Surface a clear, actionable error
        # instead of crashing with an AttributeError on None.
        finish_reason = choices[0].get("finish_reason")
        reasoning_content = (message.get("reasoning_content") or "").strip()
        if reasoning_content:
            return reasoning_content.splitlines()[-1].strip()
        raise RuntimeError(
            "Sarvam LLM returned an empty response "
            f"(finish_reason={finish_reason}). Try again, or increase "
            "max_tokens if this keeps happening."
        )

    return content.strip()


# ----------------------------------------------------------------------------
# Pipeline stage 3 — Sarvam Bulbul (text-to-speech)
# ----------------------------------------------------------------------------

def bulbul_speak(text: str, language_code: str, speaker: str,
                 api_key: str, pace: float = 1.0, pitch: float = 0.0) -> bytes:
    """Synthesise natural Indic speech; returns WAV bytes."""
    payload = {
        "inputs": [text],
        "target_language_code": language_code,
        "speaker": speaker,
        "model": TTS_MODEL,
        "pitch": pitch,
        "pace": pace,
        "loudness": 1.0,
        "speech_sample_rate": 22050,
        "enable_preprocessing": True,
    }
    r = requests.post(
        f"{API_BASE}/text-to-speech",
        headers={"api-subscription-key": api_key},
        json=payload,
        timeout=60,
    )
    r.raise_for_status()
    audio_b64 = r.json()["audios"][0]
    return base64.b64decode(audio_b64)


# ----------------------------------------------------------------------------
# Offline demo mode (works without an API key — for dry-runs and demos)
# ----------------------------------------------------------------------------

def demo_expand(fragment: str, urgency: str) -> str:
    tod = time_context()
    tod_hi = {
        "early morning (before noon)": "सुबह-सुबह",
        "afternoon (post-lunch)": "दोपहर में",
        "evening": "शाम को",
        "night (late hours)": "रात में",
    }.get(tod, "")
    if urgency == "EMERGENCY":
        return f"कृपया तुरंत मदद कीजिए, मुझे {fragment} की बहुत ज़रूरत है!"
    if urgency == "Urgent":
        return f"क्या आप {tod_hi} मुझे {fragment} दे देंगे? जल्दी हो तो अच्छा होगा।"
    return f"कृपया {tod_hi} मुझे {fragment} दे दीजिए, बहुत कृपया।"


def demo_tone(seconds: float = 0.5, freq: int = 440) -> bytes:
    """Generate a soft placeholder tone as WAV (demo mode only)."""
    rate = 22050
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        for i in range(int(rate * seconds)):
            sample = int(12000 * math.sin(2 * math.pi * freq * i / rate))
            w.writeframes(struct.pack("<h", sample))
    return buf.getvalue()


# ----------------------------------------------------------------------------
# End-to-end pipeline runner
# ----------------------------------------------------------------------------

def run_pipeline(source: str, fragment: str):
    """source: 'switch' | 'voice' | 'text' — runs the full pipeline and
    plays the resulting audio to the caregiver."""
    api_key = get_api_key()
    demo = st.session_state.get("demo_mode", False) or not api_key

    lang_label = st.session_state.get("language", "Hindi (हिन्दी)")
    language_code = LANGUAGES[lang_label]
    language_name = lang_label.split(" (")[0]
    speaker = st.session_state.get("speaker", "Anushka")
    pace = st.session_state.get("pace", 1.0)
    urgency = st.session_state.get("urgency", "Normal")

    t0 = time.time()
    status = st.status(f"🎙️ Expanding intent “{fragment}” ...", expanded=True)

    # ---- Stage 2: contextual expansion (Sarvam LLM) --------------------
    if demo:
        if not api_key:
            status.update(label="⚠️ No API key found — running in OFFLINE DEMO mode.", state="warning")
        sentence = demo_expand(fragment, urgency)
        time.sleep(0.2)
    else:
        try:
            sentence = expand_intent(fragment, language_code, language_name,
                                      urgency, api_key)
        except Exception as e:
            status.update(label=f"❌ Sarvam LLM call failed: {e}", state="error")
            return

    # ---- Stage 3: speech synthesis (Bulbul) ------------------------------
    if demo:
        audio = demo_tone()
    else:
        try:
            audio = bulbul_speak(sentence, language_code, speaker,
                                 api_key, pace=pace)
        except Exception as e:
            status.update(label=f"❌ Bulbul TTS call failed: {e}", state="error")
            return

    latency = time.time() - t0
    status.update(label=f"✅ Ready in {latency:.1f}s", state="complete")

    # ---- Caregiver interface: autoplay + log ------------------------------
    st.markdown(f"### 🗣️ “{sentence}”")
    st.audio(audio, format="audio/wav", autoplay=True)

    st.session_state["last_sentence"] = sentence
    st.session_state["last_audio"] = audio
    st.session_state["log"].insert(0, {
        "Time": datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "Source": source,
        "Raw input": fragment,
        "Expanded sentence": sentence,
        "Language": language_code,
        "Urgency": urgency,
        "Latency (s)": f"{latency:.1f}",
    })


# ----------------------------------------------------------------------------
# UI — Sidebar (settings)
# ----------------------------------------------------------------------------

st.sidebar.title("⚙️ Swar-Sparsh Pro")
st.sidebar.caption("स्वर-स्पर्श — contextual assistive communication")

with st.sidebar.expander("🔐 API & Connection", expanded=True):
    st.session_state["api_key"] = st.text_input(
        "Sarvam API key", type="password",
        help="Leave empty to use the SARVAM_API_KEY environment variable.")
    st.session_state["demo_mode"] = st.checkbox(
        "Offline demo mode (no API calls)",
        help="Runs the pipeline with canned expansions and a placeholder tone.")

with st.sidebar.expander("🌐 Language & Voice"):
    st.session_state["language"] = st.selectbox("Target language", list(LANGUAGES))
    st.session_state["speaker"] = st.selectbox("Bulbul speaker", SPEAKERS)
    st.session_state["pace"] = st.slider("Speech pace", 0.5, 2.0, 1.0, 0.1)

with st.sidebar.expander("⚡ Urgency level"):
    st.session_state["urgency"] = st.radio("Current urgency", URGENCY_LEVELS,
                                           horizontal=True)

with st.sidebar.expander("🎛️ Switch intents (KEY 1–3)"):
    for key in DEFAULT_INTENTS:
        st.session_state[f"intent_{key}"] = st.text_input(
            key, value=st.session_state.get(f"intent_{key}", DEFAULT_INTENTS[key]))

st.sidebar.markdown(f"🕐 Context Engine → **{time_context()}**")

# ----------------------------------------------------------------------------
# UI — Main panel
# ----------------------------------------------------------------------------

st.title("Swar-Sparsh Pro — स्वर-स्पर्श")
st.markdown(
    "Turning minimal switch keypresses and micro-utterances into natural, "
    "context-aware speech for caregivers — powered by Sarvam AI's sovereign "
    "Indic stack (**Saaras → Sarvam-30b → Bulbul**)."
)

tab_switch, tab_voice, tab_text, tab_log = st.tabs(
    ["🎛️ Switch Access", "🎙️ Micro-utterance", "⌨️ Direct Text", "📋 Caregiver Log"]
)

# ---- Tab 1: Switch access (Keys 1–3) ---------------------------------------
with tab_switch:
    st.subheader("Switch Access Mode (Keys 1–3)")
    st.markdown("Each big button emulates an adaptive switch. Keyboard also works: press 1, 2 or 3.")

    cols = st.columns(3)
    key_cols = list(DEFAULT_INTENTS.keys())
    for col, k, color in zip(cols, key_cols, ["primary", "secondary", "danger"]):
        with col:
            if st.button(
                f"{k}\n\n💬 “{st.session_state.get(f'intent_{k}', DEFAULT_INTENTS[k])}”",
                use_container_width=True, type=color,
            ):
                run_pipeline("switch", st.session_state.get(f"intent_{k}", DEFAULT_INTENTS[k]))

# ---- Tab 2: Micro-utterance (Saaras STT) -------------------------------------
with tab_voice:
    st.subheader("Micro-utterance Mode (Saaras STT)")
    st.markdown("Speak a single word or short fragment — Saaras transcribes it, "
                "then the pipeline expands it into a full sentence.")

    audio_value = None
    try:
        audio_value = st.audio_input("Record a micro-utterance", key="mic")
    except Exception:
        audio_value = None  # older Streamlit without st.audio_input

    uploaded = st.file_uploader(
        "...or upload a WAV/MP3 snippet", type=["wav", "mp3", "m4a"])

    source_audio = audio_value or uploaded
    if source_audio is not None and st.button("▶️ Process utterance", type="primary"):
        api_key = get_api_key()
        if not api_key and not st.session_state.get("demo_mode", False):
            st.error("Please provide a Sarvam API key (sidebar) or enable offline demo mode.")
        else:
            with st.status("🎙️ Saaras transcribing ...", expanded=True):
                if api_key and not st.session_state.get("demo_mode", False):
                    try:
                        fragment = saaras_transcribe(
                            source_audio.getvalue(),
                            LANGUAGES[st.session_state.get("language", "Hindi (हिन्दी)")],
                            api_key)
                    except Exception as e:
                        st.error(f"Saaras STT call failed: {e}")
                        fragment = ""
                else:
                    fragment = st.text_input(
                        "(Demo mode) Type the spoken word:", value="पानी")
            if fragment:
                st.info(f"Transcribed fragment: **{fragment}**")
                run_pipeline("voice", fragment)

# ---- Tab 3: Direct text ------------------------------------------------------
with tab_text:
    st.subheader("Direct Text / Caregiver Typing")
    typed = st.text_input("Enter an intent fragment or word",
                          value=st.session_state.get("typed", "दर्द"))
    if st.button("🔊 Speak it", type="primary"):
        if typed.strip():
            run_pipeline("text", typed.strip())

# ---- Tab 4: Caregiver log -----------------------------------------------------
with tab_log:
    st.subheader("Caregiver Log — patient request history")
    if not st.session_state["log"]:
        st.markdown("_No requests yet. Use any input mode above._")
    else:
        st.dataframe(st.session_state["log"], use_container_width=True)

        csv_buf = io.StringIO()
        writer = csv.DictWriter(csv_buf, fieldnames=list(st.session_state["log"][0].keys()))
        writer.writeheader()
        writer.writerows(st.session_state["log"])
        st.download_button(
            "⬇️ Download log as CSV",
            data=csv_buf.getvalue(),
            file_name=f"swar_sparsh_log_{datetime.date.today()}.csv",
            mime="text/csv",
        )
        if st.button("🗑️ Clear log"):
            st.session_state["log"] = []
            st.rerun()

# ----------------------------------------------------------------------------
st.divider()
st.caption(
    "Swar-Sparsh Pro — built entirely on Sarvam AI's sovereign Indic stack: "
    "Saaras (STT) · Sarvam-30b (LLM) · Bulbul (TTS). "
    "For users with ALS, Cerebral Palsy, post-stroke aphasia and other "
    "motor/speech impairments."
)
