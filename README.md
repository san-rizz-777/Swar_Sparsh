# Swar-Sparsh Pro (स्वर-स्पर्श)

A context-aware, vernacular assistive communication platform for individuals with
severe motor/speech impairments (ALS, Cerebral Palsy, post-stroke paralysis/aphasia).

Minimal switch keypresses or micro-utterances are expanded by **Sarvam AI's sovereign
Indic stack** into natural, first-person spoken sentences for caregivers:

| Stage | Model | Role |
|---|---|---|
| 1 | **Sarvam Saaras** (`saaras:v4`) | Transcribes short, accented, code-mixed micro-utterances |
| 2 | **Sarvam LLM** (`sarvam-30b`) | Expands a 1-word intent into a polite, context-aware sentence (time of day, urgency) |
| 3 | **Sarvam Bulbul** (`bulbul:v3`) | Synthesises natural Indic speech (Hindi, Tamil, Telugu, Bengali, Kannada, ...) |

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Sarvam API key (https://dashboard.sarvam.ai)
export SARVAM_API_KEY="your-key-here"

# 3. Launch
streamlit run app.py
```

The app opens at `http://localhost:8501`. You can also paste the API key directly
into the sidebar instead of using the environment variable.

> No API key? Enable **Offline demo mode** in the sidebar — the full UI flow works
> with canned expansions and a placeholder tone, ideal for dry-runs before a demo.

## Using the app

- **Switch Access tab** — three large buttons emulating adaptive switches (KEY 1–3,
  mapped to "पानी", "बाथरूम", "मदद" by default; editable in the sidebar).
- **Micro-utterance tab** — record a single spoken word; Saaras transcribes it,
  then the pipeline expands and speaks it.
- **Direct Text tab** — type an intent fragment directly.
- **Caregiver Log tab** — persistent request history with CSV export.

Set the target language, Bulbul speaker, speech pace and urgency level in the
sidebar. The context engine automatically injects time-of-day context.

## Deploying to Streamlit Community Cloud

1. Push `app.py` and `requirements.txt` to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) → *New app* → select the repo.
3. Add your API key under **Settings → Secrets**:
   ```toml
   SARVAM_API_KEY = "your-key-here"
   ```
   (Never commit the key to the repo.)
4. Deploy — every push to the branch auto-redeploys.

## Config knobs (top of `app.py`)

```python
STT_MODEL = "saaras:v4"
LLM_MODEL = "sarvam-30b"
TTS_MODEL = "bulbul:v3"
```

## Troubleshooting

**`'NoneType' object has no attribute 'strip'` (Sarvam LLM call failed)**
`sarvam-105b` is a reasoning model that defaults to `reasoning_effort="medium"`.
Reasoning tokens are billed against `max_tokens`, so with a small token budget
the model could burn its entire budget on hidden reasoning and return
`content: null` in the response — which crashed the old `.strip()` call. This
is fixed by explicitly sending `"reasoning_effort": None` (disables reasoning)
and raising `max_tokens` to 400. If you still see the error, it now surfaces
as a clear `RuntimeError` instead of a crash, telling you the `finish_reason`.

## Notes

- The LLM model id is configurable — if your Sarvam account exposes the chat
  model under a different name, change `LLM_MODEL`.
- All audio output is 22.05 kHz WAV, auto-played for the caregiver.
- The request log lives in the session; use the CSV export for records.
