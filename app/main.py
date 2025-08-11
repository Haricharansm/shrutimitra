# Streamlit: Shrutimitra (Vedic Chant Coach) – v2 (no soundfile)
# - Upload/record audio
# - Pitch + onsets, DTW alignment
# - Accent/timing feedback
# - Optional phoneme feedback using guru-provided phones + MFCC templates
import io
import numpy as np
import streamlit as st

from utils import (
    load_audio_mono_16k,
    extract_pitch_pyinnish,
    detect_onsets_times,
    resample_to_len,
    dtw_align_cost_path,
    segment_stats,
    compare_segments_and_score,
    draw_wave_and_pitch_with_segments,
    parse_phones_from_text,
    build_phone_templates_from_guru,
    classify_learner_phones,
    generate_natural_language_feedback
)

st.set_page_config(page_title="Shrutimitra — Vedic Chant Coach", layout="wide")

# --- Branding header ---
# --- Branding header (with Sanskrit wordmark) ---
from pathlib import Path
BRAND_DIR = Path(__file__).parent / "branding"
ASSETS = BRAND_DIR / "assets"

# Load CSS
with open(BRAND_DIR / "custom.css", "r", encoding="utf-8") as _f:
    st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)

def render_brand_header():
    col1, col2 = st.columns([0.15, 1])
    with col1:
        try:
            st.image(str(ASSETS / "shrutimitra_icon_256.png"), width=64)
        except Exception:
            st.write("🎧")  # fallback emoji
    with col2:
        try:
            st.image(str(ASSETS / "shrutimitra_wordmark.png"), use_container_width=False)
        except Exception:
            st.markdown("**श्रुतिमित्र**", unsafe_allow_html=True)
        st.markdown(
            '<div class="brand-title">Shrutimitra</div>'
            '<div class="brand-tagline">Perfect your Vedic chanting — accents, timing, and pronunciation.</div>',
            unsafe_allow_html=True
        )
        st.markdown('<div class="badge">MVP</div>', unsafe_allow_html=True)

render_brand_header()

st.title("🎧 Vedic Chant Coach")
st.caption("Upload a benchmark (Guru) chant and a learner recording, then get targeted feedback.")

with st.sidebar:
    st.header("Settings")
    sr = st.selectbox("Sample rate", [16000, 22050], index=0)
    f0_min = st.number_input("F0 min (Hz)", value=75, min_value=40, max_value=200)
    f0_max = st.number_input("F0 max (Hz)", value=400, min_value=150, max_value=800)
    smooth_ms = st.slider("Pitch smoothing (ms)", 0, 200, 50)
    onset_backtrack = st.checkbox("Backtrack onsets", value=True)
    onset_sensitivity = st.slider("Onset sensitivity", 0.0, 1.0, 0.3, 0.05)
    strictness = st.slider("Coach strictness", 0.0, 1.0, 0.5, 0.05)
    st.markdown("---")
    st.caption("Tip: WAV/OGG/FLAC recommended. MP3 may work if ffmpeg is available.")

# --- Guru inputs ---
st.subheader("① Guru / Benchmark")
gcol1, gcol2 = st.columns([1,1])
with gcol1:
    guru_file = st.file_uploader("Upload benchmark audio", type=["wav","ogg","flac"], key="guru")
with gcol2:
    st.write("Optional: phones per segment (one line per segment; space-separated if multiple).")
    phones_text = st.text_area("Paste phones (guru)", height=120, placeholder="e.g.\nśa\nṭa\nā\nna\nṣa")
phones_file = st.file_uploader("...or upload phones.txt", type=["txt"], key="phonesfile")
if phones_file and not phones_text:
    phones_text = phones_file.read().decode("utf-8", errors="ignore")

# --- Learner inputs ---
st.subheader("② Learner")
lcol1, lcol2 = st.columns([1,1])
with lcol1:
    learner_file = st.file_uploader("Upload learner audio", type=["wav","ogg","flac"], key="learner")
with lcol2:
    st.write("Or record in-browser:")
    try:
        from audio_recorder_streamlit import audio_recorder
        audio_bytes = audio_recorder(text="Click to record / stop", pause_threshold=2.0, sample_rate=sr)
        if audio_bytes:
            learner_file = io.BytesIO(audio_bytes)
            learner_file.name = "recorded.wav"
            st.success("Captured microphone audio.")
            st.audio(learner_file, format="audio/wav")
    except Exception:
        st.info("Install `audio-recorder-streamlit` locally for mic capture, or just upload a file.")

if guru_file and learner_file:
    guru_y, sr_out = load_audio_mono_16k(guru_file, target_sr=sr)
    user_y, _ = load_audio_mono_16k(learner_file, target_sr=sr)

    st.audio(guru_file, format="audio/wav")
    st.audio(learner_file, format="audio/wav")

    with st.spinner("Estimating pitch and onsets..."):
        f0_g, times_g = extract_pitch_pyinnish(guru_y, sr_out, f0_min, f0_max, smooth_ms=smooth_ms)
        f0_u, times_u = extract_pitch_pyinnish(user_y, sr_out, f0_min, f0_max, smooth_ms=smooth_ms)
        onsets_g = detect_onsets_times(guru_y, sr_out, backtrack=onset_backtrack, onset_sensitivity=onset_sensitivity)
        onsets_u = detect_onsets_times(user_y, sr_out, backtrack=onset_backtrack, onset_sensitivity=onset_sensitivity)

    with st.spinner("Aligning pitch contours (DTW)..."):
        L = max(len(f0_g), len(f0_u))
        f0_g_r = resample_to_len(f0_g, L)
        f0_u_r = resample_to_len(f0_u, L)
        _, _ = dtw_align_cost_path(f0_g_r, f0_u_r)

    segs_g = segment_stats(times_g, f0_g, onsets_g)
    segs_u = segment_stats(times_u, f0_u, onsets_u)

    report_basic, overall = compare_segments_and_score(segs_g, segs_u, strictness=strictness)

    # Optional phones
    phone_labels = parse_phones_from_text(phones_text) if phones_text else None
    phone_feedback = []
    if phone_labels and len(phone_labels) <= len(segs_g):
        with st.spinner("Building guru phone templates + classifying learner phones..."):
            templates = build_phone_templates_from_guru(guru_y, sr_out, onsets_g, phone_labels)
            phone_feedback = classify_learner_phones(user_y, sr_out, onsets_u, phone_labels, templates)

    st.subheader("③ Visualizations")
    st.markdown("**Guru (Benchmark)**")
    fig_g = draw_wave_and_pitch_with_segments(guru_y, sr_out, times_g, f0_g, segs_g, title="Guru")
    st.pyplot(fig_g, clear_figure=True)

    st.markdown("**Learner**")
    fig_u = draw_wave_and_pitch_with_segments(user_y, sr_out, times_u, f0_u, segs_u, title="Learner")
    st.pyplot(fig_u, clear_figure=True)

    st.subheader("④ Feedback")
    st.markdown(f"**Overall Score:** {overall['score']:.1f} / 100")
    st.progress(min(max(int(overall['score']), 0), 100))
    nl = generate_natural_language_feedback(report_basic, phone_feedback)
    st.write(nl)

    with st.expander("Detailed items"):
        st.write("**Accent/Timing**")
        if not report_basic:
            st.write("No major accent/timing deviations.")
        else:
            for r in report_basic:
                st.write(f"- {r['message']} (segment ~{r['time_sec_start']:.2f}–{r['time_sec_end']:.2f}s)")
        if phone_feedback:
            st.write("**Pronunciation**")
            for p in phone_feedback:
                st.write(f"- {p['message']} (segment ~{p['time_sec_start']:.2f}–{p['time_sec_end']:.2f}s)")

    data = dict(overall=overall, accent_timing=report_basic, phoneme=phone_feedback)
    st.download_button("Download feedback (JSON)", data=io.BytesIO(bytes(str(data), "utf-8")),
                       file_name="shrutimitra_feedback.json", mime="application/json")
else:
    st.info("Upload a Guru audio and provide a Learner audio (upload or record) to begin.")
