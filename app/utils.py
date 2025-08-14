import io
import subprocess
import numpy as np
import librosa
import soundfile as sf
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from fastdtw import fastdtw


@dataclass
class Segment:
    t0: float
    t1: float
    dur: float
    f0_mean: float
    f0_slope: float
    accent: str
# ============ NEW: Vedic accent helpers & visualizations ============

def _safe_mean_pitch(f: np.ndarray) -> float:
    f = f[np.isfinite(f) & (f > 0)]
    return float(np.mean(f)) if f.size else 0.0

def _segment_pitch(times: np.ndarray, f0: np.ndarray, t0: float, t1: float) -> np.ndarray:
    mask = (times >= t0) & (times < t1)
    return f0[mask]

def _classify_vedic(mean_hz: float, slope: float, global_med: float) -> str:
    """
    Heuristic:
      - udātta   : mean above global median by > ~10% (raised)
      - anudātta : mean below global median by > ~10% (low)
      - svarita  : otherwise, especially if slope is falling
    """
    if mean_hz <= 0 or global_med <= 0:
        return "unknown"
    rel = (mean_hz - global_med) / global_med
    if rel > 0.10:
        return "udātta"
    if rel < -0.10:
        return "anudātta"
    # around the median – prefer svarita if falling slope
    return "svarita" if slope < -5.0 else "udātta" if slope > 5.0 else "anudātta"

def label_segments_vedic(segs: List[Segment], times: np.ndarray, f0: np.ndarray) -> List[str]:
    global_med = _safe_mean_pitch(f0)
    labels = []
    for s in segs:
        seg_f = _segment_pitch(times, f0, s.t0, s.t1)
        m = _safe_mean_pitch(seg_f)
        labels.append(_classify_vedic(m, s.f0_slope, global_med))
    return labels

def _cents(a_hz: float, b_hz: float) -> float:
    """Return pitch difference a vs b in cents (positive = a higher)."""
    if a_hz <= 0 or b_hz <= 0:
        return np.nan
    return float(1200.0 * np.log2(a_hz / b_hz))

def score_segment(guru: Segment, learner: Segment, guru_mean_hz: float, learner_mean_hz: float) -> Dict[str, float]:
    """
    Combines pitch mean (cents), slope/accent and duration into a 0-100 score.
    """
    cents_err = abs(_cents(learner_mean_hz, guru_mean_hz))
    # tolerances (can be tuned or tied to "strictness")
    cents_tol = 40.0      # ~ 1/3 semitone
    slope_tol = 8.0
    dur_rel = abs(learner.dur - guru.dur) / max(guru.dur, 1e-6)

    # sub-scores 0..1
    sp = max(0.0, 1.0 - cents_err / cents_tol)
    ss = max(0.0, 1.0 - min(abs(learner.f0_slope - guru.f0_slope), 2*slope_tol) / slope_tol)
    sd = max(0.0, 1.0 - min(dur_rel, 1.0))

    total = 100.0 * (0.5*sp + 0.3*ss + 0.2*sd)
    return dict(score=total, cents_err=cents_err, dur_err_pct=100.0*dur_rel)

def draw_overlay_guru_learner(times_g, f0_g, segs_g, times_u, f0_u, segs_u, verdicts=None, title="Guru vs Learner"):
    """
    Single figure with Guru & Learner pitch overlaid; optional colored
    vertical bands (green/yellow/red) per aligned segment index.
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 3.2))
    plt.plot(times_g, f0_g, alpha=0.9, label="Guru pitch (Hz)")
    plt.plot(times_u, f0_u, alpha=0.9, linestyle="--", label="Learner pitch (Hz)")
    N = min(len(segs_g), len(segs_u))
    for i in range(N):
        s = segs_g[i]
        color = None
        if verdicts and i < len(verdicts):
            v = verdicts[i]
            if v >= 85: color = (0.85, 1.0, 0.85, 0.3)   # light green
            elif v >= 65: color = (1.0, 1.0, 0.75, 0.3)  # light yellow
            else: color = (1.0, 0.85, 0.85, 0.35)        # light red
        if color:
            plt.axvspan(s.t0, s.t1, color=color, linewidth=0)
        else:
            plt.axvline(s.t0, alpha=0.15, linewidth=1)
    plt.title(title)
    plt.xlabel("Time (s)"); plt.ylabel("Pitch (Hz)")
    plt.legend(loc="upper right"); plt.tight_layout()
    return fig

def slice_audio_to_wav_bytes(y: np.ndarray, sr: int, t0: float, t1: float) -> io.BytesIO:
    """Return a BytesIO WAV for [t0, t1) seconds."""
    import wave
    y = y.copy()
    s0 = max(0, int(t0*sr)); s1 = max(s0+1, int(t1*sr))
    seg = y[s0:s1]
    # 20 ms fade to avoid clicks
    w = max(1, int(0.02*sr))
    if len(seg) >= 2*w:
        env = np.ones_like(seg)
        env[:w] = np.linspace(0,1,w); env[-w:] = np.linspace(1,0,w)
        seg = seg * env
    seg16 = np.clip(seg*32767, -32768, 32767).astype(np.int16)
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(seg16.tobytes())
    bio.seek(0)
    return bio
# ======================= END NEW HELPERS ============================


# --- Audio I/O (soundfile-first, Py3.13-safe) ---
def _read_all_bytes(file_like_or_path) -> bytes:
    if hasattr(file_like_or_path, "read"):
        return file_like_or_path.read()
    with open(file_like_or_path, "rb") as f:
        return f.read()

def _ffmpeg_decode_to_wav_bytes(raw_bytes: bytes, target_sr: int = 16000, mono: bool = True) -> bytes:
    """
    Use system ffmpeg to decode arbitrary audio to WAV in-memory.
    Requires `ffmpeg` to be available on PATH (install via packages.txt).
    """
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0",
        "-f", "wav",
    ]
    if mono:
        cmd += ["-ac", "1"]
    if target_sr:
        cmd += ["-ar", str(target_sr)]
    cmd += ["pipe:1"]

    proc = subprocess.run(cmd, input=raw_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"ffmpeg decode failed: {proc.stderr.decode('utf-8', errors='ignore')[:200]}")
    return proc.stdout

def load_audio_mono_16k(file_like, target_sr: int = 16000):
    """
    Robust loader that avoids audioread/aifc on Py3.13 by using soundfile.
    - Tries soundfile directly (WAV/FLAC/OGG supported via libsndfile).
    - If that fails (e.g., MP3), falls back to ffmpeg -> WAV bytes -> soundfile.
    Returns mono float32 audio normalized to [-1, 1] at `target_sr`.
    """
    raw = _read_all_bytes(file_like)

    # 1) Try soundfile directly
    try:
        y, sr = sf.read(io.BytesIO(raw), always_2d=False)
    except Exception:
        # 2) Fallback: ffmpeg decode to WAV, then read with soundfile
        wav_bytes = _ffmpeg_decode_to_wav_bytes(raw, target_sr=target_sr, mono=True)
        y, sr = sf.read(io.BytesIO(wav_bytes), always_2d=False)

    # to mono
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # resample if needed (ffmpeg path already did it, but sf path might not)
    if sr != target_sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # light normalization
    if y.size and np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    return y.astype(np.float32), sr


# --- Pitch & onsets ---
def moving_average(x, w):
    if w <= 1: return x
    return np.convolve(x, np.ones(w)/w, mode='same')

def extract_pitch_pyinnish(y, sr, fmin=75, fmax=400, frame_length=2048, hop_length=256, smooth_ms=50):
    f0, _, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048, hop_length=hop_length)
    f0 = np.nan_to_num(f0)
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    if smooth_ms and smooth_ms > 0:
        win = max(1, int((smooth_ms/1000.0) * (sr / hop_length)))
        f0 = moving_average(f0, win)
    return f0.astype(np.float32), times.astype(np.float32)

def detect_onsets_times(y, sr, backtrack=True, onset_sensitivity=0.3):
    o_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(
        onset_envelope=o_env, sr=sr, backtrack=backtrack, units='time',
        pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=onset_sensitivity
    )
    times = [0.0] + list(onsets)
    if len(y) > 0:
        times.append(len(y)/sr)
    times = sorted(set([float(t) for t in times if t >= 0.0]))
    return np.array(times, dtype=np.float32)

def resample_to_len(x, L):
    if len(x) == L: return x
    xp = np.linspace(0, 1, len(x))
    x_newp = np.linspace(0, 1, L)
    return np.interp(x_newp, xp, x).astype(np.float32)

def dtw_align_cost_path(a, b):
    distance, path = fastdtw(a, b, dist=lambda x, y: abs(float(x) - float(y)))
    return distance, path


# --- Segments & scoring ---
def _accent_from_slope(m):
    if m > 5.0: return "rising"
    if m < -5.0: return "falling"
    return "level"

def segment_stats(times, f0, onset_times):
    segs = []
    for i in range(len(onset_times)-1):
        t0, t1 = float(onset_times[i]), float(onset_times[i+1])
        mask = (times >= t0) & (times < t1)
        if not np.any(mask):
            segs.append(Segment(t0, t1, t1-t0, 0.0, 0.0, "unknown")); continue
        f = f0[mask]
        if len(f) < 3:
            segs.append(Segment(t0, t1, t1-t0, float(np.mean(f)) if len(f)>0 else 0.0, 0.0, "unknown")); continue
        x = np.arange(len(f), dtype=np.float32)
        m = float(np.polyfit(x, f, 1)[0])
        acc = _accent_from_slope(m)
        segs.append(Segment(t0, t1, t1-t0, float(np.mean(f)), m, acc))
    return segs

def compare_segments_and_score(segs_g, segs_u, strictness=0.5):
    N = min(len(segs_g), len(segs_u))
    msgs = []; pitch_penalty = 0.0; dur_penalty = 0.0; ok = 0
    slope_tol = np.interp(strictness, [0,1], [12.0, 4.0])
    dur_tol   = np.interp(strictness, [0,1], [0.30, 0.10])

    for i in range(N):
        g, u = segs_g[i], segs_u[i]
        if g.accent != u.accent:
            msgs.append(dict(idx=i, time_sec_start=u.t0, time_sec_end=u.t1,
                             message=f"Expected **{g.accent}** accent; heard **{u.accent}**."))
            pitch_penalty += 1.0
        elif abs(g.f0_slope - u.f0_slope) > slope_tol:
            msgs.append(dict(idx=i, time_sec_start=u.t0, time_sec_end=u.t1,
                             message=f"Accent slope off by ~{abs(g.f0_slope - u.f0_slope):.1f}. Aim for a clearer {g.accent} contour."))
            pitch_penalty += 0.5
        else:
            ok += 1

        if g.dur > 0 and u.dur > 0:
            rel = abs(u.dur - g.dur) / g.dur
            if rel > dur_tol:
                hint = "too short" if u.dur < g.dur else "too long"
                msgs.append(dict(idx=i, time_sec_start=u.t0, time_sec_end=u.t1,
                                 message=f"Duration {hint} by ~{rel*100:.0f}%.")) 
                dur_penalty += min(1.0, rel*2)

    score = 100.0 - 10.0 * (pitch_penalty + dur_penalty) / max(N,1)
    score = max(0.0, min(100.0, score))
    msgs = sorted(msgs, key=lambda d: d["time_sec_start"])
    return msgs, dict(score=score, segments_compared=N, ok_segments=ok)

def draw_wave_and_pitch_with_segments(y, sr, times, f0, segs, title="Audio"):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 3))
    t_wave = np.linspace(0, len(y)/sr, len(y))
    plt.plot(t_wave, y, alpha=0.4, label="waveform")
    plt.plot(times, f0, alpha=0.9, label="pitch (Hz)")
    for s in segs: plt.axvline(s.t0, alpha=0.2)
    plt.title(title); plt.xlabel("Time (s)"); plt.ylabel("Amplitude / Hz"); plt.legend(loc="upper right")
    plt.tight_layout(); return fig


# --- Phones (template-based) ---
def parse_phones_from_text(txt: Optional[str]) -> Optional[List[List[str]]]:
    if not txt: return None
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    return [ln.split() for ln in lines] if lines else None

def _segment_audio(y: np.ndarray, sr: int, onsets: np.ndarray, i: int) -> np.ndarray:
    t0, t1 = float(onsets[i]), float(onsets[i+1])
    s0, s1 = max(0, int(t0*sr)), max(0, int(t1*sr))
    seg = y[s0:s1]
    if len(seg) < int(0.03*sr):
        seg = np.pad(seg, (0, int(0.03*sr)-len(seg)), mode='constant')
    return seg

def _mfcc_feat(y: np.ndarray, sr: int) -> np.ndarray:
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return np.mean(m, axis=1, keepdims=True)  # 20x1 centroid

def build_phone_templates_from_guru(y_guru: np.ndarray, sr: int, onsets_g: np.ndarray, phone_labels: List[List[str]]) -> Dict[str, List[np.ndarray]]:
    templates: Dict[str, List[np.ndarray]] = {}
    M = min(len(onsets_g)-1, len(phone_labels))
    for i in range(M):
        phones = phone_labels[i]
        seg = _segment_audio(y_guru, sr, onsets_g, i)
        feat = _mfcc_feat(seg, sr)
        for ph in phones:
            templates.setdefault(ph, []).append(feat.squeeze())
    # collapse to mean vector per phone
    for ph in list(templates.keys()):
        arr = np.stack(templates[ph], axis=0)  # Kx20
        templates[ph] = [arr.mean(axis=0)]
    return templates

def _nearest_phone(feat_vec: np.ndarray, templates: Dict[str, List[np.ndarray]]):
    best_ph, best_d = None, 1e9
    v = feat_vec.squeeze()[None, :]
    for ph, vecs in templates.items():
        for ref in vecs:
            refv = ref[None, :]
            num = float(np.sum(v*refv)); den = float(np.linalg.norm(v)*np.linalg.norm(refv) + 1e-9)
            d = 1 - (num/den)  # cosine distance
            if d < best_d: best_d, best_ph = d, ph
    return best_ph or "?", float(best_d)

def _phone_hint(exp: str, heard: str) -> str:
    pairs = {
        ("ś","s"): "Keep the tongue blade closer to the palate for **ś**.",
        ("ṣ","s"): "Retroflex **ṣ** needs the tongue curled back; avoid dental **s**.",
        ("ṭ","t"): "Retroflex **ṭ**: curl the tongue back slightly; contact is further back.",
        ("ḍ","d"): "Retroflex **ḍ**: pull the tongue back; release is less dental.",
        ("ṇ","n"): "Retroflex **ṇ**: tongue curls back; steady nasal resonance.",
        ("kh","k"): "Aspirated **kh**: add a clear puff of breath after **k**.",
        ("gh","g"): "Aspirated **gh**: audible breath after the stop.",
        ("ch","c"): "Aspirated **ch**: ensure a breathy release; avoid plain **c**.",
        ("jh","j"): "Aspirated **jh**: add breath after **j**.",
        ("ph","p"): "Aspirated **ph**: stronger breath after **p**.",
        ("th","t"): "Aspirated **th** (dental): puff of breath after **t**.",
        ("dh","d"): "Aspirated **dh**: breath after **d**.",
        ("bh","b"): "Aspirated **bh**: breath after **b**.",
        ("ś","ṣ"): "Palatal **ś** vs retroflex **ṣ**: raise tongue body forward for **ś**."
    }
    for (a,b), hint in pairs.items():
        if exp==a and heard==b: return hint
    return "Articulate slowly once, exaggerate the place of articulation, then speed up while keeping clarity."

def classify_learner_phones(y_user: np.ndarray, sr: int, onsets_u: np.ndarray, phone_labels: List[List[str]], templates: Dict[str, List[np.ndarray]]):
    out = []
    M = min(len(onsets_u)-1, len(phone_labels))
    for i in range(M):
        exp = phone_labels[i][0]  # primary label for segment
        seg = _segment_audio(y_user, sr, onsets_u, i)
        feat = _mfcc_feat(seg, sr)
        ph, dist = _nearest_phone(feat, templates)
        mismatch = (ph != exp) or (dist > 0.25)
        if mismatch:
            tip = _phone_hint(exp, ph)
            out.append(dict(
                idx=i, expected=exp, heard=ph, distance=dist,
                message=f"Pronunciation off for **{exp}**; sounded like **{ph}**. {tip}",
                time_sec_start=float(onsets_u[i]), time_sec_end=float(onsets_u[i+1])
            ))
    return out


# --- Natural language summary ---
def generate_natural_language_feedback(accent_msgs: List[Dict[str,Any]], phone_msgs: List[Dict[str,Any]]) -> str:
    parts = []
    if not accent_msgs and not phone_msgs:
        return "Excellent! Your accents, timing, and pronunciation closely match the benchmark. Keep practicing to stabilize this quality."
    if accent_msgs:
        parts.append(f"**Accent & Timing:** {len(accent_msgs)} segment(s) need attention. Shape the rise/fall clearly and hold long vowels fully.")
    else:
        parts.append("**Accent & Timing:** Contours and durations are generally solid.")
    if phone_msgs:
        exp_counts = {}
        for m in phone_msgs: exp_counts[m['expected']] = exp_counts.get(m['expected'], 0) + 1
        common = ", ".join([f"{k}×{v}" for k,v in sorted(exp_counts.items(), key=lambda x:-x[1])[:4]])
        parts.append(f"**Pronunciation:** Watch {common}. Try slow contrast drills (e.g., ṭ–t, ś–ṣ–s).")
        tips = [m.get("message","") for m in phone_msgs[:2]]
        if tips: parts.append("Tips:\n- " + "\n- ".join(tips))
    else:
        parts.append("**Pronunciation:** Consonants and vowels are clearly articulated. Nice work on the tricky contrasts.")
    parts.append("Replay the flagged segments, imitate the exact rise/fall and length, then record again. Consistent slow practice will lock it in.")
    return "\n\n".join(parts)
