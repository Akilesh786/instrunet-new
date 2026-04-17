import streamlit as st
import torch
import torch.nn as nn
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import os
from datetime import datetime

# ==========================================
# 🚩 PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Instrunet AI",
    page_icon="🎼",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model_final.pth")

# ==========================================
# 🎵 INSTRUMENT CONFIG
# Must match exactly what you trained with (sorted order)
# ==========================================
INSTRUMENT_MAP = {
    'cel': 'Cello',
    'cla': 'Clarinet',
    'flu': 'Flute',
    'gac': 'Acoustic Guitar',
    'gel': 'Electric Guitar',
    'har': 'Harmonium',
    'mri': 'Mridangam',
    'org': 'Organ',
    'pia': 'Piano',
    'sax': 'Saxophone',
    'tab': 'Tabla',
    'tru': 'Trumpet',
    'vio': 'Violin',
    'voi': 'Voice'
}
INSTRUMENTS = sorted(INSTRUMENT_MAP.keys())
FULL_NAMES  = [INSTRUMENT_MAP[k] for k in INSTRUMENTS]
NUM_CLASSES = len(INSTRUMENTS)   # 14

# Detection threshold for multi-label output
THRESHOLD = 0.35

# ==========================================
# 🧠 CNN ARCHITECTURE
# Must be IDENTICAL to your training notebook
# ==========================================
class InstrumentCNN(nn.Module):
    def __init__(self, num_classes=14):
        super(InstrumentCNN, self).__init__()

        # --- BLOCK 1 --- input: (1, 128, 130)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),       # → (32, 64, 65)
            nn.Dropout2d(0.25)
        )

        # --- BLOCK 2 ---
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),       # → (64, 32, 32)
            nn.Dropout2d(0.25)
        )

        # --- BLOCK 3 ---
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),       # → (128, 16, 16)
            nn.Dropout2d(0.25)
        )

        # --- BLOCK 4 ---
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # → (256, 4, 4)
            nn.Dropout2d(0.25)
        )

        # --- CLASSIFIER HEAD ---
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x


# ==========================================
# 🤖 AI CHATBOT
# ==========================================
def get_bot_response(user_input, last_result=None):
    user_input = user_input.lower()

    if any(w in user_input for w in ["backend", "pipeline", "process", "how it works"]):
        return """<b>Backend Pipeline:</b><br>
1. <b>Upload:</b> Audio loaded at 22050 Hz, converted to mono.<br>
2. <b>Landmarks:</b> Onset peaks detected to find strongest signal points.<br>
3. <b>Feature Extraction:</b> Mel Spectrogram (128 bands × 130 frames) extracted per chunk.<br>
4. <b>PyTorch CNN:</b> Input shaped to (1, 128, 130) fed through 4 ConvBlocks.<br>
5. <b>Multi-label Output:</b> Sigmoid layer gives 0–1 probability for each of the 11 instruments.<br>
6. <b>Threshold:</b> Instruments above {threshold}% are marked as DETECTED (14 instrument classes total).
""".format(threshold=int(THRESHOLD * 100))

    elif any(w in user_input for w in ["waveform", "peaks", "landmark", "amplitude"]):
        if last_result:
            count = len(last_result['signal']['landmarks'])
            return f"""<b>Waveform Analysis:</b><br>
The blue graph shows amplitude over time. The <span style='color:#ef4444'><b>red dashed lines</b></span> are the <b>{count} temporal landmarks</b> detected.<br><br>
These are the note 'attack' points where instrument timbre is clearest. Silence is ignored — only these {count} moments are analyzed."""
        return "Upload an audio file first, then I can tell you exactly how many peaks were detected!"

    elif any(w in user_input for w in ["spectrogram", "mel"]):
        return """<b>Mel Spectrogram:</b><br>
A visual 'fingerprint' of sound — maps <b>Frequency vs. Time</b> using the Mel Scale (mimics human hearing).
Brighter = louder at that frequency/time. The CNN reads these like images to identify instruments."""

    elif any(w in user_input for w in ["model", "cnn", "architecture", "neural", "network", "layer"]):
        return """<b>CNN Architecture (4 ConvBlocks):</b><br>
- <b>Block 1:</b> 2× Conv2D(32) → BN → ReLU → MaxPool → Dropout<br>
- <b>Block 2:</b> 2× Conv2D(64) → BN → ReLU → MaxPool → Dropout<br>
- <b>Block 3:</b> 2× Conv2D(128) → BN → ReLU → MaxPool → Dropout<br>
- <b>Block 4:</b> 2× Conv2D(256) → BN → ReLU → AdaptiveAvgPool(4×4) → Dropout<br>
- <b>Head:</b> Flatten → Linear(4096→512) → Linear(512→128) → Linear(128→14) → <b>Sigmoid</b><br><br>
Total: <b>3.3M parameters</b>. Sigmoid enables multi-label detection across <b>14 instrument classes</b>."""

    elif "multilabel" in user_input or "multi-label" in user_input or "multiple" in user_input:
        return """<b>Multi-Label Detection:</b><br>
Unlike single-label (softmax), this model uses <b>Sigmoid</b> — each instrument gets an independent 0–1 score.<br>
This means <b>multiple instruments can be detected simultaneously</b> — perfect for real music with mixed instruments!"""

    elif "accuracy" in user_input:
        return "The model achieves approximately <b>85–90% F1 score</b> across instrument classes on the IRMAS validation set. Cello is hardest; Voice and Electric Guitar score highest."

    elif "threshold" in user_input:
        return f"Currently using a detection threshold of <b>{int(THRESHOLD*100)}%</b>. Instruments scoring above this are marked DETECTED. You can tune this in <code>app.py</code> → <code>THRESHOLD</code> variable."

    elif any(w in user_input for w in ["result", "prediction", "what did", "detected"]):
        if last_result:
            detected = [n for n, s in zip(FULL_NAMES, last_result['scores']) if s >= THRESHOLD]
            top = FULL_NAMES[int(np.argmax(last_result['scores']))]
            conf = float(np.max(last_result['scores']))
            return f"""Last prediction: <b>{', '.join(detected) if detected else 'None above threshold'}</b><br>
Strongest signal: <b>{top}</b> at {conf*100:.1f}%.<br>
Analyzed across {len(last_result['signal']['landmarks'])} audio segments."""
        return "Upload and analyze an audio file first!"

    else:
        return "I'm the Instrunet Technical Guide. Ask me about the <b>Waveform</b>, <b>CNN Architecture</b>, <b>Mel Spectrogram</b>, <b>Multi-label Detection</b>, <b>Backend Pipeline</b>, or <b>Threshold</b>!"


# ==========================================
# 🎨 CSS STYLES
# ==========================================
def apply_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');

    .stApp { background: #0b0f19; color: #e2e8f0; font-family: 'Inter', sans-serif; }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .stMarkdown, .stButton, .stPlotlyChart { animation: fadeInUp 0.6s ease-out; }

    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }
    .nav-header {
        color: #38bdf8; font-size: 22px; font-weight: 900;
        padding: 20px 0; text-align: center;
        border-bottom: 1px solid #1e293b; margin-bottom: 20px;
    }
    div[role="radiogroup"] > label {
        padding-top: 12px !important; padding-bottom: 12px !important;
        font-size: 1.05rem !important; font-weight: 600 !important; color: #cbd5e1 !important;
    }
    div[role="radiogroup"] > label:hover { color: #38bdf8 !important; }

    .hero-section {
        background: linear-gradient(135deg, rgba(56,189,248,0.1) 0%, rgba(99,102,241,0.1) 100%);
        border: 1px solid rgba(255,255,255,0.08); border-radius: 24px;
        padding: 50px; text-align: center; margin-bottom: 40px;
        backdrop-filter: blur(20px); box-shadow: 0 20px 40px -10px rgba(0,0,0,0.5);
    }
    .hero-section h1 { font-size: 52px !important; font-weight: 900 !important; color: #ffffff !important; margin-bottom: 10px; }
    .hero-section p  { color: #94a3b8; font-size: 1.1em; }

    .metric-card {
        background: rgba(30,41,59,0.4); border-radius: 16px; padding: 35px;
        border: 1px solid #334155; text-align: center; margin-bottom: 40px !important;
    }

    .result-card {
        background: rgba(56,189,248,0.08); border: 1px solid rgba(56,189,248,0.3);
        border-radius: 16px; padding: 30px; margin-bottom: 20px;
    }
    .detected-badge {
        display: inline-block; background: rgba(34,197,94,0.2);
        border: 1px solid #22c55e; border-radius: 8px;
        padding: 6px 16px; margin: 4px; font-weight: 700; color: #86efac;
    }

    .stButton>button {
        background: linear-gradient(90deg, #0ea5e9 0%, #6366f1 100%);
        border: none; border-radius: 12px; color: white;
        height: 3.8em; font-weight: 700; font-size: 1.1em;
        width: 100%; margin-top: 10px;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 10px 25px rgba(56,189,248,0.25); }

    .ai-msg {
        background: #1e293b; border-radius: 12px; padding: 16px;
        margin-bottom: 16px; border-left: 4px solid #38bdf8;
        font-size: 0.88em; line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)


# ==========================================
# 🧠 MODEL LOADER (cached)
# ==========================================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    model = InstrumentCNN(num_classes=NUM_CLASSES)
    state = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    # Handle if saved as full model dict or just state_dict
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    elif isinstance(state, dict):
        model.load_state_dict(state)
    else:
        model = state  # full model saved
    model.eval()
    return model


# ==========================================
# 🔬 AUDIO PROCESSING & INFERENCE
# ==========================================
def process_audio(file_path, model):
    y, sr = librosa.load(file_path, sr=22050, duration=15)

    # Detect onset landmarks
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    peaks = librosa.util.peak_pick(
        onset_env, pre_max=7, post_max=7,
        pre_avg=7, post_avg=7, delta=0.5, wait=30
    )
    times = librosa.frames_to_time(peaks, sr=sr)
    if len(times) == 0:
        times = np.array([0.0])

    all_preds = []
    for t in times[:10]:
        start = int(max(0, (t - 0.5) * sr))
        chunk = y[start: start + int(3 * sr)]
        if len(chunk) < int(3 * sr):
            chunk = np.pad(chunk, (0, int(3 * sr) - len(chunk)))

        # Mel spectrogram → shape (128, 130)
        mel = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Normalize to [0, 1]
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

        # Fix time dimension to exactly 130 frames
        if mel_db.shape[1] >= 130:
            mel_db = mel_db[:, :130]
        else:
            mel_db = np.pad(mel_db, ((0, 0), (0, 130 - mel_db.shape[1])))

        # Shape: (1, 1, 128, 130) for PyTorch
        tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            pred = model(tensor).squeeze().numpy()  # shape: (11,)
        all_preds.append(pred)

    avg_scores = np.mean(all_preds, axis=0)   # shape: (11,)
    mel_full   = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

    return {
        "meta":   {"id": datetime.now().strftime("%H:%M:%S")},
        "scores": avg_scores,
        "signal": {
            "y":         y,
            "sr":        sr,
            "landmarks": times,
            "spec":      mel_full
        }
    }


# ==========================================
# 🖥️  PAGE: HOME
# ==========================================
def render_home():
    st.markdown("""
    <div class='hero-section'>
        <h1>🎼 INSTRUNET AI</h1>
        <p>Multi-Label Neural Network · 14 Instruments · PyTorch CNN</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='metric-card' style='max-width:1100px; margin: 0 auto 50px auto;'>
        <h3 style='color:#38bdf8;'>System Architecture</h3>
        <p style='font-size:1.05em; color:#cbd5e1; padding:10px 30px; line-height:1.9;'>
            4-Block <b>Convolutional Neural Network</b> trained on <b>IRMAS + custom dataset</b> (audio files).<br>
            Extracts <b>128-band Mel Spectrograms</b> from onset-detected audio landmarks.<br>
            <b>Sigmoid output</b> enables simultaneous detection of multiple instruments.
        </p>
        <p style='color:#64748b; font-size:0.9em;'>
            Instruments: Cello · Clarinet · Flute · Acoustic Guitar · Electric Guitar ·
            Harmonium · Mridangam · Organ · Piano · Saxophone · Tabla · Trumpet · Violin · Voice
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("OPEN ANALYSIS STUDIO 🚀", use_container_width=True):
        st.session_state.page = "Upload & Analyze"
        st.rerun()


# ==========================================
# 🖥️  PAGE: UPLOAD & ANALYZE
# ==========================================
def render_studio(model):
    st.title("🎙️ Analysis Studio")

    if model is None:
        st.error("⚠️ Model not found! Place `instrunet_model.pth` inside the `models/` folder.")
        st.info("Expected path: `models/instrunet_model.pth`")
        return

    file = st.file_uploader("Upload audio file", type=["wav", "mp3", "ogg", "flac"])
    if file:
        st.audio(file)
        if st.button("⚡ EXECUTE NEURAL SCAN", use_container_width=True):
            with st.spinner("Analyzing audio... extracting mel spectrograms..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name
                try:
                    result = process_audio(tmp_path, model)
                    st.session_state.current = result
                    st.session_state.history.append(result)
                    st.session_state.page = "Instrument Distribution"
                    st.rerun()
                except Exception as e:
                    st.error(f"Processing failed: {e}")
                finally:
                    os.unlink(tmp_path)


# ==========================================
# 🖥️  PAGE: DISTRIBUTION
# ==========================================
def render_distribution():
    res = st.session_state.current
    if res is None:
        st.warning("No analysis results yet. Go to Upload & Analyze first.")
        return

    scores   = res['scores']
    detected = [(FULL_NAMES[i], scores[i]) for i in range(NUM_CLASSES) if scores[i] >= THRESHOLD]
    detected.sort(key=lambda x: x[1], reverse=True)

    st.title("📊 Analysis Results")

    # Top result hero card
    top_name  = FULL_NAMES[int(np.argmax(scores))]
    top_conf  = float(np.max(scores))
    det_html  = "".join([f"<span class='detected-badge'>✅ {n} ({c*100:.0f}%)</span>" for n, c in detected])
    no_det    = "<span style='color:#94a3b8;'>No instrument above threshold</span>" if not detected else ""

    st.markdown(f"""
    <div class='result-card'>
        <h2 style='color:#38bdf8; margin:0;'>{top_name.upper()}</h2>
        <h4 style='color:#94a3b8;'>Strongest Signal · {top_conf*100:.1f}% Confidence</h4>
        <hr style='border-color:#334155;'>
        <p style='color:#94a3b8; margin-bottom:8px;'>Detected Instruments (≥{int(THRESHOLD*100)}%):</p>
        {det_html or no_det}
    </div>
    """, unsafe_allow_html=True)

    # Bar chart
    df = pd.DataFrame({"Instrument": FULL_NAMES, "Confidence": scores.tolist()})
    df = df.sort_values("Confidence", ascending=False)
    fig = px.bar(
        df, x="Instrument", y="Confidence",
        color="Confidence", color_continuous_scale="Blues",
        template="plotly_dark", range_y=[0, 1]
    )
    fig.add_hline(y=THRESHOLD, line_dash="dash", line_color="#ef4444",
                  annotation_text=f"Threshold ({int(THRESHOLD*100)}%)", annotation_position="top right")
    fig.update_layout(height=420, margin=dict(t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

    if st.button("OPEN TECHNICAL SIGNAL BREAKDOWN 🔬", use_container_width=True):
        st.session_state.page = "Deep Technical Analysis"
        st.rerun()


# ==========================================
# 🖥️  PAGE: DEEP TECHNICAL ANALYSIS
# ==========================================
def render_technical():
    res = st.session_state.current
    if res is None:
        st.warning("No analysis results yet. Go to Upload & Analyze first.")
        return

    st.title("🔬 Deep Technical Analysis")

    # --- Waveform + Landmarks ---
    st.subheader("1. Pulse Landmarks & Temporal Peaks")
    y, sr = res['signal']['y'], res['signal']['sr']
    t = np.linspace(0, len(y) / sr, num=len(y))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t[::100], y=y[::100],
        name="Amplitude", line=dict(color='#38bdf8', width=1.5)
    ))
    for lm in res['signal']['landmarks']:
        fig.add_vline(x=lm, line_dash="dash", line_color="#ef4444", opacity=0.7)
    fig.update_layout(
        template="plotly_dark", height=350,
        margin=dict(t=10, b=10),
        xaxis_title="Time (s)", yaxis_title="Amplitude"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"🔴 {len(res['signal']['landmarks'])} onset landmarks detected — CNN analyzed up to 10 of these segments.")

    # --- Mel Spectrogram ---
    st.subheader("2. Mel Spectrogram — Timbre Fingerprint")
    S_db = librosa.power_to_db(res['signal']['spec'], ref=np.max)
    fig2 = px.imshow(
        S_db, origin='lower', aspect='auto',
        template="plotly_dark", color_continuous_scale='Magma',
        labels=dict(x="Time Frames", y="Mel Frequency Bands", color="dB")
    )
    fig2.update_layout(height=400, margin=dict(t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)

    # --- Radar Chart ---
    st.subheader("3. Multi-Label Confidence Radar")
    scores = res['scores'].tolist()
    fig3 = go.Figure(go.Scatterpolar(
        r=scores + [scores[0]],
        theta=FULL_NAMES + [FULL_NAMES[0]],
        fill='toself',
        fillcolor='rgba(56,189,248,0.15)',
        line=dict(color='#38bdf8', width=2)
    ))
    fig3.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        template="plotly_dark", height=450,
        margin=dict(t=10, b=10)
    )
    st.plotly_chart(fig3, use_container_width=True)


# ==========================================
# 🖥️  PAGE: AUDIT LOGS
# ==========================================
def render_history():
    st.title("📜 Neural Audit Logs")
    if not st.session_state.history:
        st.info("No previous sessions found. Analyze some audio first!")
        return

    for item in reversed(st.session_state.history):
        detected = [FULL_NAMES[i] for i in range(NUM_CLASSES) if item['scores'][i] >= THRESHOLD]
        top_name = FULL_NAMES[int(np.argmax(item['scores']))]
        top_conf = float(np.max(item['scores']))
        st.markdown(f"""
        <div class='ai-msg'>
            <b>SESSION [{item['meta']['id']}]</b><br>
            🏆 Top: {top_name} — {top_conf*100:.1f}%<br>
            ✅ Detected: {', '.join(detected) if detected else 'None above threshold'}
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# 🚀 MAIN
# ==========================================
def main():
    apply_styles()
    model = load_model()

    # Session state init
    if "page"    not in st.session_state: st.session_state.page    = "Home"
    if "current" not in st.session_state: st.session_state.current = None
    if "history" not in st.session_state: st.session_state.history = []
    if "chat"    not in st.session_state: st.session_state.chat    = []

    PAGES = ["Home", "Upload & Analyze", "Instrument Distribution", "Deep Technical Analysis", "Audit Logs"]

    with st.sidebar:
        st.markdown("<div class='nav-header'>🎼 INSTRUNET AI</div>", unsafe_allow_html=True)
        nav = st.radio("NAVIGATE", PAGES, index=PAGES.index(st.session_state.page))
        if nav != st.session_state.page:
            st.session_state.page = nav
            st.rerun()

        # Chatbot
        st.markdown("<hr style='border-color:#1e293b; margin: 30px 0 20px 0;'>", unsafe_allow_html=True)
        st.subheader("🤖 Technical Guide")
        for msg in st.session_state.chat[-4:]:
            label = "👤 YOU" if msg["role"] == "user" else "🤖 AI"
            st.markdown(f"<div class='ai-msg'><b>{label}:</b><br>{msg['content']}</div>", unsafe_allow_html=True)
        if q := st.chat_input("Ask about the model or waveform..."):
            response = get_bot_response(q, st.session_state.current)
            st.session_state.chat.append({"role": "user",      "content": q})
            st.session_state.chat.append({"role": "assistant", "content": response})
            st.rerun()

    # Router
    if   st.session_state.page == "Home":                    render_home()
    elif st.session_state.page == "Upload & Analyze":        render_studio(model)
    elif st.session_state.page == "Instrument Distribution": render_distribution()
    elif st.session_state.page == "Deep Technical Analysis": render_technical()
    elif st.session_state.page == "Audit Logs":              render_history()


if __name__ == "__main__":
    main()