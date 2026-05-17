import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sys
import tempfile
import time

# Add root to sys.path to allow importing backend services
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import run_pipeline

# ── App Config & Styling ──────────────────────────────────────────────────────
st.set_page_config(page_title="AI Screenplay Doctor", page_icon="🎬", layout="wide")

st.markdown("""
<style>
    /* Dark Cinematic Theme */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .metric-card {
        background: #1E2329;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
        border-top: 3px solid #E50914; /* Cinematic red */
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FFFFFF;
    }
    .metric-label {
        font-size: 1rem;
        color: #B3B3B3;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .critique-box {
        background: #1E2329;
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid #E50914;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ── State Management ──────────────────────────────────────────────────────────
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "sentiment_data" not in st.session_state:
    st.session_state.sentiment_data = None
if "pacing_data" not in st.session_state:
    st.session_state.pacing_data = None
if "critique_data" not in st.session_state:
    st.session_state.critique_data = None


# ── Data Loading Helpers ──────────────────────────────────────────────────────
def load_json(filename):
    path = os.path.join(ROOT, filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

def refresh_data():
    st.session_state.sentiment_data = load_json("sentiment_output.json")
    st.session_state.pacing_data = load_json("pacing_output.json")
    st.session_state.critique_data = load_json("critique_output.json")
    if st.session_state.sentiment_data and st.session_state.pacing_data and st.session_state.critique_data:
        st.session_state.analysis_done = True

# Initialize state from disk on first load if available
if not st.session_state.analysis_done:
    refresh_data()


# ── Sidebar & Pipeline Runner ─────────────────────────────────────────────────
with st.sidebar:
    st.title("🎬 Script Analyzer")
    st.write("Upload a screenplay (.srt) to analyze its emotional arc, pacing, and dialogue.")
    
    uploaded_file = st.file_uploader("Upload Subtitles (.srt)", type=["srt"])
    
    st.subheader("⚙️ Settings")
    max_chunks = st.slider("Max Chunks to Analyze", min_value=1, max_value=36, value=18, 
                           help="1 chunk = 5 minutes. 18 chunks = 90 mins.")
    fast_mode = st.checkbox("Fast Mode (No LLM API)", value=False, 
                            help="Use keyword fallback. Extremely fast but inaccurate.")
    skip_micro = st.checkbox("Skip Micro Analysis", value=False, 
                             help="Skip line-by-line LLM checks. Saves time and API calls.")
    
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        if uploaded_file is not None:
            # Save uploaded file to temp path
            temp_path = os.path.join(ROOT, "app_temp.srt")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            # Temporarily patch the pipeline SRT_FILE
            run_pipeline.SRT_FILE = temp_path
            
            with st.spinner("Analyzing Script... (this may take a few minutes if LLM is enabled)"):
                t0 = time.time()
                run_pipeline.run(
                    max_chunks=max_chunks,
                    use_llm=not fast_mode,
                    run_micro=not skip_micro,
                    user_climax_chunk=None,
                    run_critique=True
                )
                elapsed = time.time() - t0
                
            refresh_data()
            st.success(f"Analysis Complete! ({elapsed:.1f}s)")
            os.remove(temp_path)
        else:
            st.warning("Please upload a .srt file first.")

# ── Main Dashboard UI ─────────────────────────────────────────────────────────
st.title("AI Screenplay Doctor Dashboard")

if not st.session_state.analysis_done:
    st.info("👈 Upload a script and run the analysis to view the dashboard.")
    st.stop()

# Data shortcuts
sent_data = st.session_state.sentiment_data
pace_data = st.session_state.pacing_data
crit_data = st.session_state.critique_data

# ── TOP METRICS ──
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Screenplay Score</div>
            <div class="metric-value">{crit_data.get('score', 0):.1f}<span style='font-size:1.5rem;color:#777;'>/10</span></div>
        </div>
    """, unsafe_allow_html=True)
with m2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Overall Sentiment</div>
            <div class="metric-value">{sent_data.get('overall_score', 0):+.2f}</div>
        </div>
    """, unsafe_allow_html=True)
with m3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Pacing Score</div>
            <div class="metric-value">{pace_data.get('pacing_score', 0):.1f}<span style='font-size:1.5rem;color:#777;'>/10</span></div>
        </div>
    """, unsafe_allow_html=True)
with m4:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Peak Tension Chunk</div>
            <div class="metric-value">#{sent_data.get('peak_tension_chunk', '?')}</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── VISUALIZATIONS ──
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("📈 Emotional Arc (Macro)")
    
    # Line Chart Prep
    chunk_list = sent_data.get('chunks', [])
    chunk_dict = {int(ch['chunk_id']): ch for ch in chunk_list}
    chunks = sorted(chunk_dict.keys())
    scores = [chunk_dict[c].get('macro_score', 0) for c in chunks]
    
    df_arc = pd.DataFrame({"Chunk": chunks, "Sentiment Score": scores})
    
    fig_arc = px.line(df_arc, x="Chunk", y="Sentiment Score", markers=True, 
                      template="plotly_dark", color_discrete_sequence=["#E50914"])
    
    # Add horizontal zero line
    fig_arc.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutral", opacity=0.5)
    
    # Add Act boundaries if they exist
    acts = pace_data.get('act_boundaries', [])
    if acts:
        for act_chunk in acts:
            fig_arc.add_vline(x=act_chunk, line_dash="dot", line_color="#00FFAA", 
                              annotation_text="Act Change", annotation_position="top right", opacity=0.7)
            
    fig_arc.update_layout(yaxis_range=[-1.1, 1.1], height=400, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_arc, use_container_width=True)

with col_right:
    st.subheader("🔥 Emotion Heatmap")
    
    # Heatmap Prep
    emotions = ["joy", "trust", "anticipation", "surprise", "fear", "sadness", "disgust", "anger"]
    heat_data = []
    
    for c in chunks:
        ch_dict = chunk_dict.get(c, {})
        em_dict = ch_dict.get('emotions', {e:0 for e in emotions})
        heat_data.append([em_dict.get(e, 0) for e in emotions])
        
    df_heat = pd.DataFrame(heat_data, index=chunks, columns=emotions)
    df_heat = df_heat.T # Transpose so chunks are X, emotions are Y
    
    fig_heat = px.imshow(df_heat, text_auto=".1f", aspect="auto", 
                         color_continuous_scale="RdBu_r", origin='lower',
                         template="plotly_dark")
    fig_heat.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_heat, use_container_width=True)


st.markdown("<hr>", unsafe_allow_html=True)

# ── CRITIQUE & HERO'S JOURNEY ──
col_crit, col_hero = st.columns([2, 1])

with col_crit:
    st.subheader("📝 Critique Engine Verdict")
    st.markdown(f"""
        <div class="critique-box">
            <h4>🤖 Overall Verdict</h4>
            <p>{crit_data.get('verdict', 'No verdict generated.')}</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.subheader("🔧 Top 3 Fixes")
    for fix in crit_data.get('top_fixes', []):
        st.info(fix)
        
    plot_issues = crit_data.get('plot_issues', [])
    if plot_issues:
        st.subheader("🔍 Plot Issues Detected")
        for issue in plot_issues:
            icon = "🔴" if issue.get('severity') == "critical" else "⚠️"
            chunks_str = ", ".join(map(str, issue.get('affected_chunks', [])))
            st.warning(f"{icon} **Chunk(s) {chunks_str}**: {issue['description']}")

with col_hero:
    st.subheader("🗺️ Hero's Journey Map")
    hero_map = pace_data.get('hero_journey', {})
    
    if not hero_map:
        st.info("Hero's Journey analysis was skipped (likely due to short chunk count).")
    else:
        # Sort by chunk number
        mapped_items = []
        for stage, chunk_idx in hero_map.items():
            if chunk_idx is not None:
                mapped_items.append((chunk_idx, stage))
        mapped_items.sort()
        
        for chunk, stage in mapped_items:
            st.markdown(f"**Chunk {chunk}:** {stage}")

# ── MS-7 STYLE TRANSFER PANEL (Bonus) ──
st.markdown("<hr>", unsafe_allow_html=True)
with st.expander("✍️ Style Transfer Panel (Rewrite a Scene)"):
    st.write("Select a scene to rewrite in the style of a famous director.")
    
    col_s1, col_s2, col_s3 = st.columns([1, 1, 2])
    with col_s1:
        chunk_sel = st.selectbox("Select Chunk", options=chunks)
    with col_s2:
        director = st.selectbox("Select Director", 
                                ["Anurag Kashyap", "Imtiaz Ali", "Mani Ratnam"])
    with col_s3:
        st.write("") # spacing
        st.write("")
        st.write("")
        if st.button("Rewrite Scene (Mock)"):
            import style_transfer_prompts as stp
            st.toast("Style Transfer triggered!")
            prompt = getattr(stp, f"PROMPT_{list(['Anurag Kashyap', 'Imtiaz Ali', 'Mani Ratnam']).index(director)+1}_{director.replace(' ', '_').upper()}")
            st.code(prompt, language="text")
            st.info("Backend integration for style transfer would run here using the selected prompt and chunk text.")
