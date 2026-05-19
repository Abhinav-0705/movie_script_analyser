# 🎬 AI Screenplay Doctor & Movie Script Analyzer

> An AI-powered screenplay analysis platform that acts as your personal studio script editor — analyzing movie scripts and subtitle files like a professional film critic, NLP researcher, and screenwriting coach combined.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Groq](https://img.shields.io/badge/LLM-Groq%20%7C%20Llama3-orange)](https://groq.com)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green)](https://fastapi.tiangolo.com)

---

## 🎯 Problem Statement

Writers and filmmakers often struggle with:
- **Weak, flat dialogue** that fails to engage
- **Inconsistent emotional pacing** — no tension in Act 2
- **Flat character arcs** — characters who don't grow or change
- **Missing conflict** — scenes that go nowhere dramatically
- **Expensive manual script coverage** — studio readers charge thousands

> This project automates screenplay understanding using **NLP + LLMs + microservices** to provide instant cinematic analysis that would otherwise require a professional script editor.

---

## 🏗️ System Architecture — 7 Microservices

```
User uploads .srt / .txt file
          │
          ▼
╔══════════════════════════════╗
║  MS-1: Scene Extractor       ║  ✅ IMPLEMENTED
║  srt_parser.py               ║
╚══════════╤═══════════════════╝
           │
     ┌─────┴──────┐
     ▼            ▼
╔══════════════╗  ╔══════════════════════════════════╗
║  MS-2:       ║  ║  MS-3: Sentiment & Emotion       ║  ✅ IMPLEMENTED
║  Character   ║  ║  Analyzer (Two-Layer)            ║
║  Extractor   ║  ║  services/sentiment_analyzer.py  ║
╚══════╤═══════╝  ╚══════════════╤═══════════════════╝
       │                         │
       └────────────┬────────────┘
                    ▼
╔══════════════════════════════════╗
║  MS-4: Conflict & Pacing         ║  ✅ IMPLEMENTED
║  Detector                        ║
║  services/pacing_detector.py     ║
╚══════════════════╤═══════════════╝
                   │
                   ▼
╔══════════════════════════════════╗
║  MS-5: Screenplay Critique       ║  ✅ IMPLEMENTED
║  Engine (LLM)                    ║
║  services/critique_engine.py     ║
╚══════════════════╤═══════════════╝
                   │
      ┌────────────┴────────────┐
      ▼                         ▼
╔══════════════════╗  ╔═════════════════════════════╗
║  MS-6: Visual    ║  ║  MS-7: Style Transfer        ║  ✅ IMPLEMENTED
║  Dashboard       ║  ║  Engine                      ║
║  app/main_app.py ║  ║  services/style_transfer.py  ║
╚══════════════════╝  ╚═════════════════════════════╝
          │
          ▼
╔══════════════════════════════════╗
║  Phase 4: FastAPI REST API       ║  ✅ IMPLEMENTED
║  api/server.py                   ║
╚══════════════════════════════════╝
```

---

## ✅ What's Implemented

### MS-1: File Ingestion & Scene Extractor
**File:** `srt_parser.py`

Parses `.srt` subtitle files with full robustness:
- Handles Windows `\r\n` and `\r\r\n` line endings
- Strips HTML tags (`<i>`, `<b>`, `<font>`)
- **Separates song lyrics from spoken dialogue** (italic = song → excluded)
- Groups dialogue into configurable N-minute scene chunks
- Converts timestamps to seconds for timeline plotting

```bash
python3 srt_parser.py "RRR 2022 JPN UHD en full.srt"
```

**Output on RRR (2022):**
```
✅ Total subtitle blocks : 1403
🎤 Spoken dialogue lines : 1205
🎵 Song lyric lines      : 198
🎬 Scene chunks (5-min)  : 36
```

---

### MS-2: Character Extractor & Tracker
**File:** `services/character_extractor.py`

Uses Groq LLM (llama-3.3-70b) to attribute speaker identity to each dialogue line — critical because SRT files contain no speaker labels.

- Processes dialogue in batches of 25 lines
- Maintains a growing known-characters list for cross-batch consistency
- Builds `CharacterProfile` objects: line count, first/last scene, sample lines
- Falls back gracefully to `UNKNOWN` if API fails

**Key design:** Known character list is fed into each subsequent batch prompt → later batches are more accurate as context accumulates.

---

### MS-3: Sentiment & Emotion Analyzer ⭐ (Two-Layer Architecture)
**File:** `services/sentiment_analyzer.py`

The core NLP engine. Uses **two separate Groq API keys** for two independent analysis layers:

#### Layer 1 — Macro Analysis (Key 1: `llama-3.3-70b-versatile`)
- Analyzes the **full 5-minute chunk** as a single unit
- Returns: valence score (-1.0 → +1.0), emotion label, **Plutchik's 8-emotion breakdown**
- Drives: **Emotional Arc Line Chart** (the main visualization)

#### Layer 2 — Micro Analysis (Key 2: `llama-3.1-8b-instant`)
- Analyzes **each individual dialogue line** in the chunk
- Returns: per-line emotion, intensity (0.0–1.0), valence
- Lines with `intensity > 0.7` are **flagged** as dramatically important
- Drives: **Critique Engine** — tells MS-5 exactly which lines to target

#### Trajectory Analysis
- Splits each chunk's lines into **thirds** (start / mid / end)
- Classifies the emotional arc shape: `Rising | Falling | V-shape | Inverted-V | Flat | Mixed`
- Drives: **Pacing Detector** — a "Flat" arc for 3+ chunks = conflict missing

**Emotion Model:** [Plutchik's Wheel of Emotions](https://en.wikipedia.org/wiki/Robert_Plutchik) (8 primary)
```
joy, trust, anticipation, surprise, fear, sadness, disgust, anger
```

```bash
python3 services/sentiment_analyzer.py
```

**Output on RRR — First 4 chunks:**
```
Chunk 1: -0.80 [Sorrowful] dom=sadness   | Trajectory: Falling (+0.43→+0.26→-0.60)
Chunk 2: -0.80 [Wrathful]  dom=anger     | Trajectory: Flat
Chunk 3: -0.50 [Tense]     dom=fear      | Trajectory: Flat
Chunk 4: -0.50 [Tense]     dom=fear      | Trajectory: Falling (+0.33→-0.63→-0.14)

High-intensity lines flagged (26 total):
  "They've bought your daughter."
  "Please give me back my child."
  "he will break its teeth, pry its jaws open"
  "It seems that the shepherd has come to Delhi to begin his hunt."
```

---

### MS-7: Style Transfer (Prompts Ready)
**File:** `style_transfer_prompts.py`

Director-style rewrite prompts crafted for Indian cinema:

| Director | Style Profile Summary |
|---|---|
| **Anurag Kashyap** | Raw, gritty, dark humour buried in violence, understated threat |
| **Mani Ratnam** | Poetic, restrained, visual metaphors, political subtext |
| **Imtiaz Ali** | Improvised dialogue, identity-searching characters, unresolved scenes |
| **SS Rajamouli** | Mythic, operatic, escalating stakes, brotherhood as war |

Three full prompts using real RRR scenes are ready to paste into any LLM.

---

## ✅ Implemented — All Microservices Complete

### MS-4: Conflict & Pacing Detector
**File:** `services/pacing_detector.py`

Consumes `SentimentTimeline` from MS-3 and applies rule-based analysis:
- **Flat Zone Detection** — 3+ consecutive chunks with `|score| < 0.2` → "Scene lacks tension"
- **Hero's Journey Mapping** — LLM maps chunks to 10-stage framework
- **Act Break Detection** — sharp sentiment shifts mark likely Act 1→2→3 transitions
- **Climax Validation** — checks if peak tension falls in the final third (as expected)
- **Pacing Score** — variance of sentiment scores (low variance = monotonous film)
- **3 Innovative Features:** Ideal Pacing Curve comparison, Tension Debt Accumulator, Narrative Momentum

### MS-5: Screenplay Critique Engine
**File:** `services/critique_engine.py`

The main LLM critique — fed the outputs of MS-2, MS-3, MS-4:
- Generates **character arc summaries** (who each character is at start vs end)
- Flags **clunky dialogue** (using flagged lines from MS-3 Layer 2)
- Identifies **plot holes** (character disappears, unresolved conflict)
- Maps screenplay to **Hero's Journey framework**
- Produces an **overall screenplay score (1–10)** with "Top 3 fixes"

### MS-6: Visual Dashboard
**File:** `app/main_app.py`

Streamlit web interface:
- 📂 File upload (`.srt`, `.txt`)
- 📊 **Emotional Arc Line Chart** (Plotly) — x=chunk, y=sentiment score, Act overlays
- 👥 **Character Roster Table** — line count, first/last scene, arc summary
- 🔥 **Emotion Heatmap** — Plutchik 8 emotions across all 36 chunks
- 🗺️ **Hero's Journey Stage Map**
- 📝 **Critique Report Cards** — Critical 🔴 / Warning ⚠️ / Good ✅
- ✍️ **Style Transfer Panel** — select scene + director → get rewrite

### Phase 4: FastAPI REST API
**File:** `api/server.py`

All services wrapped as REST endpoints with auto-generated Swagger docs:
```
GET  /health                  → Health check
POST /api/extract-scenes      → MS-1 Scene Extractor
POST /api/extract-characters  → MS-2 Character Extractor
POST /api/analyze-sentiment   → MS-3 Sentiment Analyzer
POST /api/detect-pacing       → MS-4 Pacing Detector
POST /api/critique            → MS-5 Critique Engine
POST /api/style-transfer      → MS-7 Style Transfer
POST /api/run-pipeline        → Full pipeline (MS-1→5)
```

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/Abhinav-0705/movie_script_analyser.git
cd movie_script_analyser
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API keys
```bash
cp .env.example .env
```

Edit `.env`:
```env
GROQ_API_KEY_1=your_groq_key_1_here    # Layer 1 (macro chunk analysis)
GROQ_API_KEY_2=your_groq_key_2_here    # Layer 2 (micro line analysis)
GEMINI_API_KEY=your_gemini_key_here    # optional fallback
```

> Get free Groq API keys at: **console.groq.com** (no credit card needed)

---

## ▶️ Running

### Phase 1 — Terminal Pipeline (MS-1 + MS-3)
```bash
# Parse SRT and run sentiment analysis
python3 services/sentiment_analyzer.py

# Parse SRT only
python3 srt_parser.py "your_movie.srt"

# Character extraction
python3 services/character_extractor.py
```

### Phase 2 — Streamlit UI
```bash
streamlit run app/main_app.py
```

### Phase 4 — FastAPI REST API
```bash
uvicorn api.server:app --reload
# Swagger docs available at: http://127.0.0.1:8000/docs
```

---

## 📁 Project Structure

```
movie_script_analyser/
│
├── srt_parser.py                    # MS-1: SRT parser ✅
│
├── services/
│   ├── character_extractor.py       # MS-2: Speaker attribution ✅
│   ├── sentiment_analyzer.py        # MS-3: Two-layer emotion analysis ✅
│   ├── pacing_detector.py           # MS-4: Conflict & pacing ✅
│   ├── critique_engine.py           # MS-5: LLM critique ✅
│   └── style_transfer.py            # MS-7: Director rewriter ✅
│
├── app/
│   └── main_app.py                  # MS-6: Streamlit dashboard ✅
│
├── api/
│   ├── __init__.py
│   └── server.py                    # FastAPI REST API (Phase 4) ✅
│
├── style_transfer_prompts.py        # Director style prompt library ✅
├── run_pipeline.py                  # Pipeline orchestrator ✅
├── sentiment_output.json            # Sample output from MS-3
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 🎬 Demo Dataset

The system was validated on **RRR (2022)** English subtitles:

| Metric | Value |
|---|---|
| Total subtitle blocks | 1,403 |
| Spoken dialogue lines | 1,205 |
| Song lyric lines (filtered) | 198 |
| Scene chunks (5-min windows) | 36 |
| High-intensity lines flagged (4 chunks) | 26 |

**Sentiment results validated against actual film:**
- Chunk 1 → `-0.80 Sorrowful` ✅ (Malli being taken from her mother)
- Chunk 4 → `-0.50 Tense` ✅ (The Shepherd speech — Bheem coming to Delhi)

---

## 🛠️ Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| **LLM (Layer 1)** | Groq `llama-3.3-70b-versatile` | Macro chunk sentiment |
| **LLM (Layer 2)** | Groq `llama-3.1-8b-instant` | Micro line-level emotion |
| **Frontend** | Streamlit | Interactive dashboard |
| **Backend API** | FastAPI | Microservice REST endpoints |
| **Visualisation** | Plotly | Charts and heatmaps |
| **Storage** | SQLite | Analysis history (Phase 3) |
| **Deploy** | Streamlit Cloud + Render.com | Cloud-native hosting |

---

## 📊 Two-Layer Sentiment Architecture (MS-3 Deep Dive)

```
Scene Chunk Text (5-min dialogue)
          │
    ┌─────┴──────┐
    ▼            ▼
LAYER 1         LAYER 2
(Key 1 / 70b)  (Key 2 / 8b)
    │            │
Chunk-level    Line-level
valence score  emotion per line
+ 8 Plutchik   + intensity score
  emotions     + flagging (>0.7)
    │            │
    └─────┬──────┘
          │
     TRAJECTORY
  (split into thirds)
  start → mid → end
  classify arc shape:
  Rising | Falling |
  V-shape | Flat | etc.
          │
          ▼
   ChunkSentiment object
   → feeds MS-4, MS-5, MS-6
```

---

## 👥 Team

Built for the **GenAI NLP System Building Challenge Hackathon** — 3 days, 2 people.

---

## 📄 License

MIT

---

## 📖 Developer Docs

For a full technical walkthrough of every function, design decision, and modification guide see:

**[PIPELINE_WALKTHROUGH.md](./PIPELINE_WALKTHROUGH.md)**

Covers MS-1 (SRT Parser), MS-3 (Sentiment Analyzer), MS-4 (Pacing Detector) in depth — intended for teammates who want to understand or modify the pipeline.

