# 🎬 AI Screenplay Doctor & Script Analyzer

> An AI-powered screenplay analysis platform that acts as your personal studio script editor — analyzing movie scripts and subtitle files to provide emotional arc mapping, character analysis, conflict detection, and cinematic critique.

---

## 🚀 What It Does

Upload any `.srt` subtitle file or `.txt` script and get:

| Feature | Description |
|---|---|
| 📊 **Emotional Arc Chart** | Sentiment score mapped across every scene — visualize your film's pacing |
| 👥 **Character Tracker** | Extracts all characters, their dialogue, and generates arc summaries |
| 🔥 **Conflict Heatmap** | Flags scenes with no tension or missing conflict |
| 🗺️ **Hero's Journey Map** | Maps your script to standard screenwriting frameworks |
| 📝 **Screenplay Critique** | AI critique report — dialogue issues, plot holes, pacing problems |
| ✍️ **Style Transfer** | Rewrites any scene in the style of famous directors (Anurag Kashyap, Mani Ratnam, Rajamouli, Tarantino, Nolan) |

---

## 🏗️ Architecture — 7 Microservices

```
.srt / .txt file
      ↓
[MS-1] Scene Extractor        → Parses file into scene chunks
      ↓
[MS-2] Character Extractor    → Identifies speakers & tracks arcs
[MS-3] Sentiment Analyzer     → Scores emotion per scene chunk      (parallel)
      ↓
[MS-4] Pacing Detector        → Flags flat zones, maps Hero's Journey
      ↓
[MS-5] Critique Engine        → LLM generates full screenplay critique
      ↓
[MS-6] Dashboard              → Streamlit visual interface
[MS-7] Style Transfer         → Director-style scene rewriter        (on demand)
```

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone <repo-url>
cd Hackathon_iREL
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your API key
```bash
cp .env.example .env
# Add your Gemini API key to .env
```

`.env` format:
```
GEMINI_API_KEY=your_key_here
```

---

## ▶️ Running the App

### Phase 1 — Terminal pipeline
```bash
python main.py --file "your_script.srt" --genre action
```

### Phase 2 — Streamlit UI
```bash
streamlit run app/main_app.py
```

### Phase 4 — FastAPI backend
```bash
uvicorn api.server:app --reload
```

---

## 📁 Project Structure

```
Hackathon_iREL/
├── services/
│   ├── scene_extractor.py       # MS-1: SRT/TXT parser
│   ├── character_extractor.py   # MS-2: Speaker identification
│   ├── sentiment_analyzer.py    # MS-3: Emotional arc scoring
│   ├── pacing_detector.py       # MS-4: Conflict & pacing flags
│   ├── critique_engine.py       # MS-5: LLM screenplay critique
│   └── style_transfer.py        # MS-7: Director style rewriter
├── app/
│   └── main_app.py              # MS-6: Streamlit UI
├── api/
│   └── server.py                # FastAPI endpoints (Phase 4)
├── prompts/
│   └── director_profiles.py     # Director style prompt configs
├── tests/
│   └── test_*.py                # Unit tests
├── srt_parser.py                # Core SRT parsing module
├── main.py                      # Terminal entry point
├── requirements.txt
└── README.md
```

---

## 🎬 Demo — RRR (2022)

The system was tested on the RRR subtitle file:
- **1,403** subtitle blocks parsed
- **1,205** spoken dialogue lines extracted  
- **198** song lyric lines filtered out
- **36** scene chunks (5-min windows) analyzed

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| LLM | Google Gemini API (`gemini-1.5-flash`) |
| Frontend | Streamlit |
| Backend API | FastAPI |
| Visualization | Plotly |
| Storage | SQLite |
| Deploy | Streamlit Cloud + Render.com |

---

## 👥 Team

Built for the **GenAI NLP System Building Challenge Hackathon** — 3 days.

---

## 📄 License

MIT
