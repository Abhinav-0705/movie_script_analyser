# 🎬 Movie Script Analyser — Project Status Analysis

---

## Part 1: What Your Teammate (Abhinav) Has Done

### Overall Summary

Abhinav has built **the entire backend analysis pipeline** — from raw SRT file ingestion through sentiment analysis to pacing detection. This covers **Phase 0 (Research) and Phase 1 (Terminal-Based Modular System)** of the hackathon almost entirely, plus significant documentation. Here's the breakdown:

---

### ✅ MS-1: SRT Parser — `srt_parser.py` (192 lines)

**Fully implemented and tested on RRR (2022).**

| Feature | Detail |
|---|---|
| `Subtitle` dataclass | Stores index, timestamps (raw + seconds), cleaned text, song detection flag, scene chunk ID |
| `parse_srt()` | Reads `.srt` files, handles `\r\n` / `\r\r\n` / `\n`, strips HTML (`<i>`, `<b>`, `<font>`), detects songs via `<i>` or ♪ |
| `assign_scene_chunks()` | Groups subtitles into configurable N-minute windows (default 5 min) using `int(start_sec / chunk_sec) + 1` |
| `get_dialogue_only()` | Filters out song lyrics (critical — songs would skew sentiment scores) |
| `get_scene_chunk_texts()` | Returns `{chunk_id: "combined dialogue text"}` dict — direct input for MS-3 |
| `extract_character_lines()` | Naive regex-based `CHARACTER: dialogue` detector (fallback for MS-2) |
| CLI entry point | `python3 srt_parser.py "filename.srt"` — prints stats and previews |

**Validated output on RRR:** 1,403 blocks → 1,205 dialogue lines + 198 songs → 36 scene chunks.

---

### ✅ MS-2: Character Extractor — `services/character_extractor.py` (318 lines)

**Fully implemented using Gemini (`gemini-1.5-flash`).**

| Feature | Detail |
|---|---|
| `AttributedLine` dataclass | Links subtitle index, chunk, timestamp, speaker name, and dialogue |
| `CharacterProfile` dataclass | Aggregates: name, all lines, chunks seen, line count, first/last chunk, arc summary placeholder |
| `_build_attribution_prompt()` | Sends dialogue batches (25 lines) to Gemini with film context + growing known-characters list |
| `_call_gemini()` | Handles JSON parsing, markdown stripping, retry logic (3 attempts) |
| `extract_characters()` | Main entry: iterates batches, feeds known-chars into subsequent prompts for consistency |
| `get_character_summary()` | Returns clean dict excluding NARRATOR/UNKNOWN — ready for Streamlit |

**Key design decision:** Known characters from earlier batches are fed into later batch prompts → improves attribution consistency across the film.

> [!NOTE]
> MS-2 uses **Gemini API** (not Groq like the other services). Requires `GEMINI_API_KEY` in `.env`.

---

### ✅ MS-3: Two-Layer Sentiment Analyzer — `services/sentiment_analyzer.py` (556 lines)

**The core NLP engine. Fully implemented with two independent Groq API keys.**

#### Layer 1 — Macro (Chunk-level)
- Model: `llama-3.3-70b-versatile` via `GROQ_API_KEY_1`
- Input: Full 5-min chunk text (up to 2000 chars)
- Output: valence score (-1.0 → +1.0), emotional label, dominant Plutchik emotion, all 8 emotion scores, reasoning

#### Layer 2 — Micro (Line-level)
- Model: `llama-3.1-8b-instant` via `GROQ_API_KEY_2`
- Input: Individual dialogue lines (batched in groups of 20)
- Output: per-line emotion, intensity (0.0-1.0), valence
- Lines with `intensity > 0.7` are **flagged** as dramatically important

#### Trajectory Analysis
- Splits each chunk's micro-level lines into **thirds** (start / mid / end)
- Computes average valence per third
- Classifies arc shape: `Rising | Falling | V-shape | Inverted-V | Flat | Mixed`

#### Additional Features
| Feature | Detail |
|---|---|
| Keyword fallback (`_fast_score_chunk`) | No-API mode using positive/negative word lists — works with `--fast` flag |
| Flat zone detection | 3+ consecutive chunks with `|score| < 0.2` |
| Act boundary detection | Chunks where sentiment shift exceeds 0.4 |
| `timeline_to_dict()` | Serializes everything to JSON for downstream services |
| Full data models | `LineSentiment`, `TrajectoryArc`, `ChunkSentiment`, `SentimentTimeline` |

---

### ✅ MS-4: Pacing Detector — `services/pacing_detector.py` (632 lines)

**Fully implemented. The most sophisticated module — contains 3 original analytical innovations.**

#### Innovation 1: Genre Ideal Curve Comparison
- `IDEAL_CURVES` dict with 7 genre templates (action, epic, romance, thriller, drama, comedy, default)
- `_interpolate_ideal()` — linear interpolation to compute ideal sentiment at any film position
- Computes per-chunk `{actual, ideal, delta}` and overall RMSE deviation
- Flags chunks where `|actual - ideal| > 0.5`

#### Innovation 2: Tension Debt Accumulator
- Novel concept: sustained positivity builds "debt" the film must repay with darkness
- Calm scenes → debt += 0.15; dark scenes → debt -= 0.40
- Flags emotional fatigue risk when debt > 1.2

#### Innovation 3: Narrative Momentum
- Derivative of sentiment arc — rate of emotional change between chunks
- Detects **spike types** with flatness context:
  - Rising spike breaking flatness → "Victory/Relief" (good storytelling)
  - Falling spike breaking flatness → "Betrayal/Shock" (maximum dramatic impact)
  - Spikes mid-chaos → "Plateau" or "Deepening" (diminishing returns)
- `avg_momentum` measures overall film volatility

#### Standard Features
| Feature | Detail |
|---|---|
| Mode auto-detection | ≥8 chunks = full film (all features); <8 = scene mode (climax/hero's journey disabled) |
| Flat zone detection | 3+ consecutive chunks below emotional threshold |
| Climax validation | **Opt-in only** (`--climax N`) — checks if climax is at 60-88% of film |
| Hero's Journey mapping | LLM-powered — maps 10 stages to chunk IDs using Groq |
| Pacing score formula | `10.0 - (RMSE×3) - (flat_chunks×0.2) - (debt_peak×0.5) + (dramatic_spike_bonus×0.3)` |
| `report_to_dict()` | Full JSON serialization |

---

### ✅ Pipeline Runner — `run_pipeline.py` (149 lines)

**Chains MS-1 → MS-3 → MS-4 into a single CLI command.**

```bash
python3 run_pipeline.py                    # first 90 min (chunks 1-18)
python3 run_pipeline.py --chunks 36        # full film
python3 run_pipeline.py --chunks 10 --fast # keyword mode (no API)
python3 run_pipeline.py --no-micro         # skip Layer 2
python3 run_pipeline.py --climax 28        # validate climax position
```

- Saves `sentiment_output.json` and `pacing_output.json`
- Prints comprehensive terminal summary with ASCII bar charts, flags, hero's journey map, and timing

---

### ✅ Style Transfer Prompts — `style_transfer_prompts.py` (230 lines)

**Three production-ready rewrite prompts using actual RRR scenes:**

| Prompt | Director | Scene |
|---|---|---|
| Prompt 1 | Anurag Kashyap (Gangs of Wasseypur) | The Shepherd Speech (Chunk 4) |
| Prompt 2 | Imtiaz Ali (Tamasha, Rockstar) | Ram-Jenny Meet-Cute (Chunk 10) |
| Prompt 3 | Mani Ratnam (Roja, Dil Se) | Undercover Mission (Chunk 6) |

Each prompt includes 8-9 detailed style rules + the full original scene text + specific rewriting instructions. These are **copy-paste ready** for any LLM but not yet integrated into a service.

---

### ✅ Sample Outputs (Pre-generated)

| File | Content |
|---|---|
| `sentiment_output.json` (379 lines) | 18 chunks analyzed — macro scores, labels, emotions, no micro/trajectory data (run was LLM macro-only) |
| `pacing_output.json` (354 lines) | Full pacing analysis — 7.81/10 score, hero's journey map, momentum timeline, 15 flags, tension debt curve |

---

### ✅ Documentation

| File | Size | Content |
|---|---|---|
| `README.md` | 397 lines | Full project overview, architecture diagram, setup guide, tech stack, demo data, two-layer architecture deep dive |
| `PIPELINE_WALKTHROUGH.md` | 632 lines | Detailed technical walkthrough of every function in MS-1, MS-3, MS-4 — design decisions, modifiable parameters, output formats |

---

### ✅ Infrastructure

| File | Detail |
|---|---|
| `requirements.txt` | 10 dependencies: groq is **missing** (uses `google-generativeai`, `streamlit`, `fastapi`, `uvicorn`, `plotly`, `pandas`, `transformers`, `torch`, `python-dotenv`, `pydantic`) |
| `.env.example` | Only has `GEMINI_API_KEY` — **missing** `GROQ_API_KEY_1` and `GROQ_API_KEY_2` |
| `.gitignore` | Covers Python, venvs, .env, databases, IDEs, media files |
| `RRR 2022 JPN UHD en full.srt` | 98KB test dataset — English subtitles for the full film |

---

## Part 2: What's Left To Do

### 🔴 Priority Legend
- 🔴 **Critical** — Must have for submission
- 🟡 **Important** — Expected by hackathon rubric
- 🟢 **Bonus** — Would impress judges

---

### 🔴 1. Fix `requirements.txt` and `.env.example` (10 min)

**The repo is currently broken for anyone who clones it.**

- `requirements.txt` is **missing `groq`** — the main LLM library used by MS-3 and MS-4
- `.env.example` is missing `GROQ_API_KEY_1` and `GROQ_API_KEY_2`
- `torch` + `transformers` are listed but **never used** anywhere — remove to speed up install

```diff
# requirements.txt fixes needed:
+ groq>=0.9.0
- transformers>=4.41.0
- torch>=2.3.0

# .env.example fixes needed:
+ GROQ_API_KEY_1=your_groq_key_1_here
+ GROQ_API_KEY_2=your_groq_key_2_here
```

---

### 🔴 2. MS-5: Critique Engine — `services/critique_engine.py` (Not started)

**What it should do** (per README):
- Takes `sentiment_output.json` + `pacing_output.json` + character data
- Generates **character arc summaries** (who each character is at start vs end)
- Flags **clunky dialogue** (using flagged lines from MS-3 Layer 2)
- Identifies **plot holes** (character disappears, unresolved conflict)
- Maps screenplay to **Hero's Journey framework** (already done in MS-4 — integrate)
- Produces an **overall screenplay score (1–10)** with "Top 3 fixes"
- Output: `critique_output.json`

**Estimated effort:** 3-4 hours (heavy LLM prompt engineering + aggregation logic)

---

### 🔴 3. MS-6: Streamlit Dashboard — `app/main_app.py` (Not started, `app/` directory doesn't exist)

**This is Phase 2 of the hackathon — the minimal frontend. Critical for the demo.**

Required components (per README):
| Component | Priority | Detail |
|---|---|---|
| 📂 File upload widget | 🔴 | Accept `.srt` / `.txt` files |
| 📊 Emotional Arc Line Chart | 🔴 | Plotly line chart: x=chunk, y=sentiment score, with Act overlays |
| 🔥 Emotion Heatmap | 🔴 | Plutchik 8 emotions × N chunks (Plotly heatmap) |
| 👥 Character Roster Table | 🟡 | Line count, first/last scene, arc summary |
| 🗺️ Hero's Journey Stage Map | 🟡 | Visual timeline of mapped stages |
| 📝 Critique Report Cards | 🟡 | Critical 🔴 / Warning ⚠️ / Good ✅ styling |
| ✍️ Style Transfer Panel | 🟢 | Select scene + director → get rewrite |

**Estimated effort:** 4-6 hours (Plotly charts + Streamlit layout + connecting all services)

---

### 🔴 4. MS-7: Style Transfer Service — `services/style_transfer.py` (Not started)

The prompts exist in `style_transfer_prompts.py` but there's **no actual service** that:
- Takes a scene chunk + director name as input
- Calls an LLM with the appropriate style prompt
- Returns the rewritten scene

**Estimated effort:** 1-2 hours (mostly prompt templating + LLM call wrapper)

---

### 🔴 5. Unit Tests (Not started)

**The hackathon rubric explicitly requires "Unit Tests and Testing Methodology".**

Tests needed:
| Test | What to test |
|---|---|
| `test_srt_parser.py` | Parsing, song detection, chunk assignment, edge cases (empty blocks, malformed timestamps) |
| `test_sentiment_analyzer.py` | Keyword fallback scoring, trajectory classification, flat zone detection |
| `test_pacing_detector.py` | Ideal curve interpolation, tension debt logic, momentum calculation, flat zone detection, mode detection |
| `test_character_extractor.py` | Profile aggregation, known-characters accumulation |
| `test_pipeline_integration.py` | End-to-end: SRT → sentiment JSON → pacing JSON |

**Estimated effort:** 2-3 hours

---

### 🟡 6. FastAPI Backend — `api/server.py` (Not started, `api/` directory doesn't exist)

**Phase 4 of the hackathon — wrapping services as REST endpoints.**

Required endpoints (per README):
```
POST /api/extract-scenes       → MS-1
POST /api/extract-characters   → MS-2
POST /api/analyze-sentiment    → MS-3
POST /api/detect-pacing        → MS-4
POST /api/critique             → MS-5
POST /api/style-transfer       → MS-7
```

**Estimated effort:** 2-3 hours

---

### 🟡 7. Data Persistence Layer (Phase 3) (Not started)

**Phase 3 — "Add Data Layer (Optional)" but would significantly improve the demo.**

Options:
- SQLite (simplest — already in requirements concept)
- JSON file storage (current outputs already are JSON — just need a session/history layer)

What to persist:
- Upload history (user inputs)
- Analysis results (generated content)
- Session logs/summaries

**Estimated effort:** 1-2 hours (SQLite with 3-4 tables)

---

### 🟡 8. Cloud Deployment (Phase 4)

**Options from the rubric:**

| What | Where | How |
|---|---|---|
| Streamlit Dashboard | Streamlit Cloud | `streamlit deploy` (free, easiest) |
| FastAPI Backend | Render.com | Connect GitHub repo, set env vars |

**Pre-requisite:** Streamlit app (MS-6) and FastAPI (api/server.py) must be built first.

**Estimated effort:** 1-2 hours (mostly config + env vars + testing)

---

### 🟡 9. Requirement Specification Document (Not started)

**Explicitly required in the submission checklist: "Requirement Specification Document"**

Should cover:
- Problem statement
- Functional requirements (each microservice)
- Non-functional requirements (performance, scalability, API rate limits)
- User stories / use cases
- Input/output specifications

**Estimated effort:** 1 hour (much content can be extracted from existing README)

---

### 🟡 10. System Design Diagram (Not started)

**Required: "Any and all system design diagram (draw.io, Excalidraw, or hand-drawn OK)"**

The README has an ASCII architecture diagram, but you need a **proper visual diagram** showing:
- Microservice boundaries
- Data flow between services
- API endpoints
- Frontend ↔ backend communication
- Database connections

**Estimated effort:** 30-60 min (Excalidraw or draw.io)

---

### 🟡 11. 5-Minute Walkthrough Notes

**Required: "5-min Walkthrough or Notes on Architecture, GenAI usage, What worked and what didn't"**

Should cover:
1. Architecture overview (microservice breakdown)
2. GenAI usage reflection (as required by the guidelines):
   - What worked well
   - Where it fell short
   - How it impacted development
3. Demo of the working system
4. Key design decisions (two-layer sentiment, tension debt, etc.)

**Estimated effort:** 1 hour

---

### 🟢 12. Integrate `run_pipeline.py` with MS-2 (Character Extractor)

Currently `run_pipeline.py` chains MS-1 → MS-3 → MS-4 but **skips MS-2 entirely**. Character extraction is standalone — its output isn't consumed by any downstream service.

**What's needed:**
- Add MS-2 call in `run_pipeline.py` after MS-1
- Save `character_output.json`
- Feed character data into MS-5 (critique engine)

**Estimated effort:** 30 min

---

### 🟢 13. Bonus: Docker, GitHub Actions, LangChain

The rubric mentions these as bonus items:
- **Dockerfile** — containerize the app
- **GitHub Actions** — CI/CD pipeline running tests on push
- **LangChain** — could replace raw Groq/Gemini calls with chains

**Estimated effort:** 2-4 hours total (low priority)

---

## Summary: Priority Task List

| # | Task | Priority | Effort | Phase |
|---|---|---|---|---|
| 1 | Fix `requirements.txt` + `.env.example` | 🔴 | 10 min | Infra |
| 2 | MS-6: Streamlit Dashboard | 🔴 | 4-6 hrs | Phase 2 |
| 3 | MS-5: Critique Engine | 🔴 | 3-4 hrs | Phase 1 |
| 4 | Unit Tests | 🔴 | 2-3 hrs | Testing |
| 5 | MS-7: Style Transfer Service | 🔴 | 1-2 hrs | Phase 1 |
| 6 | FastAPI Backend (`api/server.py`) | 🟡 | 2-3 hrs | Phase 4 |
| 7 | Requirement Specification Document | 🟡 | 1 hr | Docs |
| 8 | System Design Diagram | 🟡 | 30-60 min | Docs |
| 9 | 5-Minute Walkthrough Notes | 🟡 | 1 hr | Docs |
| 10 | Data Persistence (SQLite) | 🟡 | 1-2 hrs | Phase 3 |
| 11 | Cloud Deployment | 🟡 | 1-2 hrs | Phase 4 |
| 12 | Integrate MS-2 into pipeline | 🟢 | 30 min | Phase 1 |
| 13 | Bonus: Docker / GitHub Actions | 🟢 | 2-4 hrs | Bonus |

**Total estimated remaining work: ~20-30 hours**

