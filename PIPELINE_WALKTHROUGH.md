# 🎬 Pipeline Walkthrough — AI Screenplay Analyzer
> **For teammates:** This document explains every module, every function, what it does, why we built it that way, what the output looks like, and how to modify it.

---

## Overall Pipeline Flow

```
User uploads .srt file
        ↓
┌─────────────────────────────────┐
│  MS-1: srt_parser.py            │  Parses raw SRT → clean Subtitle objects
└────────────────┬────────────────┘
                 ↓
┌─────────────────────────────────┐
│  MS-3: sentiment_analyzer.py    │  Two-layer emotion scoring per chunk
│  Layer 1 (Groq Key 1 / 70b)    │  → Chunk-level valence + 8 Plutchik emotions
│  Layer 2 (Groq Key 2 / 8b)     │  → Line-level emotion + intensity flags
└────────────────┬────────────────┘
                 ↓
         sentiment_output.json
                 ↓
┌─────────────────────────────────┐
│  MS-4: pacing_detector.py       │  3 innovations + Hero's Journey mapping
└────────────────┬────────────────┘
                 ↓
          pacing_output.json
```

**All outputs are JSON** — every downstream service reads the previous service's JSON file. This makes each module independently testable.

---

## MS-1: `srt_parser.py`

### Purpose
Converts a raw `.srt` subtitle file into clean, structured Python objects that every downstream module can use. Handles all the messy real-world SRT format issues.

### Data Model: `Subtitle`
```python
@dataclass
class Subtitle:
    index:       int      # subtitle number from SRT
    start_sec:   float    # start time in seconds
    end_sec:     float    # end time in seconds
    text:        str      # cleaned dialogue text
    is_song:     bool     # True if this is a song lyric (italic in SRT)
    scene_chunk: int      # which 5-min window this belongs to (set later)
```

---

### Function: `parse_srt(filepath)`

**What it does:**  
Reads the `.srt` file and converts each subtitle block into a `Subtitle` object.

**Step by step:**
1. Reads file with `latin-1` encoding (handles special characters)
2. Normalises line endings — SRT files from different sources use `\r\n`, `\r\r\n`, or `\n`. We convert all to `\n\n` so splitting on double-newline works reliably
3. Splits on blank lines — each SRT block is `index / timestamp / text`
4. Parses timestamp with regex: `00:01:23,456 --> 00:01:25,789`
5. Converts to seconds: `hours*3600 + minutes*60 + seconds + milliseconds/1000`
6. Detects songs: if the text block contains `<i>` HTML tags → `is_song = True`
7. Strips all HTML tags from text (`<i>`, `<b>`, `<font color=...>`, etc.)

**Why we separate songs:**  
Song lyrics are not spoken dialogue. If you include "♪ Naatu Naatu ♪" in the emotion analysis, it will falsely inflate joy scores during musical numbers. Songs are kept in the objects but filtered out by `get_dialogue_only()` before any NLP.

**Output:** `list[Subtitle]` — all subtitle blocks, songs included, `scene_chunk` not yet set.

**Modifiable:**
- Change `latin-1` to `utf-8` if your SRT files use UTF-8
- Add more HTML tag patterns if your SRTs have unusual formatting
- Change song detection logic if your SRTs don't use italics for songs

---

### Function: `assign_scene_chunks(subtitles, chunk_minutes=5)`

**What it does:**  
Assigns a `scene_chunk` integer to every subtitle based on its timestamp.

**Logic:**
```python
scene_chunk = int(start_sec / (chunk_minutes * 60)) + 1
# A subtitle at 7:30 → int(450 / 300) + 1 = 2 → Chunk 2
# A subtitle at 14:59 → int(899 / 300) + 1 = 3 → Chunk 3
```

**Why 5-minute chunks:**  
5 minutes is the standard unit in screenwriting — one "page" of a screenplay = ~1 minute of screen time, so 5 minutes ≈ 5 pages ≈ one beat. Long enough to have a coherent emotional tone; short enough to track pacing changes. For a 3-hour film like RRR this gives 36 chunks — a manageable dataset for LLM analysis.

**Modifiable:** Change `chunk_minutes` parameter. Use 10 for broader analysis, 3 for finer detail.

---

### Function: `get_dialogue_only(subtitles)`
Returns only subtitles where `is_song == False`. Simple filter, but critical — called before every downstream service.

### Function: `get_scene_chunk_texts(subtitles)`
**Output:** `{chunk_id: "all dialogue text concatenated"}` — a dict used by MS-3.

---

### MS-1 Output on RRR (2022):
```
Total subtitle blocks : 1403
Spoken dialogue lines : 1205  (filtered to these)
Song lyric lines      : 198   (excluded from NLP)
Scene chunks (5-min)  : 36
```

---

## MS-3: `services/sentiment_analyzer.py`

### Purpose
The core NLP engine. Assigns emotion scores to every scene chunk using two independent LLM calls via two separate Groq API keys.

### Why Two API Keys?
- **Key 1** → Layer 1 (chunk-level macro analysis) — uses the heavier `llama-3.3-70b-versatile` model
- **Key 2** → Layer 2 (line-level micro analysis) — uses the faster `llama-3.1-8b-instant` model

This means both layers can run without either key hitting rate limits. The two analyses are independent of each other — Layer 1 doesn't wait for Layer 2.

### Emotion Model: Plutchik's Wheel (8 primary emotions)

We use Plutchik instead of basic positive/negative because Indian cinema requires more nuance:

| Emotion | Why it matters for Indian cinema |
|---|---|
| `joy` | Celebration, dance, reunion |
| `trust` | Brotherhood, loyalty — the core of RRR |
| `anticipation` | Building suspense, hope |
| `surprise` | Plot twists, revelations |
| `fear` | Threat, chase, imprisonment |
| `sadness` | Loss, grief, sacrifice |
| `disgust` | Colonial oppression, betrayal |
| `anger` | Revenge, confrontation, injustice |

---

### Layer 1: Macro Chunk Analysis

**Function:** `_build_sentiment_prompt(chunk_id, text, film_title)` + `_call_macro(prompt)`

**Input:** Full text of a 5-minute scene chunk (~300 words)  
**LLM:** `llama-3.3-70b-versatile` via Groq Key 1  
**Temperature:** 0.15 (low = consistent, deterministic scoring)

**What Gemini/Groq is asked:**
> "Score the overall emotional tone of this scene. Use Plutchik's 8 emotions. Return JSON with: score (-1.0 to +1.0), label, dominant_emotion, all 8 emotion scores, and one-sentence reasoning."

**Score scale:**
```
-1.0  Extremely dark: brutal violence, despair, devastating loss
-0.5  Tense: threat, confrontation, anxiety
 0.0  Neutral: expository, transitional dialogue  
+0.5  Warm: friendship, romance, hope
+1.0  Triumphant: victory, joyful reunion, celebration
```

**Output per chunk:**
```json
{
  "score": -0.80,
  "label": "Sorrowful",
  "dominant_emotion": "sadness",
  "emotions": {
    "joy": 0.0, "trust": 0.1, "anticipation": 0.0,
    "surprise": 0.2, "fear": 0.3, "sadness": 0.9,
    "disgust": 0.4, "anger": 0.2
  },
  "reasoning": "Scene depicts child abduction by colonial officers"
}
```

**Why not use a sentiment library like VADER or TextBlob?**  
VADER was trained on English tweets and product reviews. It has no concept of cinematic context, Indian cultural references, or the difference between "kill" in a threat vs. "killed it" as slang. The LLM understands film context.

---

### Layer 2: Micro Line Analysis

**Function:** `_micro_prompt(lines, film_title)` + `_call_micro(lines, film_title)`

**Input:** List of individual dialogue lines from a chunk (batched in groups of 20)  
**LLM:** `llama-3.1-8b-instant` via Groq Key 2  
**Temperature:** 0.1 (very deterministic)

**What it asks:**
> "For each dialogue line, identify the dominant Plutchik emotion, intensity (0.0–1.0), and valence (-1.0 to +1.0)."

**Key rule — intensity thresholds:**
- `intensity > 0.7` → line is **flagged** as high-intensity / dramatically important
- `intensity < 0.2` → expository / transitional line, no dramatic weight

**Why this matters:**  
Layer 1 tells you *Chunk 14 is overall warm (+0.45)*. Layer 2 tells you *within Chunk 14, line "I'm not who you think I am" has fear intensity 0.88* — the guilt hidden inside the friendship scene. Without Layer 2, you miss the emotional complexity.

**Output per line:**
```json
{"emotion": "guilt", "intensity": 0.88, "valence": -0.6}
```

---

### Trajectory Analysis

**Function:** `_compute_trajectory(line_sentiments)`

**What it does:** Splits each chunk's lines into thirds and computes average valence for each third.

```
Chunk 14:
  First 8 lines  (start): avg valence = +0.70  (joy — Ram-Bheem banter)
  Middle 8 lines (mid):   avg valence = +0.20  (slight unease)
  Last 7 lines   (end):   avg valence = -0.30  (guilt — hidden identity)
```

**Arc shape classification:**
```python
if end > start + 0.25:           → "Rising"      (builds to positive)
elif end < start - 0.25:         → "Falling"     (deteriorates)
elif mid < min(s,e) - 0.2:       → "V-shape"     (dips then recovers)
elif mid > max(s,e) + 0.2:       → "Inverted-V"  (peaks then falls)
elif abs(end - start) < 0.12:    → "Flat"         (no change)
else:                             → "Mixed"
```

**Why trajectory matters:** A chunk scored `+0.45` tells MS-4 very little. A chunk scored `+0.45` with shape `"Falling"` tells MS-4 *"this chunk is deceptively warm but trending darker — potential dramatic turn incoming."*

---

### Data Models

```python
@dataclass
class LineSentiment:
    line_text:  str
    emotion:    str     # Plutchik emotion
    intensity:  float   # 0.0 → 1.0
    valence:    float   # -1.0 → +1.0
    is_flagged: bool    # True if intensity > 0.7

@dataclass  
class ChunkSentiment:
    chunk_id:          int
    macro_score:       float   # Layer 1 output
    macro_label:       str
    dominant_emotion:  str
    emotions:          dict    # all 8 Plutchik scores
    line_sentiments:   list    # Layer 2 output
    peak_line:         str     # most intense line in this chunk
    flagged_lines:     list    # all intensity > 0.7 lines
    trajectory:        TrajectoryArc

@dataclass
class SentimentTimeline:
    chunks:             list[ChunkSentiment]
    overall_score:      float
    peak_tension_chunk: int     # lowest macro_score chunk
    peak_joy_chunk:     int     # highest macro_score chunk
    flat_zones:         list    # chunks with |score| < 0.2 for 3+ run
    act_boundaries:     list    # chunks where score shifts > 0.4
```

---

### MS-3 Output: `sentiment_output.json`

This is the input to MS-4. Key fields MS-4 uses:
- `chunks[i].macro_score` — for ideal curve comparison and momentum
- `chunks[i].macro_label` — for Hero's Journey prompt
- `act_boundaries` — passed through to pacing report

### Modifiable in MS-3:
| What | Where | Effect |
|---|---|---|
| Chunk size | `chunk_minutes` in `srt_parser.py` | Finer/coarser analysis |
| LLM model for Layer 1 | `MODEL_MACRO` | Quality vs. speed |
| LLM model for Layer 2 | `MODEL_MICRO` | Speed vs. accuracy |
| Micro batch size | `MICRO_BATCH = 20` | Smaller = more API calls |
| Flag threshold | `intensity > 0.7` | Raise for fewer flags |
| Trajectory thresholds | `_compute_trajectory()` | Sensitivity of arc detection |

---

## MS-4: `services/pacing_detector.py`

### Purpose
Takes the `sentiment_output.json` from MS-3 and runs structural analysis — no new LLM calls except for Hero's Journey mapping.

### Mode Auto-Detection
```python
FULL_FILM_THRESHOLD = 8  # chunks

if n >= 8:  mode = "full_film"   # all features enabled
else:        mode = "scene"       # climax/Hero's Journey disabled
```

**Why:** Climax position (should be at 60-88% of film) is meaningless for a 3-scene excerpt. Hero's Journey requires a full narrative arc. But tension debt and momentum work at any granularity. With `--force-mode full_film` you can override.

---

### Innovation 1: `compute_ideal_vs_actual()` — Genre Curve Comparison

**The `IDEAL_CURVES` dictionary:**  
A lookup table of `(film_position, ideal_sentiment_score)` tuples for each genre.

```python
"action": [
    (0.0,  +0.1),   # Opens slightly hopeful (we meet the hero)
    (0.1,  -0.3),   # Stakes introduced
    (0.25, -0.5),   # Conflict escalates
    (0.5,  -0.6),   # Mid-film pressure
    (0.75, -0.9),   # "All Is Lost" beat — the darkest moment
    (0.85, -0.4),   # Beginning of resolution
    (1.0,  +0.6),   # Triumph
]
```

**`_interpolate_ideal(position, genre)`:**  
Linear interpolation between two nearest curve points. At position 0.3, if curve has points at 0.25 and 0.4, it calculates the straight line between them and returns the value at 0.3.

**RMSE (Root Mean Square Error):**  
Single number measuring how far the actual arc deviates from the ideal curve.
- `RMSE < 0.3` = follows genre conventions closely
- `RMSE 0.3–0.6` = moderate deviation (intentional or structural issue)
- `RMSE > 0.6` = significant departure from genre expectations

**Individual chunk flags:** Any chunk where `|actual - ideal| > 0.5` gets flagged. That specific scene is emotionally "wrong" for its position in the story.

**Important note on RRR:** The system flagged chunks 9-11 as "should feel darker" — but that's the Ram-Bheem friendship peak, which is RRR intentionally subverting the genre formula. High RMSE ≠ bad film. It means the film breaks conventions — which could be artistic genius or a pacing problem. Context is needed.

**Modifiable:**
- `IDEAL_CURVES["epic"]` — tune for Indian/Tollywood cinema specifically
- Flag threshold `0.5` → raise to `0.7` to only flag severe deviations
- Replace linear interpolation with spline for smoother curve

---

### Innovation 2: `compute_tension_debt()` — Audience Fatigue Tracker

**The concept:** If a film stays too comfortable for too long, audiences build up an unconscious expectation of darkness. The film "owes" them tension.

**Algorithm:**
```
debt = 0.0

For each chunk:
    if score > +0.1 (comfortable):
        debt += 0.15          ← tension debt accumulates slowly
        consecutive_calm += 1
    
    elif score < -0.4 (dark/tense):
        debt -= 0.40          ← one dark scene discharges more than it cost
        consecutive_calm = 0
    
    if debt > 1.2:            ← FATIGUE_THRESHOLD
        → Flag: "Emotional fatigue risk"
```

**Constants:**
| Constant | Value | Meaning |
|---|---|---|
| `DEBT_BUILD_RATE` | 0.15 | Slow accumulation — a few calm scenes is fine |
| `DEBT_DISCHARGE` | 0.40 | One intense scene resets audience patience significantly |
| `FATIGUE_THRESHOLD` | 1.2 | ~8 consecutive calm chunks triggers the flag |

**The "sawtooth" pattern:** A healthy film's debt curve looks like a sawtooth — it builds, gets discharged, builds again. A flat curve (never discharges) = the film never gives the audience a real emotional hit.

**What RRR's debt curve looked like (first 18 chunks):**
- Chunks 1-8: All dark → debt stays at 0.0 (no accumulation — too negative)
- Chunks 9-11: Positive → debt starts building (0.15 → 0.30 → 0.45)
- Chunk 12: Crash to -0.80 → debt discharged back to 0.05
- Never hit fatigue threshold → RRR's first half is well-paced

**Modifiable:**
- Add `DARK_DEBT` — mirror concept for films that are too dark too long
- Adjust `FATIGUE_THRESHOLD` based on genre (comedies can tolerate longer positive runs)

---

### Innovation 3: `compute_momentum()` — Narrative Rate of Change

**The concept:** The derivative of the sentiment arc. How fast is the emotional tone changing between chunks?

**Algorithm:**
```python
for each consecutive pair:
    delta = score[i] - score[i-1]
    is_spike = abs(delta) > 0.45
```

**Flatness context (`_is_flat_context`):**  
A spike is only dramatically significant if it *breaks* a flat or stable zone. A spike mid-chaos is just noise.

```
Flat zone detected (2 preceding chunks with |score| < 0.18)?
    ↓
Rising spike + broke flatness  →  "Victory/Relief"   ← great storytelling
Falling spike + broke flatness →  "Betrayal/Shock"   ← most dramatic impact
Rising spike + no flat zone    →  "Plateau"           ← diminishing returns
Falling spike + no flat zone   →  "Deepening"         ← compounding darkness
```

**Why this matters:** The Chunk 12 falling spike in RRR (delta = -1.60, from +0.80 to -0.80) was classified as "Deepening" because it followed an emotionally volatile zone (chunks 9-11 were also volatile). If the 2 chunks before chunk 12 had been flat/neutral, it would have been classified as "Betrayal/Shock" — maximum dramatic weight.

**Pacing score bonus:** Well-placed Betrayal/Shock spikes get a `-0.3` penalty reduction in the pacing score — because they indicate strong dramatic structure, not a flaw.

**`avg_momentum`:** Mean of all `|delta|` values.
- High avg_momentum (> 0.4) = volatile film — lots of emotional swings
- Low avg_momentum (< 0.15) = monotonous — nothing changes much

**Modifiable:**
- `SPIKE_THRESHOLD = 0.45` → lower to `0.3` to catch subtler shifts
- `FLAT_CONTEXT_WIN = 2` → increase to 3 for stricter flatness requirement
- `FLAT_CONTEXT_THR = 0.18` → tune how "flat" a chunk needs to be

---

### Standard: `detect_flat_zones()`

**What it does:** Scans for 3+ consecutive chunks where `|score| < 0.2`.

**Why 0.2 is the threshold:** Scores between -0.2 and +0.2 have no strong emotional direction. It's the "dead zone" where neither tension nor warmth is present.

**Why 3 chunks minimum:** One neutral chunk = fine (scene transition). Two = borderline. Three = the film has lost momentum for 15+ minutes — that's when real audiences disengage.

**Output:** List of flat chunk IDs + a Critical flag for each flat run.

**Modifiable:** Raise `min_run` to 4 for stricter detection.

---

### Standard: `validate_climax(chunks, user_climax_chunk)` ← OPT-IN ONLY

**Important:** This function is **only called if the user explicitly provides `--climax <N>`**.  
We removed automatic climax detection because lowest sentiment score ≠ dramatic climax.  
(In RRR, the interval fight scene is a *positive* climax for Bheem even though it ends in arrest.)

**What it does:** Validates that the user-specified climax chunk falls between 60-88% of the film.

```
user says: --climax 28
film has 36 chunks
position = 28/36 = 77.8%  → within 60-88% → ✅ well placed
```

**Screenwriting basis for 60-88%:**
- Save the Cat "All Is Lost" beat: 75%
- Hero's Journey "The Ordeal": 65-75%
- 3-Act Structure climax: 80-85%

**CLI usage:**
```bash
python3 run_pipeline.py --chunks 36 --climax 28
```

---

### LLM Call: `map_hero_journey(chunks, film_title)`

**What it does:** Sends all chunk summaries (ID + label + sample dialogue) to Groq and asks it to map each of the 10 Hero's Journey stages to the most fitting chunk.

**Why LLM and not rule-based:**  
Hero's Journey requires *narrative understanding* — recognising that "Call to Adventure" means a disruption of the ordinary world. You can't detect that from sentiment scores alone. The LLM reads the sample dialogue and infers narrative meaning.

**The 10 stages:**
```
1. Ordinary World      → Peaceful baseline before disruption
2. Call to Adventure   → The inciting incident  
3. Refusal of the Call → Hesitation (sometimes absent)
4. Meeting the Mentor  → Ally, guide, or wisdom figure appears
5. Crossing Threshold  → Hero commits to the journey
6. Tests/Allies/Enemies→ Challenges, friendships, antagonists
7. The Ordeal          → The central crisis, near-death moment
8. The Reward          → Temporary victory or insight
9. The Road Back       → Consequences, renewed pursuit
10. The Return         → Resolution, transformed hero
```

**RRR mapping result:**
```
Ordinary World        → Chunk 1  (tribal life, Malli happy)
Call to Adventure     → Chunk 2  (Malli taken by British)
Meeting the Mentor    → Chunk 9  (Ram-Bheem meet, friendship)
Crossing Threshold    → Chunk 10 (undercover mission deepens)
Tests/Allies/Enemies  → Chunk 12 (Ram vs Bheem tension)
The Ordeal            → Chunk 15 (Bheem's capture attempt)
The Return            → Chunk 18 (interval — Bheem arrested)
```

This is **remarkably accurate** for fully automated analysis with no prior film knowledge.

---

### `detect_pacing()` — Main Orchestrator

**Full call flow:**
```
1. detect_analysis_mode(chunks)         → "full_film" or "scene"
2. compute_ideal_vs_actual(chunks, genre) → rmse, ideal_vs_actual dict
3. compute_tension_debt(chunks)          → debt_curve, debt_flags
4. compute_momentum(chunks)              → momentum, momentum_flags, rising_spikes, falling_spikes
5. detect_flat_zones(chunks)             → flat_ids, flat_flags
6. validate_climax() [if user specified] → climax_chunk, well_placed, climax_flags
7. map_hero_journey() [if full_film]     → hero_map dict
8. Compute pacing_score
9. Build PacingReport
```

**Pacing Score formula:**
```
score = 10.0
  - (RMSE × 3.0)                         # Genre curve deviation
  - (flat_chunk_count × 0.2)             # Dead zones
  - (1.5 if user climax is misplaced)    # Only if --climax provided
  - (min(tension_debt_peak, 2.0) × 0.5) # Sustained positive fatigue
  + (0.3 per Betrayal/Shock spike)       # BONUS for good structure
```

**Why these weights:**
- RMSE × 3.0 is the biggest driver — overall arc shape matters most
- Flat zone penalty is soft (0.2 each) — some quiet scenes are intentional
- Climax penalty (1.5) only fires when user provides it — we trust the user knows their story
- Betrayal/Shock bonus rewards well-placed dramatic structure

---

### `run_pipeline.py` — The Entry Point

Chains all services together. Args:

```bash
python3 run_pipeline.py \
    --chunks 18       # How many 5-min chunks to analyze (default: 18 = first 90 min)
    --climax 28       # Optional: validate climax at chunk 28
    --fast            # Use keyword mode (no API calls)
    --no-micro        # Skip Layer 2 micro analysis (faster)
```

**For demo:** Pre-run with `--chunks 36 --no-micro` to generate full-film results in ~3 min. Then show the pre-generated JSON in Streamlit without needing live API calls during the presentation.

---

## JSON Output Files

### `sentiment_output.json` (MS-3 output, MS-4 input)

```json
{
  "overall_score": -0.317,
  "peak_tension_chunk": 1,
  "peak_joy_chunk": 11,
  "flat_zones": [],
  "act_boundaries": [9, 10, 12],
  "chunks": [
    {
      "chunk_id": 1,
      "macro_score": -0.80,
      "macro_label": "Sorrowful",
      "dominant_emotion": "sadness",
      "emotions": {"joy": 0.0, "trust": 0.1, "sadness": 0.9, ...},
      "trajectory": {"start": 0.43, "mid": 0.26, "end": -0.60, "shape": "Falling"},
      "flagged_lines": ["They've bought your daughter.", "Please give me back my child."],
      "peak_line": "MALLI!",
      "word_count": 111
    },
    ...
  ]
}
```

### `pacing_output.json` (MS-4 output, MS-5 input)

```json
{
  "pacing_score": 7.8,
  "tension_debt_peak": 0.45,
  "avg_momentum": 0.424,
  "curve_deviation": 0.655,
  "flat_zones": [],
  "climax_chunk": null,
  "act_boundaries": [9, 10, 12, 13],
  "hero_journey_map": {"Ordinary World": 1, "Call to Adventure": 2, ...},
  "momentum_timeline": [{"chunk_id": 9, "delta": 0.7, "direction": "Rising", "is_spike": true}],
  "tension_debt_curve": [0.0, 0.0, 0.0, ..., 0.15, 0.30, 0.45, 0.05],
  "flags": [
    {"chunk_id": 10, "type": "misplaced_peak", "severity": "warning",
     "message": "Chunk 10: actual +0.70 vs ideal -0.51", "suggestion": "..."}
  ]
}
```

---

## Quick Modification Reference

| What to change | Where | How |
|---|---|---|
| Chunk duration | `srt_parser.py` | `chunk_minutes=5` parameter |
| LLM models | `sentiment_analyzer.py` | `MODEL_MACRO`, `MODEL_MICRO` |
| Flag sensitivity | `sentiment_analyzer.py` | `intensity > 0.7` threshold |
| Genre ideal curves | `pacing_detector.py` | `IDEAL_CURVES` dict |
| Spike sensitivity | `pacing_detector.py` | `SPIKE_THRESHOLD = 0.45` |
| Flat zone strictness | `pacing_detector.py` | `min_run=3` in `detect_flat_zones()` |
| Tension debt speed | `pacing_detector.py` | `DEBT_BUILD_RATE`, `DEBT_DISCHARGE` |
| Pacing score weights | `pacing_detector.py` | penalty formula in `detect_pacing()` |
| Full film threshold | `pacing_detector.py` | `FULL_FILM_THRESHOLD = 8` chunks |

---

AI Screenplay Analyzer: Complete Architecture Guide
This document provides a comprehensive, deep-dive explanation of the entire AI Screenplay Analyzer pipeline, from file ingestion to the final AI-generated studio verdict. The system is split into 5 Microservices (MS) that mimic the analytical process of a professional Hollywood script reader.

MS-1: Ingestion & Parsing (srt_parser.py)
1. Why the .srt Format?
A raw screenplay PDF is notoriously difficult for a computer to parse accurately due to massive formatting variations across different writers. The SubRip Subtitle (.srt) format solves this: it is the universal standard for timing dialogue to screen time. It provides exactly what the AI needs: dialogue text perfectly mapped to exact timestamps.

2. How Parsing is Done
Sanitization: The parser cleans out messy formatting like HTML tags (<i>, <b>) and normalizes line endings.
Song Filtering: It detects and completely filters out song lyrics (usually denoted by italics or music notes). This is crucial for Indian cinema/Bollywood—if we don't filter them, a dramatic scene might incorrectly register as "Joyful" just because an upbeat background song is playing.
3. The 5-Minute Chunking Strategy
The parser groups the dialogue into 5-minute blocks.

Why 5 minutes? In screenwriting, 1 page of a script equals roughly 1 minute of screen time. Therefore, 5 minutes is ~5 pages. This is the standard cinematic length for a "beat" or a "sequence." It is long enough to establish an emotional tone, but short enough to give us high-resolution tracking across a 3-hour film (yielding exactly 36 chunks for a 180-minute movie like Endgame).
MS-3: Sentiment & Emotion Engine (The NLP Layer)
This engine doesn't just do basic "Positive vs. Negative" sentiment. It uses a Two-Layer LLM Architecture:

1. Macro Analysis (Layer 1)
Uses a heavy LLM (llama-3.3-70b) to analyze the entire 5-minute chunk as a single block.

Valence Score: It assigns an overall score from -1.0 (Maximum Despair/Tension) to +1.0 (Maximum Joy/Triumph).
Trajectory: It determines if the scene is Rising, Falling, or Flat within those 5 minutes.
2. Micro Analysis (Layer 2)
Uses a faster LLM (llama-3.1-8b) to read every individual line of dialogue.

Intensity Flagging: If it spots a line with an intensity over 0.7, it flags it as a "Dramatic Anchor." This ensures the system recognizes brilliant individual lines even if the overall scene is emotionally neutral.
3. The Emotion Heatmap (Plutchik's Wheel)
Binary positive/negative isn't enough. A negative scene could be Sad (a funeral), Angry (a fistfight), or Fearful (a horror chase).

The heatmap uses Plutchik's 8 primary emotions (Joy, Trust, Anticipation, Surprise, Fear, Sadness, Disgust, Anger).
Why this model? It lets us see the "cocktail" of a scene, and mathematically proves if a film is getting emotionally monotonous (e.g., if a script has 20 minutes of non-stop "Anger", the audience will feel burnt out).
MS-4: Conflict & Pacing Detector (The Math Layer)
This is where the magic happens. It takes the sentiment scores and applies mathematical pacing algorithms:

1. Ideal Pacing Curves
How it's calculated: The system holds mathematical arrays representing perfect "Hollywood arcs" for specific genres. (e.g., an Action curve dips at 25%, drops drastically at 75% for the "All is Lost" moment, and spikes at 100%). It uses linear interpolation to map the actual film against this ideal curve and calculates the RMSE (Root Mean Square Error).
What it depicts: How closely the film adheres to expected, structurally sound Hollywood genre pacing.
2. Tension Debt Curve
How it's calculated:
If a scene is calm/happy (score > 0.1), the system adds +0.15 to the debt.
If a scene is dark/tense (score < -0.4), the system discharges -0.40 from the debt.
What it interprets: This tracks audience fatigue. Audiences have an unconscious "budget" for happy scenes. If the debt crosses a threshold of 1.2 (about 8 calm scenes in a row), the audience is bored. They are owed a conflict.
3. Narrative Momentum
How it's calculated: This is the mathematical derivative (rate of change) of the sentiment arc (Score of Chunk 2 - Score of Chunk 1). If the |delta| > 0.45, it triggers a "Spike."
What it interprets:
A massive negative spike that interrupts a calm zone is flagged as a "Betrayal / Shock" (maximum dramatic impact).
A positive spike is a "Victory / Relief."
Tiny bars mean a "Flat Zone" (the movie is spinning its wheels).
4. Hero's Journey Map
The system prompts the LLM to map the scene summaries to Joseph Campbell's 10-stage monomyth (Ordinary World, The Ordeal, The Return, etc.).
What it indicates: It proves whether the protagonist undergoes a complete, recognizable psychological transformation. If stages are missing, the character arc is likely broken.
MS-5: The Critique Engine (Scoring & Rules)
This takes all the math above and calculates exactly how "good" the script is based on strict rules.

How Plot Issues are Detected:
Front-Loaded Drama: Mathematically checks if the absolute lowest tension score occurs in the first 20% of the film. (Unless the genre is "Epic/Sequel", this is a severe warning).
Missing Act 2 Conflict: Checks if the middle third of the chunks ever drop below a -0.3 score. If not, Act 2 is missing a major obstacle.
Repetitive Emotion: Checks if the exact same dominant Plutchik emotion appears 4 chunks in a row.
Unresolved Tension: Checks if the Tension Debt is high at the very end of the film without a climax discharge preceding it.
Score Breakdown & Weights:
Pacing (30%): Starts at 10. It subtracts points for RMSE curve deviation (RMSE * 3.0), tension debt peaks (debt * 0.5), and flat zones (count * 0.2). It adds bonus points if it detects well-structured "Betrayal" momentum spikes.
Emotional Range (25%): Takes Highest Score - Lowest Score. If the spread is huge > 1.2, it gets a 9.0/10. If the film is emotionally flat, it drops to a 4.0/10.
Dialogue (20%): Starts at 8.0. Adds +0.3 for every intense line found in Micro-Analysis. Subtracts -0.2 if a 5-minute chunk has over 400 words of dialogue (meaning it's a massive, clunky exposition dump).
Structure (25%): Starts at 8.0. Subtracts a full -1.0 for critical plot issues (like Missing Act 2 Conflict). Adds a +0.5 bonus if the Hero's Journey map was completely filled out.
The Final Verdict:
The engine takes these four weighted scores to calculate a Final Screenplay Score. Finally, it injects all the data and plot issues into a prompt and sends this to the Groq LLM with strict instructions: "You are a nuanced cinematic analyst. Look at these numbers through the lens of the film's specific genre, and output a 3-sentence verdict and the Top 3 structural fixes."

