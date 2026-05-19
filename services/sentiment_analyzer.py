"""
services/sentiment_analyzer.py  (v2 — Two-Layer + Trajectory)
---------------------------------------------------------------
MS-3: Sentiment & Emotion Analyzer

TWO-LAYER ANALYSIS ARCHITECTURE:
  ┌─────────────────────────────────────────────────────────┐
  │  LAYER 1 — Macro (Chunk-level)       KEY 1 (Groq)      │
  │  • Full 5-min chunk text → 1 LLM call                  │
  │  • Returns: valence score, Plutchik emotion breakdown   │
  │  • Drives: Emotional Arc Line Chart                     │
  ├─────────────────────────────────────────────────────────┤
  │  LAYER 2 — Micro (Line-level)        KEY 2 (Groq)      │
  │  • Each individual dialogue line → batch LLM call      │
  │  • Returns: per-line emotion + intensity                │
  │  • Drives: Critique flagging, peak moment detection     │
  └─────────────────────────────────────────────────────────┘
  +
  TRAJECTORY — splits chunk into thirds, computes arc shape
  • Rising | Falling | V-shape | Inverted-V | Flat | Mixed
  • Drives: Pacing detector (MS-4)

Emotion Model: Plutchik's Wheel (8 primary emotions)
  joy, trust, anticipation, surprise, fear, sadness, disgust, anger
"""

import os
import json
import time
from dataclasses import dataclass, field
from typing import Optional
from groq import Groq
from dotenv import load_dotenv

load_dotenv(override=True)

# ── Two Groq clients — one per API key ──────────────────────────────────────
# KEY 1 → Layer 1 (macro chunk analysis) — higher quality model
# KEY 2 → Layer 2 (micro line analysis)  — faster model, more calls
_CLIENT_MACRO = Groq(api_key=os.getenv("GROQ_API_KEY_1", ""))
_CLIENT_MICRO = Groq(api_key=os.getenv("GROQ_API_KEY_2", ""))

MODEL_MACRO  = "llama-3.3-70b-versatile"   # better reasoning for chunk-level
MODEL_MICRO  = "llama-3.1-8b-instant"      # faster for per-line batch calls

RETRY_DELAY  = 2
MAX_RETRIES  = 3
MICRO_BATCH  = 20   # lines per micro-analysis call

# Plutchik's 8 primary emotions
PLUTCHIK_EMOTIONS = ["joy", "trust", "anticipation", "surprise",
                     "fear", "sadness", "disgust", "anger"]


# ── Data Models ──────────────────────────────────────────────────────────────
@dataclass
class LineSentiment:
    """Micro-level: emotion for a single dialogue line."""
    line_text:  str
    emotion:    str     # one of PLUTCHIK_EMOTIONS
    intensity:  float   # 0.0 (barely) → 1.0 (extremely)
    valence:    float   # -1.0 (negative) → +1.0 (positive)
    is_flagged: bool = False  # True if intensity > 0.7 (noteworthy line)


@dataclass
class TrajectoryArc:
    """Emotional trajectory inside a single chunk (split into thirds)."""
    start_score: float   # avg valence of first third of lines
    mid_score:   float   # avg valence of middle third
    end_score:   float   # avg valence of last third
    shape:       str     # "Rising"|"Falling"|"V-shape"|"Inverted-V"|"Flat"|"Mixed"
    delta:       float   # abs(end - start) — magnitude of change


@dataclass
class ChunkSentiment:
    """Complete emotion data for one scene chunk."""
    chunk_id:          int
    # Layer 1 outputs
    macro_score:       float          # -1.0 to +1.0 overall valence
    macro_label:       str            # "Tense" | "Joyful" | etc.
    dominant_emotion:  str            # strongest Plutchik emotion
    emotions:          dict           # {emotion: score} for all 8
    # Layer 2 outputs
    line_sentiments:   list           # [LineSentiment] per line
    peak_line:         Optional[str]  # most emotionally intense line
    flagged_lines:     list[str]      # lines with intensity > 0.7
    # Trajectory
    trajectory:        Optional[TrajectoryArc] = None
    # Metadata
    word_count:        int   = 0
    sample_text:       str   = ""


@dataclass
class SentimentTimeline:
    chunks:             list[ChunkSentiment]
    overall_score:      float        = 0.0
    peak_tension_chunk: Optional[int] = None
    peak_joy_chunk:     Optional[int] = None
    flat_zones:         list[int]    = field(default_factory=list)
    act_boundaries:     list[int]    = field(default_factory=list)
    arc_shapes:         dict         = field(default_factory=dict)  # {chunk_id: shape}


# ── Layer 1: Macro Chunk Analysis ────────────────────────────────────────────
def _macro_prompt(chunk_id: int, text: str, film_title: str) -> str:
    return f"""You are an expert film critic and emotion analyst.
Analyze the overall emotional tone of this scene chunk (chunk {chunk_id}) from the film "{film_title}".

DIALOGUE:
---
{text[:2000]}
---

Use Plutchik's Wheel of Emotions. Return ONLY valid JSON, no markdown:
{{
  "score": <float -1.0 (very dark/tense) to +1.0 (very joyful/triumphant)>,
  "label": "<Joyful|Hopeful|Romantic|Comic|Neutral|Tense|Fearful|Sorrowful|Wrathful|Triumphant|Climactic>",
  "dominant_emotion": "<joy|trust|anticipation|surprise|fear|sadness|disgust|anger>",
  "emotions": {{
    "joy":          <0.0-1.0>,
    "trust":        <0.0-1.0>,
    "anticipation": <0.0-1.0>,
    "surprise":     <0.0-1.0>,
    "fear":         <0.0-1.0>,
    "sadness":      <0.0-1.0>,
    "disgust":      <0.0-1.0>,
    "anger":        <0.0-1.0>
  }},
  "reasoning": "<one sentence — why this score>"
}}

Scoring guide:
-1.0 = extremely dark, violent, despairing, brutal confrontation
-0.5 = tense, anxious, threatening, betrayal
 0.0 = neutral, expository, transitional dialogue
+0.5 = warm, hopeful, friendship-building, romantic
+1.0 = triumphant, euphoric, joyful reunion, victory celebration"""


def _call_macro(prompt: str) -> Optional[dict]:
    for attempt in range(MAX_RETRIES):
        try:
            resp = _CLIENT_MACRO.chat.completions.create(
                model    = MODEL_MACRO,
                messages = [{"role": "user", "content": prompt}],
                temperature      = 0.15,
                response_format  = {"type": "json_object"},
                max_tokens       = 400,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            print(f"    [Macro L1] Attempt {attempt+1} failed: {e}")
            time.sleep(RETRY_DELAY)
    return None


# ── Layer 2: Micro Line Analysis ─────────────────────────────────────────────
def _micro_prompt(lines: list[str], film_title: str) -> str:
    numbered = "\n".join(f'[{i}] "{line}"' for i, line in enumerate(lines))
    emotions_list = ", ".join(PLUTCHIK_EMOTIONS)
    return f"""You are an expert screenplay analyst. Classify the emotion of each dialogue line from "{film_title}".

LINES:
{numbered}

For each line return its dominant emotion and intensity.
Emotions available: {emotions_list}

Return ONLY valid JSON array, no markdown:
[
  {{
    "index": 0,
    "emotion": "<one of the 8 emotions>",
    "intensity": <0.0 barely emotional to 1.0 extremely emotional>,
    "valence": <-1.0 very negative to +1.0 very positive>
  }},
  ...
]

Rules:
- intensity > 0.7 means this line is noteworthy / dramatically important
- A calm expository line gets intensity ~0.1
- A betrayal / declaration / climactic line gets intensity ~0.9
- valence tracks positive/negative regardless of emotion type
  (e.g. anger is negative valence, joy is positive valence)"""


def _call_micro(lines: list[str], film_title: str) -> Optional[list[dict]]:
    prompt = _micro_prompt(lines, film_title)
    for attempt in range(MAX_RETRIES):
        try:
            resp = _CLIENT_MICRO.chat.completions.create(
                model    = MODEL_MICRO,
                messages = [{"role": "user", "content": prompt}],
                temperature     = 0.1,
                response_format = {"type": "json_object"},
                max_tokens      = 1200,
            )
            raw = resp.choices[0].message.content
            parsed = json.loads(raw)
            # Groq with json_object sometimes wraps in a key
            if isinstance(parsed, dict):
                # Try known keys first, then search for any list value
                if isinstance(parsed.get("lines"), list):
                    parsed = parsed["lines"]
                elif isinstance(parsed.get("results"), list):
                    parsed = parsed["results"]
                else:
                    # Find the first value that's a list
                    found_list = False
                    for v in parsed.values():
                        if isinstance(v, list):
                            parsed = v
                            found_list = True
                            break
                    if not found_list:
                        # Check if keys are string indices: {"0": {...}, "1": {...}, ...}
                        if all(k.isdigit() and isinstance(v, dict) for k, v in parsed.items()):
                            parsed = [
                                {**v, "index": int(k)} for k, v in sorted(parsed.items(), key=lambda x: int(x[0]))
                            ]
                        elif all(k in parsed for k in ("emotion", "intensity")):
                            parsed = [parsed]  # single result returned unwrapped
                        else:
                            print(f"    [Micro L2] Unexpected response structure: {list(parsed.keys())[:5]}")
                            return None
            if not isinstance(parsed, list):
                print(f"    [Micro L2] Response is not a list: {type(parsed)}")
                return None
            return parsed
        except Exception as e:
            print(f"    [Micro L2] Attempt {attempt+1} failed: {e}")
            time.sleep(RETRY_DELAY)
    return None


def _analyze_lines_micro(
    lines: list[str],
    film_title: str
) -> list[LineSentiment]:
    """Run Layer 2 on all lines of a chunk, batching every MICRO_BATCH lines."""
    all_results: list[LineSentiment] = []

    for batch_start in range(0, len(lines), MICRO_BATCH):
        batch = lines[batch_start: batch_start + MICRO_BATCH]
        raw   = _call_micro(batch, film_title)

        if raw is None:
            # Fallback — tag everything as neutral
            for line in batch:
                all_results.append(LineSentiment(
                    line_text="", emotion="surprise",
                    intensity=0.1, valence=0.0
                ))
            continue

        result_map = {r["index"]: r for r in raw if isinstance(r, dict)}
        for i, line in enumerate(batch):
            r = result_map.get(i, {})
            ls = LineSentiment(
                line_text  = line,
                emotion    = r.get("emotion", "surprise"),
                intensity  = float(r.get("intensity", 0.1)),
                valence    = float(r.get("valence", 0.0)),
                is_flagged = float(r.get("intensity", 0.1)) > 0.7
            )
            all_results.append(ls)

        time.sleep(0.3)

    return all_results


# ── Trajectory Analysis ───────────────────────────────────────────────────────
def _compute_trajectory(line_sentiments: list[LineSentiment]) -> Optional[TrajectoryArc]:
    """Split line sentiments into thirds, compute arc shape."""
    if len(line_sentiments) < 6:
        return None  # too few lines to compute trajectory

    n      = len(line_sentiments)
    third  = n // 3

    def avg_valence(sents):
        return sum(s.valence for s in sents) / len(sents) if sents else 0.0

    s = avg_valence(line_sentiments[:third])
    m = avg_valence(line_sentiments[third: 2*third])
    e = avg_valence(line_sentiments[2*third:])

    delta = abs(e - s)

    # Classify arc shape
    if   e > s + 0.25:                          shape = "Rising"
    elif e < s - 0.25:                          shape = "Falling"
    elif m < min(s, e) - 0.2:                   shape = "V-shape"
    elif m > max(s, e) + 0.2:                   shape = "Inverted-V"
    elif delta < 0.12:                           shape = "Flat"
    else:                                        shape = "Mixed"

    return TrajectoryArc(
        start_score = round(s, 3),
        mid_score   = round(m, 3),
        end_score   = round(e, 3),
        shape       = shape,
        delta       = round(delta, 3)
    )


# ── Keyword Fallback (no API) ─────────────────────────────────────────────────
_NEG = {"kill","kills","killed","killing","die","died","dies","death","dead",
        "murder","blood","bloody","fight","fighting","war","hate","hated",
        "rage","attack","attacked","destroy","destroyed","enemy","enemies",
        "pain","painful","suffer","suffering","afraid","fear","feared",
        "evil","prison","prisoner","punish","punished","beaten","beat",
        "monster","betrayal","betrayed","anger","angry","grief","grieve",
        "hurt","wound","wounded","weapon","sword","gun","shoot","shot",
        "threat","threaten","danger","dangerous","cruel","cruelty",
        "arrest","captured","torture","scream","screamed","crying","tears",
        "sacrifice","stolen","slave","oppression","oppress","violent",
        "revenge","avenge","burn","burning","drown","trap","trapped",
        "lost","lose","losing","broke","broken","sorrow","misery",
        "darkness","dark","terror","horrify","shock","chains","whip"}
_POS = {"love","loved","loves","loving","happy","happiness","joy","joyful",
        "laugh","laughed","laughing","smile","smiled","smiling","friend",
        "friends","friendship","together","hope","hopeful","hoping",
        "victory","victorious","win","winning","won","celebrate","celebrating",
        "free","freedom","beautiful","beauty","safe","safely","dance","dancing",
        "thank","thankful","thanks","trust","trusted","family","protect",
        "protecting","brave","bravery","proud","pride","cheer","cheering",
        "unite","united","unity","warm","warmth","kind","kindness",
        "welcome","gentle","peace","peaceful","comfort","embrace",
        "gift","blessing","blessed","dear","darling","precious",
        "wonderful","amazing","delight","delighted","pleased","pleasure",
        "rescue","saved","saving","glory","glorious","honour","hero",
        "brothers","brother","sister","mother","heart","care","caring"}

def _fast_score_chunk(text: str) -> dict:
    words   = text.lower().split()
    neg     = sum(1 for w in words if w.strip(".,!?\"'()") in _NEG)
    pos     = sum(1 for w in words if w.strip(".,!?\"'()") in _POS)
    matched = neg + pos
    if matched == 0:
        raw_score = 0.0
    else:
        # ratio: what fraction of emotional words are positive vs negative
        ratio = (pos - neg) / matched            # -1.0 to +1.0
        # density: what fraction of all words are emotional (0 to 1)
        density = min(matched / max(len(words), 1), 0.5) * 2  # scale up, cap at 1.0
        raw_score = ratio * max(density, 0.3)    # floor density at 0.3 so even sparse matches register
    score   = round(max(-1.0, min(1.0, raw_score)), 3)
    label   = ("Joyful" if score > 0.3 else "Hopeful" if score > 0.0
                else "Neutral" if score > -0.1 else "Tense" if score > -0.4
                else "Sorrowful" if score > -0.7 else "Wrathful")
    dom     = ("joy" if score > 0.3 else "trust" if score > 0.0
               else "surprise" if score > -0.1 else "fear" if score > -0.4
               else "sadness" if score > -0.7 else "anger")
    # Build emotion distribution based on score
    neg_intensity = max(0, -score)
    pos_intensity = max(0, score)
    return {
        "score": score, "label": label,
        "dominant_emotion": dom,
        "emotions": {
            "joy":          round(pos_intensity * 0.8, 2),
            "trust":        round(pos_intensity * 0.5, 2),
            "anticipation": round(abs(score) * 0.3, 2),
            "surprise":     round(0.2 if abs(score) < 0.3 else 0.1, 2),
            "fear":         round(neg_intensity * 0.6, 2),
            "sadness":      round(neg_intensity * 0.7, 2),
            "disgust":      round(neg_intensity * 0.3, 2),
            "anger":        round(neg_intensity * 0.8, 2),
        },
        "reasoning": f"Keyword: {pos} positive, {neg} negative out of {len(words)} words"
    }


# ── Main Entry Point ──────────────────────────────────────────────────────────
def analyze_sentiment(
    chunk_texts:  dict[int, str],      # {chunk_id: dialogue_text} from MS-1
    subtitle_map: dict[int, list] = None,  # {chunk_id: [Subtitle]} for micro
    film_title:   str  = "Film",
    use_llm:      bool = True,
    run_micro:    bool = True,          # set False to skip Layer 2
    verbose:      bool = True
) -> SentimentTimeline:
    """
    MS-3 main entry point.

    Args:
        chunk_texts:  {chunk_id: full text} from get_scene_chunk_texts()
        subtitle_map: {chunk_id: [Subtitle]} for micro line-level analysis
        film_title:   for LLM context
        use_llm:      False = keyword fallback (no API)
        run_micro:    False = skip Layer 2 (faster, less rich)
        verbose:      print progress

    Returns:
        SentimentTimeline
    """
    mode = "LLM Two-Layer (Groq)" if use_llm else "Fast Keyword"
    if verbose:
        print(f"\n{'='*60}")
        print(f"  🎭 MS-3: Sentiment Analyzer — {mode}")
        print(f"  Film: {film_title} | Chunks: {len(chunk_texts)}")
        print(f"  Layer 1 model : {MODEL_MACRO}  (Key 1)")
        print(f"  Layer 2 model : {MODEL_MICRO}  (Key 2)")
        print(f"{'='*60}\n")

    chunk_sentiments: list[ChunkSentiment] = []

    for chunk_id, text in sorted(chunk_texts.items()):
        lines      = [s.text for s in subtitle_map[chunk_id]] if subtitle_map else text.split(". ")
        word_count = len(text.split())

        if verbose:
            print(f"  Chunk {chunk_id:>3} ({word_count:>4}w, {len(lines):>3} lines)")

        # ── LAYER 1: Macro chunk analysis ────────────────────────────────────
        if verbose: print(f"    L1 macro ...", end=" ", flush=True)

        if use_llm:
            macro_raw = _call_macro(_macro_prompt(chunk_id, text, film_title), )
            if macro_raw is None:
                macro_raw = _fast_score_chunk(text)
            time.sleep(0.4)
        else:
            macro_raw = _fast_score_chunk(text)

        if verbose:
            score = float(macro_raw.get("score", 0.0))
            bar   = _bar(score)
            print(f"{bar} {score:+.2f} [{macro_raw.get('label','?')}] "
                  f"dom={macro_raw.get('dominant_emotion','?')}")

        # ── LAYER 2: Micro line analysis ─────────────────────────────────────
        line_sents: list[LineSentiment] = []
        flagged: list[str] = []
        peak_line: Optional[str] = None

        if use_llm and run_micro and lines:
            if verbose: print(f"    L2 micro ({len(lines)} lines) ...", end=" ", flush=True)
            line_sents = _analyze_lines_micro(lines, film_title)
            flagged    = [ls.line_text for ls in line_sents if ls.is_flagged]
            if line_sents:
                peak_line = max(line_sents, key=lambda ls: ls.intensity).line_text
            if verbose:
                print(f"✅  {len(flagged)} high-intensity lines flagged")

        # ── TRAJECTORY ───────────────────────────────────────────────────────
        trajectory = _compute_trajectory(line_sents) if line_sents else None
        if verbose and trajectory:
            print(f"    Trajectory: {trajectory.shape:12s}  "
                  f"({trajectory.start_score:+.2f} → "
                  f"{trajectory.mid_score:+.2f} → "
                  f"{trajectory.end_score:+.2f})")

        cs = ChunkSentiment(
            chunk_id         = chunk_id,
            macro_score      = float(macro_raw.get("score", 0.0)),
            macro_label      = macro_raw.get("label", "Neutral"),
            dominant_emotion = macro_raw.get("dominant_emotion", "surprise"),
            emotions         = macro_raw.get("emotions", {e: 0.0 for e in PLUTCHIK_EMOTIONS}),
            line_sentiments  = line_sents,
            peak_line        = peak_line,
            flagged_lines    = flagged,
            trajectory       = trajectory,
            word_count       = word_count,
            sample_text      = text[:120]
        )
        chunk_sentiments.append(cs)

        if verbose: print()

    # ── Build timeline-level analytics ───────────────────────────────────────
    scores = [c.macro_score for c in chunk_sentiments]
    overall = sum(scores) / len(scores) if scores else 0.0

    peak_tension = min(chunk_sentiments, key=lambda c: c.macro_score)
    peak_joy     = max(chunk_sentiments, key=lambda c: c.macro_score)

    flat_zones      = _detect_flat_zones(chunk_sentiments, threshold=0.2, min_run=3)
    act_boundaries  = _detect_act_boundaries(chunk_sentiments, delta=0.4)
    arc_shapes      = {c.chunk_id: c.trajectory.shape
                       for c in chunk_sentiments if c.trajectory}

    timeline = SentimentTimeline(
        chunks              = chunk_sentiments,
        overall_score       = round(overall, 3),
        peak_tension_chunk  = peak_tension.chunk_id,
        peak_joy_chunk      = peak_joy.chunk_id,
        flat_zones          = flat_zones,
        act_boundaries      = act_boundaries,
        arc_shapes          = arc_shapes,
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"  ✅ Analysis complete!")
        print(f"     Overall tone    : {overall:+.3f}")
        print(f"     Peak tension    : Chunk {peak_tension.chunk_id} "
              f"({peak_tension.macro_score:+.2f} — {peak_tension.macro_label})")
        print(f"     Peak joy        : Chunk {peak_joy.chunk_id} "
              f"({peak_joy.macro_score:+.2f} — {peak_joy.macro_label})")
        print(f"     Flat zones      : {flat_zones or 'None'}")
        print(f"     Act boundaries  : chunks {act_boundaries}")
        all_flagged = sum(len(c.flagged_lines) for c in chunk_sentiments)
        print(f"     High-intensity lines flagged : {all_flagged}")
        print(f"{'='*60}")

    return timeline


# ── Helpers ───────────────────────────────────────────────────────────────────
def _bar(score: float, w: int = 22) -> str:
    center = w // 2
    pos    = int((score + 1.0) / 2.0 * w)
    bar    = ["-"] * w
    bar[center] = "|"
    if 0 <= pos < w:
        bar[pos] = "█"
    return "[" + "".join(bar) + "]"


def _detect_flat_zones(chunks, threshold, min_run) -> list[int]:
    flat, run = [], []
    for c in chunks:
        if abs(c.macro_score) < threshold:
            run.append(c.chunk_id)
        else:
            if len(run) >= min_run:
                flat.extend(run)
            run = []
    if len(run) >= min_run:
        flat.extend(run)
    return flat


def _detect_act_boundaries(chunks, delta) -> list[int]:
    return [chunks[i].chunk_id
            for i in range(1, len(chunks))
            if abs(chunks[i].macro_score - chunks[i-1].macro_score) >= delta]


def timeline_to_dict(timeline: SentimentTimeline) -> dict:
    """Serialise to plain dict for JSON storage and Streamlit charts."""
    return {
        "overall_score":       timeline.overall_score,
        "peak_tension_chunk":  timeline.peak_tension_chunk,
        "peak_joy_chunk":      timeline.peak_joy_chunk,
        "flat_zones":          timeline.flat_zones,
        "act_boundaries":      timeline.act_boundaries,
        "arc_shapes":          timeline.arc_shapes,
        "chunks": [
            {
                "chunk_id":          c.chunk_id,
                "macro_score":       c.macro_score,
                "macro_label":       c.macro_label,
                "dominant_emotion":  c.dominant_emotion,
                "emotions":          c.emotions,
                "trajectory": {
                    "start": c.trajectory.start_score,
                    "mid":   c.trajectory.mid_score,
                    "end":   c.trajectory.end_score,
                    "shape": c.trajectory.shape,
                    "delta": c.trajectory.delta,
                } if c.trajectory else None,
                "flagged_lines":     c.flagged_lines,
                "peak_line":         c.peak_line,
                "word_count":        c.word_count,
            }
            for c in timeline.chunks
        ]
    }


# ── Quick Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, ROOT)
    from srt_parser import parse_srt, assign_scene_chunks, get_dialogue_only, get_scene_chunk_texts

    SRT = os.path.join(ROOT, "RRR 2022 JPN UHD en full.srt")
    print("Loading RRR subtitles...")
    subs     = parse_srt(SRT)
    subs     = assign_scene_chunks(subs, chunk_minutes=5)
    dialogue = get_dialogue_only(subs)
    chunks   = get_scene_chunk_texts(dialogue)

    # Build subtitle_map for micro analysis: {chunk_id: [Subtitle]}
    sub_map: dict[int, list] = {}
    for s in dialogue:
        sub_map.setdefault(s.scene_chunk, []).append(s)

    # Test LLM two-layer on first 4 chunks (saves API quota)
    test_chunks = {k: v for k, v in chunks.items() if k <= 4}
    test_submap = {k: v for k, v in sub_map.items() if k <= 4}

    print("\n=== TWO-LAYER LLM MODE (Groq, first 4 chunks) ===")
    timeline = analyze_sentiment(
        chunk_texts  = test_chunks,
        subtitle_map = test_submap,
        film_title   = "RRR",
        use_llm      = True,
        run_micro    = True,
        verbose      = True
    )

    # Save
    out_path = os.path.join(ROOT, "sentiment_output.json")
    with open(out_path, "w") as f:
        json.dump(timeline_to_dict(timeline), f, indent=2)
    print(f"\n💾 Saved → {out_path}")

    # Print some flagged lines
    print("\n🚨 HIGH-INTENSITY LINES DETECTED:")
    for c in timeline.chunks:
        for line in c.flagged_lines:
            print(f"  [Chunk {c.chunk_id}] {line[:80]}")
