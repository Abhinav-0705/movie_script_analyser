"""
services/pacing_detector.py
----------------------------
MS-4: Conflict & Pacing Detector

THREE INNOVATIVE FEATURES (beyond standard analysis):

1. IDEAL PACING CURVE — genre-specific ideal vs actual arc comparison
   Scores deviation and flags misplaced peaks/troughs

2. TENSION DEBT ACCUMULATOR — novel concept:
   Sustained positivity builds "debt" the film must repay with darkness.
   Tracks running debt and flags emotional fatigue risk.

3. NARRATIVE MOMENTUM — derivative of sentiment arc:
   Rate-of-change reveals plot twists, slow burns, and abrupt tonal shifts.

Standard features:
- Flat zone detection (3+ consecutive low-variance chunks)
- Act boundary detection
- Climax placement validation
- Hero's Journey stage mapping (via Groq)
- Overall pacing score
"""

import os
import json
import math
import time
from dataclasses import dataclass, field
from typing import Optional
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY_1", ""))
MODEL   = "llama-3.3-70b-versatile"

# ── Ideal Pacing Curves per Genre ─────────────────────────────────────────────
# Format: list of (position_fraction, ideal_score) tuples
# position_fraction: 0.0 = film start, 1.0 = film end
IDEAL_CURVES = {
    "action":    [(0.0,0.1),(0.1,-0.3),(0.25,-0.5),(0.4,-0.2),(0.5,-0.6),
                  (0.6,-0.3),(0.75,-0.9),(0.85,-0.4),(1.0,0.6)],
    "epic":      [(0.0,-0.2),(0.15,-0.5),(0.3,-0.3),(0.5,-0.7),(0.6,-0.4),
                  (0.75,-0.95),(0.9,-0.3),(1.0,0.7)],
    "romance":   [(0.0,0.3),(0.25,0.6),(0.4,-0.2),(0.5,0.1),(0.65,-0.5),
                  (0.8,0.3),(1.0,0.8)],
    "thriller":  [(0.0,-0.1),(0.2,-0.4),(0.4,-0.6),(0.55,-0.3),(0.7,-0.8),
                  (0.85,-0.5),(1.0,-0.2)],
    "drama":     [(0.0,0.1),(0.3,-0.2),(0.5,-0.5),(0.7,-0.3),(0.85,-0.7),
                  (1.0,0.2)],
    "comedy":    [(0.0,0.4),(0.2,0.6),(0.4,0.2),(0.6,0.5),(0.8,-0.1),(1.0,0.7)],
    "default":   [(0.0,0.0),(0.25,-0.3),(0.5,-0.5),(0.75,-0.8),(1.0,0.3)],
}

# ── Hero's Journey 10 Stages ──────────────────────────────────────────────────
HERO_STAGES = [
    "Ordinary World", "Call to Adventure", "Refusal of the Call",
    "Meeting the Mentor", "Crossing the Threshold", "Tests/Allies/Enemies",
    "The Ordeal", "The Reward", "The Road Back", "The Return"
]


# ── Data Models ───────────────────────────────────────────────────────────────
@dataclass
class PacingFlag:
    chunk_id:    int
    flag_type:   str    # "flat_zone"|"tension_debt"|"momentum_spike"|"misplaced_peak"|"missing_climax"
    severity:    str    # "critical"|"warning"|"info"
    message:     str
    suggestion:  str


@dataclass
class MomentumPoint:
    chunk_id:   int
    delta:      float   # sentiment change from previous chunk
    direction:  str     # "Rising"|"Falling"|"Stable"
    is_spike:   bool    # |delta| > SPIKE_THRESHOLD


@dataclass
class PacingReport:
    # Core scores
    pacing_score:        float           # 0-10 (10 = perfect pacing)
    tension_debt_peak:   float           # max tension debt reached
    avg_momentum:        float           # mean |delta| across film
    curve_deviation:     float           # RMSE vs ideal curve

    # Flags
    flags:               list[PacingFlag]

    # Innovative outputs
    momentum_timeline:   list[MomentumPoint]
    tension_debt_curve:  list[float]     # running debt per chunk
    ideal_vs_actual:     dict            # {chunk_id: {actual, ideal, delta}}

    # Hero's Journey
    hero_journey_map:    dict            # {stage_name: chunk_id}
    act_boundaries:      list[int]       # chunk IDs where acts likely change

    # Summary
    flat_zones:          list[int]
    climax_chunk:        Optional[int]
    climax_well_placed:  bool


# ── Innovation 1: Ideal Pacing Curve ─────────────────────────────────────────
def _interpolate_ideal(position: float, genre: str) -> float:
    """Linear interpolation of ideal score at a given film position (0-1)."""
    key    = genre.lower().split("/")[0]
    curve  = IDEAL_CURVES.get(key, IDEAL_CURVES["default"])

    for i in range(len(curve) - 1):
        p0, s0 = curve[i]
        p1, s1 = curve[i + 1]
        if p0 <= position <= p1:
            t = (position - p0) / (p1 - p0)
            return s0 + t * (s1 - s0)
    return curve[-1][1]


def compute_ideal_vs_actual(chunks: list, genre: str) -> dict:
    """
    Compare actual sentiment scores to genre-ideal curve.
    Returns per-chunk dict and overall RMSE deviation score.
    """
    n = len(chunks)
    comparison = {}
    sq_errors  = []

    for i, chunk in enumerate(chunks):
        pos    = i / max(n - 1, 1)
        ideal  = _interpolate_ideal(pos, genre)
        actual = chunk["macro_score"]
        delta  = actual - ideal

        comparison[chunk["chunk_id"]] = {
            "actual": round(actual, 3),
            "ideal":  round(ideal, 3),
            "delta":  round(delta, 3),
        }
        sq_errors.append(delta ** 2)

    rmse = math.sqrt(sum(sq_errors) / len(sq_errors)) if sq_errors else 0.0
    return comparison, round(rmse, 3)


# ── Innovation 2: Tension Debt Accumulator ───────────────────────────────────
DEBT_BUILD_RATE   = 0.15   # debt added per chunk when score > DEBT_THRESHOLD
DEBT_DISCHARGE    = 0.40   # debt removed per chunk when score < -0.4 (tense scene)
DEBT_THRESHOLD    = 0.1    # score above this = "too comfortable", builds debt
FATIGUE_THRESHOLD = 1.2    # debt above this = flag emotional fatigue risk

def compute_tension_debt(chunks: list) -> tuple[list[float], list[PacingFlag]]:
    """
    Track running tension debt. Positive/neutral scenes build debt;
    tense/dark scenes discharge it.
    """
    debt   = 0.0
    curve  = []
    flags  = []

    consecutive_calm = 0

    for chunk in chunks:
        score = chunk["macro_score"]

        if score > DEBT_THRESHOLD:
            debt += DEBT_BUILD_RATE
            consecutive_calm += 1
        elif score < -0.4:
            debt = max(0.0, debt - DEBT_DISCHARGE)
            consecutive_calm = 0
        else:
            consecutive_calm += 1

        curve.append(round(debt, 3))

        if debt > FATIGUE_THRESHOLD:
            flags.append(PacingFlag(
                chunk_id   = chunk["chunk_id"],
                flag_type  = "tension_debt",
                severity   = "warning",
                message    = f"Tension debt = {debt:.2f} — {consecutive_calm} consecutive calm chunks",
                suggestion = "Introduce a conflict, obstacle, or revelation in the next scene to discharge audience tension debt"
            ))

    return curve, flags


# ── Innovation 3: Narrative Momentum ─────────────────────────────────────────
SPIKE_THRESHOLD    = 0.45   # |delta| > this = momentum spike
FLAT_CONTEXT_WIN   = 2      # how many chunks before spike to check for flatness
FLAT_CONTEXT_THR   = 0.18   # chunk is "flat" if |score| < this


def _is_flat_context(chunks: list, spike_idx: int) -> bool:
    """
    Returns True if the chunks immediately BEFORE a spike index were flat.
    A spike that breaks flatness is dramatically more significant.
    """
    start = max(0, spike_idx - FLAT_CONTEXT_WIN)
    context_chunks = chunks[start:spike_idx]
    if not context_chunks:
        return False
    return all(abs(c["macro_score"]) < FLAT_CONTEXT_THR for c in context_chunks)


def compute_momentum(chunks: list) -> tuple[list[MomentumPoint], list[PacingFlag], float, list[int], list[int]]:
    """
    Compute rate-of-change of sentiment between consecutive chunks.
    Spikes indicate plot twists, revelations, or tonal whiplash.
    """
    points = []
    flags  = []
    scores = [c["macro_score"] for c in chunks]

    rising_spikes  = []   # victories, reversals, relief moments
    falling_spikes = []   # betrayals, losses, shocks

    for i in range(1, len(chunks)):
        delta     = scores[i] - scores[i - 1]
        direction = "Rising" if delta > 0.1 else "Falling" if delta < -0.1 else "Stable"
        is_spike  = abs(delta) > SPIKE_THRESHOLD

        mp = MomentumPoint(
            chunk_id  = chunks[i]["chunk_id"],
            delta     = round(delta, 3),
            direction = direction,
            is_spike  = is_spike
        )
        points.append(mp)

        if not is_spike:
            continue

        # ── Check flatness context ────────────────────────────────────────
        # A spike that BREAKS a flat zone is far more dramatically significant
        # than a spike mid-chaos (which is just noise in an already volatile arc)
        broke_flatness = _is_flat_context(chunks, i)

        if delta > 0:   # RISING spike
            rising_spikes.append(chunks[i]["chunk_id"])
            if broke_flatness:
                # Flat → sudden positive = Victory / Relief / Reunion
                # This is great storytelling — gives audiences a reward
                # after sustained tension or calm
                severity = "info"
                label    = "Victory/Relief spike"
                msg      = (f"Rising spike (\u0394{delta:+.2f}) at chunk {chunks[i]['chunk_id']} "
                            f"breaks a flat zone \u2014 strong audience reward moment")
                sug      = ("This is a well-placed positive payoff after sustained calm. "
                            "Ensure the preceding flat zone was intentional setup, not dead pacing.")
            else:
                # Rising spike mid-positive = plateau / diminishing returns
                severity = "info"
                label    = "Rising spike (volatile zone)"
                msg      = (f"Rising spike (\u0394{delta:+.2f}) at chunk {chunks[i]['chunk_id']} "
                            f"\u2014 positive surge in already active emotional zone")
                sug      = ("Scene escalates positively but audience may be numbed "
                            "if surrounded by other high-energy moments. Consider pacing.")

        else:           # FALLING spike
            falling_spikes.append(chunks[i]["chunk_id"])
            if broke_flatness:
                # Flat → sudden negative = Betrayal / Shock / Revelation
                # Most dramatically powerful moment type — subverts audience comfort
                severity = "warning"
                label    = "Betrayal/Shock spike"
                msg      = (f"Falling spike (\u0394{delta:+.2f}) at chunk {chunks[i]['chunk_id']} "
                            f"breaks a flat zone \u2014 high-impact shock or betrayal moment")
                sug      = ("Maximum dramatic impact: audience was comfortable, now blindsided. "
                            "Verify this is the intended emotional punch. "
                            "If unintentional, the preceding calm may need more tension seeding.")
            else:
                # Falling spike in already dark zone = deepening darkness
                severity = "info"
                label    = "Deepening spike"
                msg      = (f"Falling spike (\u0394{delta:+.2f}) at chunk {chunks[i]['chunk_id']} "
                            f"\u2014 emotional descent continues in already dark zone")
                sug      = ("Film deepens its darkness. Ensure this compounds earlier tension "
                            "rather than feeling repetitive. Audience needs at least one "
                            "micro-relief before the next dark beat.")

        flags.append(PacingFlag(
            chunk_id   = chunks[i]["chunk_id"],
            flag_type  = "momentum_spike",
            severity   = severity,
            message    = f"[{label}] {msg}",
            suggestion = sug
        ))

    avg_momentum = sum(abs(p.delta) for p in points) / len(points) if points else 0.0
    return points, flags, round(avg_momentum, 3), rising_spikes, falling_spikes


# ── Standard: Flat Zone Detection ────────────────────────────────────────────
# ── Mode Detection ───────────────────────────────────────────────────────────
FULL_FILM_THRESHOLD = 8   # >= 8 chunks = treat as full film

def detect_analysis_mode(chunks: list) -> str:
    """
    Auto-detect whether we're analyzing a full film or a specific scene excerpt.

    Full film  (>= 8 chunks / ~40 min): All features valid — climax, Hero's Journey,
                                         ideal curve comparison, tension debt
    Scene mode (<  8 chunks / < 40 min): Climax & Hero's Journey disabled.
                                          Focus: momentum, flat zones within the scene,
                                          tension debt as scene-level metric.
    Rationale: Climax position (65-85% through film) is meaningless for a 3-scene
               excerpt. Hero's Journey requires a full narrative arc.
               But tension debt and momentum are meaningful at ANY granularity.
    """
    n = len(chunks)
    mode = "full_film" if n >= FULL_FILM_THRESHOLD else "scene"
    return mode


def detect_flat_zones(chunks: list, threshold=0.2, min_run=3) -> tuple[list[int], list[PacingFlag]]:
    flat_ids = []
    flags    = []
    run      = []

    for c in chunks:
        if abs(c["macro_score"]) < threshold:
            run.append(c["chunk_id"])
        else:
            if len(run) >= min_run:
                flat_ids.extend(run)
                flags.append(PacingFlag(
                    chunk_id   = run[0],
                    flag_type  = "flat_zone",
                    severity   = "critical",
                    message    = f"Chunks {run[0]}–{run[-1]}: {len(run)} consecutive scenes lack emotional tension",
                    suggestion = "Add a confrontation, obstacle, or revelation. Consider merging these scenes or cutting one entirely."
                ))
            run = []

    if len(run) >= min_run:
        flat_ids.extend(run)
        flags.append(PacingFlag(
            chunk_id   = run[0],
            flag_type  = "flat_zone",
            severity   = "critical",
            message    = f"Chunks {run[0]}–{run[-1]}: {len(run)} consecutive scenes lack tension",
            suggestion = "The film ends without building sufficient emotional intensity."
        ))

    return flat_ids, flags


# ── Standard: Climax Validation ──────────────────────────────────────────────
def validate_climax(chunks: list, user_climax_chunk: int) -> tuple[int, bool, list[PacingFlag]]:
    """
    Validate placement of a USER-SPECIFIED climax chunk.
    Only called when the user explicitly tells us which chunk is the climax.
    We do NOT auto-detect climax — lowest sentiment ≠ dramatic climax.
    """
    n   = len(chunks)
    pos = user_climax_chunk / n

    # Ideal: climax between 60-88% of film
    well_placed = 0.60 <= pos <= 0.88
    flags       = []

    if not well_placed:
        if pos < 0.60:
            msg = (f"Your specified climax (chunk {user_climax_chunk}) is at "
                   f"{pos:.0%} into the film — earlier than ideal (60–88%).")
            sug = "Consider restructuring Act 2 to build more slowly before the climax."
        else:
            msg = (f"Your specified climax (chunk {user_climax_chunk}) is at "
                   f"{pos:.0%} into the film — later than ideal (60–88%).")
            sug = "Resolution feels compressed. Try moving the climax earlier to allow more denouement."

        flags.append(PacingFlag(
            chunk_id   = user_climax_chunk,
            flag_type  = "misplaced_peak",
            severity   = "warning",
            message    = msg,
            suggestion = sug
        ))

    return user_climax_chunk, well_placed, flags


# ── Hero's Journey Mapping (LLM) ─────────────────────────────────────────────
def map_hero_journey(chunks: list, film_title: str) -> dict:
    """
    Ask Groq to map each Hero's Journey stage to a chunk ID.
    Returns {stage_name: chunk_id}
    """
    chunk_summaries = "\n".join(
        f"Chunk {c['chunk_id']} ({c.get('macro_label','?')}, score {c['macro_score']:+.2f}): "
        f"{c.get('sample_text','')[:80]}"
        for c in chunks[:36]
    )

    prompt = f"""You are a film narrative analyst. Map the Hero's Journey stages to scene chunks from "{film_title}".

HERO'S JOURNEY STAGES (in order):
{chr(10).join(f'{i+1}. {s}' for i, s in enumerate(HERO_STAGES))}

SCENE CHUNKS (chunk_id, emotional label, sample dialogue):
{chunk_summaries}

Return ONLY valid JSON mapping each stage to the most likely chunk_id:
{{
  "Ordinary World": <chunk_id>,
  "Call to Adventure": <chunk_id>,
  "Refusal of the Call": <chunk_id or null>,
  "Meeting the Mentor": <chunk_id or null>,
  "Crossing the Threshold": <chunk_id>,
  "Tests/Allies/Enemies": <chunk_id>,
  "The Ordeal": <chunk_id>,
  "The Reward": <chunk_id>,
  "The Road Back": <chunk_id or null>,
  "The Return": <chunk_id>
}}"""

    try:
        resp = _CLIENT.chat.completions.create(
            model           = MODEL,
            messages        = [{"role": "user", "content": prompt}],
            temperature     = 0.1,
            response_format = {"type": "json_object"},
            max_tokens      = 300,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"  ⚠️  Hero's Journey mapping failed: {e}")
        return {}


# ── Main Entry Point ──────────────────────────────────────────────────────────
def detect_pacing(
    timeline_dict:     dict,
    film_title:        str  = "Film",
    genre:             str  = "drama",
    run_hero_map:      bool = True,
    force_mode:        str  = None,   # "full_film" | "scene" | None (auto-detect)
    user_climax_chunk: int  = None,   # user-specified climax chunk (None = skip validation)
    verbose:           bool = True
) -> PacingReport:
    """
    MS-4 main entry point.
    Takes the serialised SentimentTimeline dict from MS-3.
    Returns a PacingReport with all flags and scores.
    """
    chunks = timeline_dict["chunks"]

    mode = force_mode or detect_analysis_mode(chunks)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  🎬 MS-4: Pacing & Conflict Detector")
        print(f"  Film: {film_title} | Genre: {genre}")
        print(f"  Chunks: {len(chunks)} | Mode: {mode.upper().replace('_',' ')}")
        if mode == "scene":
            print(f"  ⚠️  Scene mode: climax validation & Hero's Journey disabled")
            print(f"       (< {FULL_FILM_THRESHOLD} chunks — treating as scene excerpt, not full film)")
        print(f"{'='*60}\n")

    all_flags = []

    # ── Innovation 1: Ideal curve comparison ─────────────────────────────────
    if verbose: print("  📈 Computing ideal pacing curve deviation...", end=" ")
    ideal_vs_actual, rmse = compute_ideal_vs_actual(chunks, genre)
    if verbose: print(f"RMSE = {rmse:.3f}")

    # Flag significant deviations
    for cid, vals in ideal_vs_actual.items():
        if abs(vals["delta"]) > 0.5:
            all_flags.append(PacingFlag(
                chunk_id   = cid,
                flag_type  = "misplaced_peak",
                severity   = "warning",
                message    = f"Chunk {cid}: actual {vals['actual']:+.2f} vs ideal {vals['ideal']:+.2f} (Δ{vals['delta']:+.2f})",
                suggestion = f"For a {genre} film, this scene should feel {'darker' if vals['delta'] > 0 else 'more uplifting'} at this point in the story."
            ))

    # ── Innovation 2: Tension debt ────────────────────────────────────────────
    if verbose: print("  💳 Computing tension debt curve...", end=" ")
    debt_curve, debt_flags = compute_tension_debt(chunks)
    all_flags.extend(debt_flags)
    if verbose: print(f"peak debt = {max(debt_curve):.2f}")

    # ── Innovation 3: Narrative momentum ─────────────────────────────────────
    if verbose: print("  ⚡ Computing narrative momentum...", end=" ")
    momentum, momentum_flags, avg_momentum, rising_spikes, falling_spikes = compute_momentum(chunks)
    all_flags.extend(momentum_flags)
    if verbose:
        print(f"avg={avg_momentum:.3f} | "
              f"↗ rising={len(rising_spikes)} "
              f"↘ falling={len(falling_spikes)} spikes")

    # ── Flat zone detection ───────────────────────────────────────────────────
    if verbose: print("  🟰  Detecting flat zones...", end=" ")
    flat_ids, flat_flags = detect_flat_zones(chunks)
    all_flags.extend(flat_flags)
    if verbose: print(f"{len(flat_ids)} flat chunks")

    # ── Climax validation (only if user specified it) ────────────────────────
    climax_chunk, well_placed = user_climax_chunk, False
    if user_climax_chunk is not None and mode == "full_film":
        if verbose: print(f"  🔺 Validating user-specified climax (chunk {user_climax_chunk})...", end=" ")
        climax_chunk, well_placed, climax_flags = validate_climax(chunks, user_climax_chunk)
        all_flags.extend(climax_flags)
        if verbose:
            pos = user_climax_chunk / len(chunks)
            print(f"({pos:.0%}) — {'✅ well placed' if well_placed else '⚠️  off-position'}")
    elif verbose:
        print(f"  🔺 Climax validation: skipped (specify --climax <chunk> to enable)")

    # ── Act boundaries (from MS-3 data) ──────────────────────────────────────
    act_boundaries = timeline_dict.get("act_boundaries", [])

    # ── Hero's Journey mapping ────────────────────────────────────────────────
    hero_map = {}
    if run_hero_map and mode == "full_film":
        if verbose: print("  🗺️  Mapping Hero's Journey (Groq)...", end=" ", flush=True)
        hero_map = map_hero_journey(chunks, film_title)
        if verbose: print(f"✅  {len(hero_map)} stages mapped")

    # ── Overall pacing score ──────────────────────────────────────────────────
    penalty = 0.0
    penalty += rmse * 3.0                                     # curve deviation
    penalty += len(flat_ids) * 0.2                            # flat chunks
    # Climax penalty only applies when user specified a climax AND it's misplaced
    if user_climax_chunk is not None and mode == "full_film" and not well_placed:
        penalty += 1.5
    penalty += min(max(debt_curve), 2.0) * 0.5                # tension debt
    # Bonus: well-placed betrayal/shock spikes (falling spikes breaking flatness)
    # These are signs of good dramatic structure — reward them
    dramatic_spikes = sum(
        1 for f in all_flags
        if f.flag_type == "momentum_spike" and "Betrayal" in f.message
    )
    penalty -= dramatic_spikes * 0.3                          # up to -0.9 bonus
    pacing_score = max(0.0, min(10.0, 10.0 - penalty))

    report = PacingReport(
        pacing_score       = round(pacing_score, 2),
        tension_debt_peak  = round(max(debt_curve), 3),
        avg_momentum       = avg_momentum,
        curve_deviation    = rmse,
        flags              = sorted(all_flags, key=lambda f: ("critical","warning","info").index(f.severity)),
        momentum_timeline  = momentum,
        tension_debt_curve = debt_curve,
        ideal_vs_actual    = ideal_vs_actual,
        hero_journey_map   = hero_map,
        act_boundaries     = act_boundaries,
        flat_zones         = flat_ids,
        climax_chunk       = climax_chunk,
        climax_well_placed = well_placed,
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"  ✅ Pacing analysis complete!")
        print(f"     Pacing score     : {pacing_score:.1f} / 10")
        print(f"     Curve deviation  : {rmse:.3f} (vs ideal {genre} arc)")
        print(f"     Tension debt peak: {max(debt_curve):.2f}")
        print(f"     Avg momentum     : {avg_momentum:.3f}")
        print(f"     Total flags      : {len(all_flags)} "
              f"({sum(1 for f in all_flags if f.severity=='critical')} critical, "
              f"{sum(1 for f in all_flags if f.severity=='warning')} warnings)")
        print(f"     Rising spikes    : {len(rising_spikes)} chunks {rising_spikes}")
        print(f"     Falling spikes   : {len(falling_spikes)} chunks {falling_spikes}")
        print(f"\n  🚩 FLAGS:")
        for f in report.flags[:10]:
            icon = "🔴" if f.severity == "critical" else "⚠️ " if f.severity == "warning" else "ℹ️ "
            print(f"     {icon} [{f.flag_type}] {f.message[:90]}")
        print(f"{'='*60}")

    return report


def report_to_dict(report: PacingReport) -> dict:
    return {
        "pacing_score":       report.pacing_score,
        "tension_debt_peak":  report.tension_debt_peak,
        "avg_momentum":       report.avg_momentum,
        "curve_deviation":    report.curve_deviation,
        "flat_zones":         report.flat_zones,
        "climax_chunk":       report.climax_chunk,
        "climax_well_placed": report.climax_well_placed,
        "act_boundaries":     report.act_boundaries,
        "hero_journey_map":   report.hero_journey_map,
        "tension_debt_curve": report.tension_debt_curve,
        "ideal_vs_actual":    {str(k): v for k, v in report.ideal_vs_actual.items()},
        "momentum_timeline":  [
            {"chunk_id": m.chunk_id, "delta": m.delta,
             "direction": m.direction, "is_spike": m.is_spike}
            for m in report.momentum_timeline
        ],
        "flags": [
            {"chunk_id": f.chunk_id, "type": f.flag_type,
             "severity": f.severity, "message": f.message,
             "suggestion": f.suggestion}
            for f in report.flags
        ],
    }


# ── Quick Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sentiment_path = os.path.join(ROOT, "sentiment_output.json")

    if not os.path.exists(sentiment_path):
        print("❌ Run sentiment_analyzer.py first to generate sentiment_output.json")
        exit(1)

    with open(sentiment_path) as f:
        timeline = json.load(f)

    report = detect_pacing(
        timeline_dict = timeline,
        film_title    = "RRR",
        genre         = "action/epic",
        run_hero_map  = True,
        verbose       = True
    )

    out_path = os.path.join(ROOT, "pacing_output.json")
    with open(out_path, "w") as f:
        json.dump(report_to_dict(report), f, indent=2)
    print(f"\n💾 Saved → {out_path}")
