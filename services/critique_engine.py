"""
services/critique_engine.py
----------------------------
MS-5: Screenplay Critique Engine

Takes outputs from MS-3 (sentiment) and MS-4 (pacing) to produce
a professional screenplay critique report with:
  - Character arc summaries (if character data available)
  - Dialogue critique (flagged lines analysis)
  - Plot/structural issue detection
  - Pacing synthesis
  - Overall screenplay score (1-10) with Top 3 fixes
  - LLM-generated verdict

Output: critique_output.json
"""

import os
import json
import time
from dataclasses import dataclass, field
from typing import Optional
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY_1", ""))
MODEL = "llama-3.3-70b-versatile"
MAX_RETRIES = 3
RETRY_DELAY = 2


# ── Data Models ──────────────────────────────────────────────────────────────
@dataclass
class CharacterArc:
    name: str
    line_count: int
    first_chunk: int
    last_chunk: int
    emotional_journey: str
    arc_type: str          # Transformational | Flat | Tragic | Absent
    screen_presence: float

@dataclass
class DialogueCritique:
    chunk_id: int
    line_text: str
    issue_type: str        # clunky | on_the_nose | powerful | repetitive
    severity: str          # critical | warning | praise
    suggestion: str

@dataclass
class PlotIssue:
    issue_type: str
    severity: str
    description: str
    affected_chunks: list
    suggestion: str

@dataclass
class CritiqueReport:
    screenplay_score: float
    score_breakdown: dict
    top_3_fixes: list
    overall_verdict: str
    character_arcs: list
    dialogue_critiques: list
    plot_issues: list
    hero_journey_map: dict
    pacing_flags: list
    film_title: str
    genre: str
    chunks_analyzed: int


# ── Step 2: Input Loading ────────────────────────────────────────────────────
def load_inputs(sentiment_path: str, pacing_path: str, character_path: str = None):
    with open(sentiment_path) as f:
        sentiment = json.load(f)
    with open(pacing_path) as f:
        pacing = json.load(f)
    character = None
    if character_path and os.path.exists(character_path):
        with open(character_path) as f:
            character = json.load(f)
    return sentiment, pacing, character


# ── Step 3: Character Arc Analysis ───────────────────────────────────────────
def analyze_character_arcs(
    sentiment: dict, character: dict = None, film_title: str = "Film"
) -> list[CharacterArc]:
    chunks = sentiment["chunks"]
    total_chunks = len(chunks)
    chunk_labels = {c["chunk_id"]: c["macro_label"] for c in chunks}
    arcs = []

    if not character:
        return arcs

    for name, info in list(character.items())[:5]:
        fc = info.get("first_chunk", 1)
        lc = info.get("last_chunk", total_chunks)
        lc_count = info.get("line_count", 0)
        active = info.get("chunks_active", 1)
        presence = round(active / max(total_chunks, 1), 2)

        start_label = chunk_labels.get(fc, "Unknown")
        end_label = chunk_labels.get(lc, "Unknown")

        if start_label == end_label:
            arc_type = "Flat"
        elif end_label in ("Sorrowful", "Wrathful", "Fearful"):
            arc_type = "Tragic"
        else:
            arc_type = "Transformational"

        if presence < 0.15:
            arc_type = "Absent"

        journey = f"Enters in a {start_label.lower()} context (chunk {fc}), exits in a {end_label.lower()} context (chunk {lc})"

        arcs.append(CharacterArc(
            name=name, line_count=lc_count, first_chunk=fc, last_chunk=lc,
            emotional_journey=journey, arc_type=arc_type,
            screen_presence=presence
        ))

    # LLM enrichment for top 3 characters
    if arcs and character:
        arcs = _enrich_arcs_llm(arcs[:3], character, film_title) + arcs[3:]

    return arcs


def _enrich_arcs_llm(arcs: list[CharacterArc], character: dict, film_title: str) -> list[CharacterArc]:
    char_info = []
    for arc in arcs:
        info = character.get(arc.name, {})
        samples = info.get("sample_lines", [])[:4]
        char_info.append(
            f"Character: {arc.name} ({arc.line_count} lines, chunks {arc.first_chunk}-{arc.last_chunk})\n"
            f"  Sample dialogue: {'; '.join(samples)}"
        )

    prompt = f"""You are a screenplay analyst. For each character in "{film_title}", write a one-sentence character arc summary describing how they change from beginning to end.

CHARACTERS:
{chr(10).join(char_info)}

Return ONLY valid JSON array, no markdown:
[
  {{"name": "CHARACTER_NAME", "arc_summary": "one sentence arc description"}},
  ...
]"""

    try:
        resp = _CLIENT.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
            max_tokens=400,
        )
        raw = json.loads(resp.choices[0].message.content)
        if isinstance(raw, dict):
            raw = raw.get("characters") or raw.get("arcs") or list(raw.values())[0]
        arc_map = {r["name"].upper(): r["arc_summary"] for r in raw if isinstance(r, dict)}
        for arc in arcs:
            if arc.name.upper() in arc_map:
                arc.emotional_journey = arc_map[arc.name.upper()]
    except Exception as e:
        print(f"    [MS-5] Character arc LLM enrichment failed: {e}")

    return arcs


# ── Step 4: Dialogue Critique ────────────────────────────────────────────────
def critique_dialogue(sentiment: dict) -> list[DialogueCritique]:
    critiques = []
    chunks = sentiment["chunks"]
    prev_emotions = []

    for c in chunks:
        cid = c["chunk_id"]
        score = c["macro_score"]
        dom = c["dominant_emotion"]
        wc = c.get("word_count", 0)
        flagged = c.get("flagged_lines", [])

        # Check for repetitive emotion
        prev_emotions.append(dom)
        if len(prev_emotions) >= 4 and len(set(prev_emotions[-4:])) == 1:
            critiques.append(DialogueCritique(
                chunk_id=cid, line_text=f"[Chunks {cid-3}–{cid}]",
                issue_type="repetitive", severity="warning",
                suggestion=f"4 consecutive chunks dominated by '{dom}'. Vary the emotional register — insert a contrasting beat."
            ))

        # Over-written scene
        if wc > 400:
            critiques.append(DialogueCritique(
                chunk_id=cid, line_text=f"[{wc} words in chunk {cid}]",
                issue_type="clunky", severity="warning",
                suggestion=f"This scene has {wc} words of dialogue — consider trimming exposition. Show, don't tell."
            ))

        # Classify flagged lines
        for line in flagged:
            if score < -0.3:
                critiques.append(DialogueCritique(
                    chunk_id=cid, line_text=line,
                    issue_type="powerful", severity="praise",
                    suggestion="Strong dramatic line — emotionally resonant in context."
                ))
            elif score > 0.3 and c.get("trajectory", {}) and c["trajectory"].get("shape") == "Flat":
                critiques.append(DialogueCritique(
                    chunk_id=cid, line_text=line,
                    issue_type="on_the_nose", severity="warning",
                    suggestion="High-intensity line in a flat trajectory — may feel forced. Consider subtlety."
                ))
            else:
                critiques.append(DialogueCritique(
                    chunk_id=cid, line_text=line,
                    issue_type="powerful", severity="praise",
                    suggestion="Dramatically significant line."
                ))

    return critiques


# ── Step 5: Plot / Structural Issue Detection ────────────────────────────────
def detect_plot_issues(sentiment: dict, pacing: dict, character: dict = None) -> list[PlotIssue]:
    issues = []
    chunks = sentiment["chunks"]
    n = len(chunks)
    scores = [c["macro_score"] for c in chunks]

    # 1. Tonal whiplash — momentum spikes > 1.0
    for m in pacing.get("momentum_timeline", []):
        if abs(m["delta"]) > 1.0:
            issues.append(PlotIssue(
                issue_type="tonal_whiplash", severity="warning",
                description=f"Extreme tonal shift (Δ{m['delta']:+.2f}) at chunk {m['chunk_id']} — audience may feel jarred.",
                affected_chunks=[m["chunk_id"]],
                suggestion="Add a transitional beat before this shift to prepare the audience emotionally."
            ))

    # 2. Unresolved tension — debt doesn't discharge by end
    debt_curve = pacing.get("tension_debt_curve", [])
    if debt_curve and debt_curve[-1] > 0.3:
        issues.append(PlotIssue(
            issue_type="unresolved_tension", severity="warning",
            description=f"Tension debt is {debt_curve[-1]:.2f} at film end — audience may feel emotionally unsatisfied.",
            affected_chunks=[chunks[-1]["chunk_id"]],
            suggestion="Add a darker or more intense scene near the end to discharge accumulated tension debt."
        ))

    # 3. Missing Act 2 conflict
    if n >= 6:
        mid_start = n // 3
        mid_end = 2 * n // 3
        mid_scores = scores[mid_start:mid_end]
        if mid_scores and all(s > -0.3 for s in mid_scores):
            issues.append(PlotIssue(
                issue_type="missing_conflict", severity="critical",
                description="The middle third of the film lacks any scene with strong negative tension (all scores > -0.3).",
                affected_chunks=list(range(chunks[mid_start]["chunk_id"], chunks[min(mid_end, n-1)]["chunk_id"] + 1)),
                suggestion="Act 2 needs an 'All Is Lost' moment — a major setback, betrayal, or revelation."
            ))

    # 4. Anticlimactic ending
    if n >= 4:
        last_two = scores[-2:]
        if all(abs(s) < 0.3 for s in last_two):
            issues.append(PlotIssue(
                issue_type="anticlimactic_ending", severity="warning",
                description="The last two chunks are emotionally flat — the film may end with a whimper.",
                affected_chunks=[chunks[-2]["chunk_id"], chunks[-1]["chunk_id"]],
                suggestion="Strengthen the final scenes with a clear emotional payoff — triumph, sacrifice, or catharsis."
            ))

    # 5. Front-loaded drama
    peak_tension = sentiment.get("peak_tension_chunk")
    if peak_tension and n >= 6:
        pos = next((i for i, c in enumerate(chunks) if c["chunk_id"] == peak_tension), 0)
        if pos < n * 0.2:
            issues.append(PlotIssue(
                issue_type="front_loaded_drama", severity="warning",
                description=f"Peak tension is at chunk {peak_tension} ({pos/n:.0%} into the film) — too early.",
                affected_chunks=[peak_tension],
                suggestion="Build tension more gradually. Save the darkest moment for 65-80% into the story."
            ))

    # 6. Repetitive emotions (4+ consecutive same dominant)
    for i in range(3, n):
        window = [chunks[j]["dominant_emotion"] for j in range(i-3, i+1)]
        if len(set(window)) == 1:
            issues.append(PlotIssue(
                issue_type="repetitive_emotion", severity="warning",
                description=f"Chunks {chunks[i-3]['chunk_id']}–{chunks[i]['chunk_id']}: 4 consecutive scenes dominated by '{window[0]}'.",
                affected_chunks=[chunks[j]["chunk_id"] for j in range(i-3, i+1)],
                suggestion=f"Break the emotional monotony — insert a scene with a contrasting emotion."
            ))
            break  # only flag once

    # 7. Character disappears (if character data available)
    if character:
        for name, info in character.items():
            fc = info.get("first_chunk", 1)
            lc = info.get("last_chunk", n)
            active = info.get("chunks_active", 0)
            span = lc - fc + 1
            if span > 5 and active < span * 0.4 and info.get("line_count", 0) > 10:
                issues.append(PlotIssue(
                    issue_type="character_disappears", severity="warning",
                    description=f"{name} appears in chunks {fc}-{lc} but is only active in {active}/{span} chunks.",
                    affected_chunks=[fc, lc],
                    suggestion=f"Either give {name} scenes during their gap or explain their absence in the story."
                ))

    return issues


# ── Step 6: Pacing Synthesis ─────────────────────────────────────────────────
def synthesize_pacing(pacing: dict) -> list[dict]:
    flags = pacing.get("flags", [])
    synthesis = []
    for f in flags:
        synthesis.append({
            "chunk_id": f["chunk_id"],
            "type": f["type"],
            "severity": f["severity"],
            "critique": f["message"],
            "fix": f["suggestion"]
        })
    return synthesis


# ── Step 7: Score Computation ────────────────────────────────────────────────
def compute_screenplay_score(
    sentiment: dict, pacing: dict,
    dialogue_critiques: list[DialogueCritique],
    plot_issues: list[PlotIssue]
) -> tuple[float, dict]:
    # 1. Pacing (30%)
    pacing_score = pacing.get("pacing_score", 5.0)

    # 2. Emotional Range (25%)
    scores = [c["macro_score"] for c in sentiment["chunks"]]
    score_range = max(scores) - min(scores) if scores else 0
    if score_range > 1.2:
        emotional_score = 9.0
    elif score_range > 0.8:
        emotional_score = 7.5
    elif score_range > 0.5:
        emotional_score = 6.0
    else:
        emotional_score = 4.0

    # 3. Dialogue (20%)
    dialogue_score = 8.0
    for dc in dialogue_critiques:
        if dc.severity == "warning" and dc.issue_type == "clunky":
            dialogue_score -= 0.5
        elif dc.severity == "warning" and dc.issue_type == "repetitive":
            dialogue_score -= 0.3
        elif dc.severity == "praise":
            dialogue_score += 0.2
    dialogue_score = max(1.0, min(10.0, dialogue_score))

    # 4. Structure (25%)
    structure_score = 8.0
    for pi in plot_issues:
        if pi.severity == "critical":
            structure_score -= 1.0
        elif pi.severity == "warning":
            structure_score -= 0.4
    hero_map = pacing.get("hero_journey_map", {})
    filled = sum(1 for v in hero_map.values() if v is not None)
    if filled >= 7:
        structure_score += 0.5
    structure_score = max(1.0, min(10.0, structure_score))

    # Weighted final
    final = (0.30 * pacing_score + 0.25 * emotional_score +
             0.20 * dialogue_score + 0.25 * structure_score)
    final = round(max(1.0, min(10.0, final)), 1)

    breakdown = {
        "pacing": round(pacing_score, 1),
        "emotional_range": round(emotional_score, 1),
        "dialogue": round(dialogue_score, 1),
        "structure": round(structure_score, 1)
    }
    return final, breakdown


# ── Step 8: Verdict + Top 3 Fixes (LLM) ─────────────────────────────────────
def generate_verdict(
    score: float, breakdown: dict,
    plot_issues: list[PlotIssue],
    dialogue_critiques: list[DialogueCritique],
    pacing: dict, film_title: str, genre: str
) -> tuple[str, list[str]]:

    issues_text = "\n".join(
        f"- [{pi.severity.upper()}] {pi.description}" for pi in plot_issues[:6]
    ) or "No major structural issues detected."

    dialogue_text = "\n".join(
        f"- [{dc.severity}] {dc.issue_type}: {dc.line_text[:60]}" for dc in dialogue_critiques
        if dc.severity != "praise"
    )[:500] or "Dialogue is generally strong."

    prompt = f"""You are a professional screenplay analyst writing a studio coverage report for "{film_title}" ({genre}).

ANALYSIS DATA:
- Screenplay Score: {score}/10
- Breakdown: Pacing {breakdown['pacing']}/10, Emotional Range {breakdown['emotional_range']}/10, Dialogue {breakdown['dialogue']}/10, Structure {breakdown['structure']}/10
- Pacing Score: {pacing.get('pacing_score', 'N/A')}/10
- Avg Momentum: {pacing.get('avg_momentum', 'N/A')}
- Tension Debt Peak: {pacing.get('tension_debt_peak', 'N/A')}

STRUCTURAL ISSUES:
{issues_text}

DIALOGUE ISSUES:
{dialogue_text}

Generate:
1. A 2-3 sentence professional verdict (like a studio reader's assessment)
2. Exactly 3 specific, actionable fixes ranked by impact

Return ONLY valid JSON:
{{
  "verdict": "2-3 sentence assessment",
  "top_3_fixes": ["fix 1", "fix 2", "fix 3"]
}}"""

    for attempt in range(MAX_RETRIES):
        try:
            resp = _CLIENT.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"},
                max_tokens=500,
            )
            result = json.loads(resp.choices[0].message.content)
            return result.get("verdict", ""), result.get("top_3_fixes", [])
        except Exception as e:
            print(f"    [MS-5] Verdict LLM attempt {attempt+1} failed: {e}")
            time.sleep(RETRY_DELAY)

    return "Analysis complete. See detailed scores below.", [
        "Address the lowest-scoring category in the breakdown.",
        "Review flagged dialogue lines for naturalness.",
        "Ensure emotional variety across the middle act."
    ]


# ── Step 9: Serialization ────────────────────────────────────────────────────
def report_to_dict(report: CritiqueReport) -> dict:
    return {
        "screenplay_score": report.screenplay_score,
        "score_breakdown": report.score_breakdown,
        "overall_verdict": report.overall_verdict,
        "top_3_fixes": report.top_3_fixes,
        "character_arcs": [
            {"name": a.name, "line_count": a.line_count,
             "first_chunk": a.first_chunk, "last_chunk": a.last_chunk,
             "emotional_journey": a.emotional_journey,
             "arc_type": a.arc_type, "screen_presence": a.screen_presence}
            for a in report.character_arcs
        ],
        "dialogue_critiques": [
            {"chunk_id": d.chunk_id, "line_text": d.line_text,
             "issue_type": d.issue_type, "severity": d.severity,
             "suggestion": d.suggestion}
            for d in report.dialogue_critiques
        ],
        "plot_issues": [
            {"issue_type": p.issue_type, "severity": p.severity,
             "description": p.description, "affected_chunks": p.affected_chunks,
             "suggestion": p.suggestion}
            for p in report.plot_issues
        ],
        "hero_journey_map": report.hero_journey_map,
        "pacing_flags": report.pacing_flags,
        "metadata": {
            "film_title": report.film_title,
            "genre": report.genre,
            "chunks_analyzed": report.chunks_analyzed
        }
    }


# ── Main Entry Point ─────────────────────────────────────────────────────────
def generate_critique(
    sentiment_path: str,
    pacing_path: str,
    character_path: str = None,
    film_title: str = "Film",
    genre: str = "drama",
    verbose: bool = True
) -> CritiqueReport:
    if verbose:
        print(f"\n{'='*60}")
        print(f"  📝 MS-5: Screenplay Critique Engine")
        print(f"  Film: {film_title} | Genre: {genre}")
        print(f"{'='*60}\n")

    # Load inputs
    sentiment, pacing, character = load_inputs(sentiment_path, pacing_path, character_path)
    chunks = sentiment["chunks"]

    if verbose:
        print(f"  Loaded: {len(chunks)} chunks from sentiment, "
              f"pacing score {pacing.get('pacing_score', '?')}/10")
        if character:
            print(f"  Character data: {len(character)} characters loaded")
        else:
            print(f"  Character data: not available (MS-2 not run)")

    # Step 3: Character arcs
    if verbose: print(f"\n  👥 Analyzing character arcs...", end=" ", flush=True)
    arcs = analyze_character_arcs(sentiment, character, film_title)
    if verbose: print(f"✅ {len(arcs)} characters analyzed")

    # Step 4: Dialogue critique
    if verbose: print(f"  💬 Critiquing dialogue...", end=" ", flush=True)
    dialogue_critiques = critique_dialogue(sentiment)
    praise_count = sum(1 for d in dialogue_critiques if d.severity == "praise")
    warn_count = sum(1 for d in dialogue_critiques if d.severity == "warning")
    if verbose: print(f"✅ {praise_count} powerful, {warn_count} issues")

    # Step 5: Plot issues
    if verbose: print(f"  🔍 Detecting structural issues...", end=" ", flush=True)
    plot_issues = detect_plot_issues(sentiment, pacing, character)
    if verbose:
        crit = sum(1 for p in plot_issues if p.severity == "critical")
        warn = sum(1 for p in plot_issues if p.severity == "warning")
        print(f"✅ {crit} critical, {warn} warnings")

    # Step 6: Pacing synthesis
    pacing_flags = synthesize_pacing(pacing)

    # Step 7: Score
    if verbose: print(f"  📊 Computing screenplay score...", end=" ", flush=True)
    score, breakdown = compute_screenplay_score(sentiment, pacing, dialogue_critiques, plot_issues)
    if verbose: print(f"✅ {score}/10")

    # Step 8: Verdict (LLM)
    if verbose: print(f"  🤖 Generating verdict (Groq)...", end=" ", flush=True)
    verdict, top_3 = generate_verdict(score, breakdown, plot_issues, dialogue_critiques,
                                       pacing, film_title, genre)
    if verbose: print(f"✅")

    report = CritiqueReport(
        screenplay_score=score,
        score_breakdown=breakdown,
        top_3_fixes=top_3,
        overall_verdict=verdict,
        character_arcs=arcs,
        dialogue_critiques=dialogue_critiques,
        plot_issues=plot_issues,
        hero_journey_map=pacing.get("hero_journey_map", {}),
        pacing_flags=pacing_flags,
        film_title=film_title,
        genre=genre,
        chunks_analyzed=len(chunks)
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"  ✅ Critique complete!")
        print(f"     Screenplay Score : {score}/10")
        print(f"     Pacing           : {breakdown['pacing']}/10")
        print(f"     Emotional Range  : {breakdown['emotional_range']}/10")
        print(f"     Dialogue         : {breakdown['dialogue']}/10")
        print(f"     Structure        : {breakdown['structure']}/10")
        print(f"\n  📜 VERDICT: {verdict[:120]}...")
        print(f"\n  🔧 TOP 3 FIXES:")
        for i, fix in enumerate(top_3, 1):
            print(f"     {i}. {fix[:90]}")
        print(f"\n  🔍 PLOT ISSUES:")
        for pi in plot_issues[:5]:
            icon = "🔴" if pi.severity == "critical" else "⚠️ "
            print(f"     {icon} {pi.description[:90]}")
        print(f"{'='*60}")

    return report


# ── Quick Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sent_path = os.path.join(ROOT, "sentiment_output.json")
    pac_path = os.path.join(ROOT, "pacing_output.json")

    if not os.path.exists(sent_path):
        print("❌ Run sentiment_analyzer.py first to generate sentiment_output.json")
        exit(1)
    if not os.path.exists(pac_path):
        print("❌ Run pacing_detector.py first to generate pacing_output.json")
        exit(1)

    report = generate_critique(
        sentiment_path=sent_path,
        pacing_path=pac_path,
        character_path=None,
        film_title="RRR",
        genre="action/epic",
        verbose=True
    )

    out_path = os.path.join(ROOT, "critique_output.json")
    with open(out_path, "w") as f:
        json.dump(report_to_dict(report), f, indent=2)
    print(f"\n💾 Saved → {out_path}")
