"""
run_pipeline.py
----------------
Runs the full MS-1 → MS-3 → MS-4 pipeline on a given SRT file.
Usage:
  python3 run_pipeline.py                      # first half of RRR (chunks 1-18)
  python3 run_pipeline.py --chunks 36          # full film
  python3 run_pipeline.py --chunks 10 --fast   # fast keyword mode (no API)
"""

import os, sys, json, argparse, time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from srt_parser        import parse_srt, assign_scene_chunks, get_dialogue_only, get_scene_chunk_texts
from services.sentiment_analyzer import analyze_sentiment, timeline_to_dict
from services.pacing_detector    import detect_pacing, report_to_dict

SRT_FILE   = os.path.join(ROOT, "RRR 2022 JPN UHD en full.srt")
FILM_TITLE = "RRR"
GENRE      = "action/epic"

def run(max_chunks: int = 18, use_llm: bool = True, run_micro: bool = True,
        user_climax_chunk: int = None):
    t0 = time.time()

    # ── MS-1: Parse SRT ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  MS-1 · Scene Extractor")
    print(f"{'='*60}")
    subs     = parse_srt(SRT_FILE)
    subs     = assign_scene_chunks(subs, chunk_minutes=5)
    dialogue = get_dialogue_only(subs)
    all_chunks = get_scene_chunk_texts(dialogue)

    # Filter to first N chunks
    chunks  = {k: v for k, v in all_chunks.items() if k <= max_chunks}
    sub_map = {}
    for s in dialogue:
        if s.scene_chunk <= max_chunks:
            sub_map.setdefault(s.scene_chunk, []).append(s)

    print(f"  Total dialogue lines : {len(dialogue)}")
    print(f"  Chunks selected      : {len(chunks)} of {max(all_chunks)} "
          f"(first {max_chunks * 5} min)")

    # ── MS-3: Sentiment Analysis ──────────────────────────────────────────────
    timeline = analyze_sentiment(
        chunk_texts  = chunks,
        subtitle_map = sub_map,
        film_title   = FILM_TITLE,
        use_llm      = use_llm,
        run_micro    = run_micro,
        verbose      = True
    )
    tl_dict = timeline_to_dict(timeline)

    sent_path = os.path.join(ROOT, "sentiment_output.json")
    with open(sent_path, "w") as f:
        json.dump(tl_dict, f, indent=2)
    print(f"\n  💾 Sentiment saved → sentiment_output.json")

    # ── MS-4: Pacing Detection ────────────────────────────────────────────────
    report  = detect_pacing(
        timeline_dict     = tl_dict,
        film_title        = FILM_TITLE,
        genre             = GENRE,
        run_hero_map      = use_llm,
        user_climax_chunk = user_climax_chunk,
        verbose           = True
    )
    pac_dict = report_to_dict(report)

    pac_path = os.path.join(ROOT, "pacing_output.json")
    with open(pac_path, "w") as f:
        json.dump(pac_dict, f, indent=2)
    print(f"  💾 Pacing saved  → pacing_output.json")

    # ── Final Summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  🎬 PIPELINE COMPLETE — {FILM_TITLE} (first {max_chunks * 5} min)")
    print(f"{'='*60}")
    print(f"  Chunks analysed      : {len(chunks)}")
    print(f"  Overall sentiment    : {tl_dict['overall_score']:+.3f}")
    print(f"  Peak tension chunk   : {tl_dict['peak_tension_chunk']}")
    print(f"  Peak joy chunk       : {tl_dict['peak_joy_chunk']}")
    print(f"  Pacing score         : {report.pacing_score:.1f} / 10")
    print(f"  Tension debt peak    : {report.tension_debt_peak:.2f}")
    print(f"  Momentum spikes ↗↘   : "
          f"{sum(1 for m in report.momentum_timeline if m.is_spike and m.delta > 0)} rising, "
          f"{sum(1 for m in report.momentum_timeline if m.is_spike and m.delta < 0)} falling")
    print(f"  Flat zones           : {report.flat_zones or 'None'}")
    print(f"  Act boundaries       : {report.act_boundaries or 'None detected'}")
    print(f"  Flags                : "
          f"{sum(1 for f in report.flags if f.severity=='critical')} critical, "
          f"{sum(1 for f in report.flags if f.severity=='warning')} warnings")
    print(f"  Time elapsed         : {elapsed:.1f}s")

    print(f"\n  📊 EMOTIONAL ARC (chunk → score):")
    for c in tl_dict["chunks"]:
        bar  = _bar(c["macro_score"])
        traj = c["trajectory"]["shape"] if c.get("trajectory") else "—"
        print(f"    Chunk {c['chunk_id']:>2}: {bar} {c['macro_score']:+.2f}  "
              f"[{c['macro_label']:<12}]  {traj}")

    if report.hero_journey_map:
        print(f"\n  🗺️  HERO'S JOURNEY MAP:")
        for stage, cid in report.hero_journey_map.items():
            print(f"    {stage:<28} → Chunk {cid}")

    print(f"\n  🚩 ALL FLAGS:")
    for flag in report.flags:
        icon = "🔴" if flag.severity == "critical" else "⚠️ " if flag.severity == "warning" else "ℹ️ "
        print(f"    {icon} {flag.message[:85]}")
        print(f"       💡 {flag.suggestion[:80]}")


def _bar(score: float, w: int = 20) -> str:
    c   = w // 2
    pos = int((score + 1.0) / 2.0 * w)
    bar = ["-"] * w
    bar[c] = "|"
    if 0 <= pos < w:
        bar[pos] = "█"
    return "[" + "".join(bar) + "]"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=18,
                    help="Max chunk number to analyze (default 18 = first 90 min)")
    ap.add_argument("--climax", type=int, default=None,
                    help="Chunk number of the intended climax scene (optional). "
                         "If not set, climax validation is skipped.")
    ap.add_argument("--fast",   action="store_true",
                    help="Use keyword mode instead of LLM (no API needed)")
    ap.add_argument("--no-micro", action="store_true",
                    help="Skip line-level micro analysis (faster)")
    args = ap.parse_args()

    run(
        max_chunks        = args.chunks,
        use_llm           = not args.fast,
        run_micro         = not args.no_micro,
        user_climax_chunk = args.climax
    )
