"""
srt_parser.py
-------------
Robust SRT subtitle parser for the CinéLens / Screenplay Doctor pipeline.
Handles:
  - Windows \r\n and Unix \n line endings
  - Multi-line subtitle blocks
  - HTML tags (<i>, <b>, <u>, <font ...>)
  - Song lyrics (italic) vs spoken dialogue separation
  - Timestamp → seconds conversion for timeline/sentiment plotting
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Subtitle:
    index: int
    start_sec: float        # start time in seconds (for plotting)
    end_sec: float          # end time in seconds
    start_raw: str          # original timestamp string
    text: str               # cleaned text (no HTML tags)
    is_song: bool           # True if the original line was in <i> italics
    scene_chunk: int        # which scene chunk this belongs to (set later)


def _timestamp_to_seconds(ts: str) -> float:
    """Convert SRT timestamp '00:03:24,685' → seconds as float."""
    ts = ts.strip().replace(",", ".")
    h, m, rest = ts.split(":")
    s = float(rest)
    return int(h) * 3600 + int(m) * 60 + s


def _clean_html(text: str) -> str:
    """Strip all HTML tags and normalise whitespace."""
    text = re.sub(r"<[^>]+>", "", text)      # remove <i>, <b>, <font ...> etc.
    text = re.sub(r"\s+", " ", text)          # collapse whitespace
    return text.strip()


def _is_song_line(raw_text: str) -> bool:
    """Heuristic: if the block is wrapped in <i>...</i> it's a song lyric."""
    stripped = raw_text.strip()
    return stripped.startswith("<i>") or stripped.startswith("♪")


def parse_srt(filepath: str) -> list[Subtitle]:
    """
    Parse an SRT file and return a list of Subtitle objects.
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Normalise all line endings to \n
    content = content.replace("\r\r\n", "\n").replace("\r\n", "\n").replace("\r", "\n")

    # Split on blank lines → each block is one subtitle entry
    blocks = re.split(r"\n\s*\n", content.strip())

    subtitles: list[Subtitle] = []

    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue  # malformed block, skip

        # Line 0: index number
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue  # not a valid block

        # Line 1: timestamps
        ts_match = re.match(
            r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})",
            lines[1]
        )
        if not ts_match:
            continue

        start_raw = ts_match.group(1)
        end_raw   = ts_match.group(2)
        start_sec = _timestamp_to_seconds(start_raw)
        end_sec   = _timestamp_to_seconds(end_raw)

        # Lines 2+: the actual subtitle text (may be multi-line)
        raw_text = "\n".join(lines[2:])
        is_song  = _is_song_line(raw_text)
        clean    = _clean_html(raw_text)

        if not clean:
            continue  # skip empty blocks (like "THE STORY" title cards optionally)

        subtitles.append(Subtitle(
            index      = index,
            start_sec  = start_sec,
            end_sec    = end_sec,
            start_raw  = start_raw,
            text       = clean,
            is_song    = is_song,
            scene_chunk= 0,          # assigned below
        ))

    return subtitles


def assign_scene_chunks(subtitles: list[Subtitle], chunk_minutes: float = 5.0) -> list[Subtitle]:
    """
    Group subtitles into scene 'chunks' of N minutes each.
    Each chunk becomes one unit for sentiment analysis.
    chunk_minutes: size of each scene window (default 5 min)
    """
    chunk_sec = chunk_minutes * 60
    for sub in subtitles:
        sub.scene_chunk = int(sub.start_sec // chunk_sec) + 1
    return subtitles


def get_dialogue_only(subtitles: list[Subtitle]) -> list[Subtitle]:
    """Filter out song lyrics — return only spoken dialogue."""
    return [s for s in subtitles if not s.is_song]


def get_scene_chunk_texts(subtitles: list[Subtitle]) -> dict[int, str]:
    """
    Returns a dict: { chunk_number → combined dialogue text }
    Ready to be fed into a sentiment analyzer one chunk at a time.
    """
    chunks: dict[int, list[str]] = {}
    for sub in subtitles:
        chunks.setdefault(sub.scene_chunk, []).append(sub.text)
    return {chunk_id: " ".join(lines) for chunk_id, lines in sorted(chunks.items())}


def extract_character_lines(subtitles: list[Subtitle]) -> dict[str, list[str]]:
    """
    Naively attempt to detect CHARACTER: dialogue format in spoken lines.
    Falls back to returning all lines under 'NARRATOR' if no speaker detected.
    """
    character_map: dict[str, list[str]] = {}
    pattern = re.compile(r"^([A-Z][A-Z\s]{1,20}):\s*(.+)$")

    for sub in get_dialogue_only(subtitles):
        match = pattern.match(sub.text)
        if match:
            char  = match.group(1).strip()
            line  = match.group(2).strip()
            character_map.setdefault(char, []).append(line)
        else:
            character_map.setdefault("NARRATOR/UNKNOWN", []).append(sub.text)

    return character_map


# ─── Quick Test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import os

    srt_path = sys.argv[1] if len(sys.argv) > 1 else "RRR 2022 JPN UHD en full.srt"

    print(f"\n{'='*60}")
    print(f"  Parsing: {os.path.basename(srt_path)}")
    print(f"{'='*60}\n")

    subs = parse_srt(srt_path)
    subs = assign_scene_chunks(subs, chunk_minutes=5)

    dialogue = get_dialogue_only(subs)
    songs    = [s for s in subs if s.is_song]

    print(f"✅ Total subtitle blocks : {len(subs)}")
    print(f"🎤 Spoken dialogue lines : {len(dialogue)}")
    print(f"🎵 Song lyric lines      : {len(songs)}")
    print(f"🎬 Scene chunks (5-min)  : {subs[-1].scene_chunk}")

    print(f"\n--- First 10 Dialogue Lines ---")
    for s in dialogue[:10]:
        print(f"  [{s.start_raw}] (chunk {s.scene_chunk}) {s.text}")

    chunks = get_scene_chunk_texts(dialogue)
    print(f"\n--- Scene Chunk 1 Text Preview (first 300 chars) ---")
    print(f"  {list(chunks.values())[0][:300]}...")

    print(f"\n--- Scene Chunk Summary ---")
    for chunk_id, text in chunks.items():
        word_count = len(text.split())
        print(f"  Chunk {chunk_id:>3}: {word_count:>5} words")
