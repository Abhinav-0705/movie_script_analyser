"""
services/character_extractor.py
---------------------------------
MS-2: Character Extractor & Tracker

Takes Subtitle objects from MS-1 (srt_parser) and uses Gemini to:
1. Attribute each dialogue line to a speaker (since SRTs have no labels)
2. Build a full character roster with line counts and scene appearances
3. Return structured CharacterProfile objects for downstream use

Strategy:
  - Process dialogue in batches of BATCH_SIZE lines
  - Each batch is sent to Gemini with film context
  - Gemini returns JSON: [{index, speaker, dialogue}]
  - Results are aggregated into character profiles
"""

import json
import time
import os
from dataclasses import dataclass, field
from typing import Optional
import google.generativeai as genai
from dotenv import load_dotenv

# Load local .env for API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ── Config ──────────────────────────────────────────────────────────────────
BATCH_SIZE   = 25       # lines per Gemini call (balance cost vs context)
MODEL_NAME   = "gemini-1.5-flash"
RETRY_DELAY  = 2        # seconds between retries on API failure
MAX_RETRIES  = 3


# ── Data Models ─────────────────────────────────────────────────────────────
@dataclass
class AttributedLine:
    """A single dialogue line with speaker attribution."""
    original_index: int         # subtitle index from MS-1
    chunk_id:       int         # scene chunk number
    start_sec:      float       # timestamp in seconds
    speaker:        str         # character name (from Gemini)
    dialogue:       str         # cleaned dialogue text


@dataclass
class CharacterProfile:
    """Aggregated data for one character across the full film."""
    name:         str
    lines:        list[str]        = field(default_factory=list)
    chunks_seen:  set[int]         = field(default_factory=set)
    line_count:   int              = 0
    first_chunk:  Optional[int]    = None
    last_chunk:   Optional[int]    = None
    arc_summary:  Optional[str]    = None   # filled by MS-5 critique engine

    def update_with_line(self, dialogue: str, chunk_id: int):
        self.lines.append(dialogue)
        self.chunks_seen.add(chunk_id)
        self.line_count += 1
        if self.first_chunk is None or chunk_id < self.first_chunk:
            self.first_chunk = chunk_id
        if self.last_chunk is None or chunk_id > self.last_chunk:
            self.last_chunk = chunk_id


# ── Core Functions ───────────────────────────────────────────────────────────
def _build_attribution_prompt(
    lines: list[dict],
    film_title: str,
    genre: str,
    known_characters: list[str]
) -> str:
    """
    Build a Gemini prompt for speaker attribution on a batch of dialogue lines.
    Returns a prompt string that asks for JSON output.
    """
    char_hint = (
        f"Known characters so far: {', '.join(known_characters)}. "
        "You may introduce new character names if needed."
        if known_characters
        else "Identify character names from context."
    )

    lines_formatted = "\n".join(
        f"[{i}] {item['text']}" for i, item in enumerate(lines)
    )

    return f"""You are a film dialogue analyst. You are reading subtitle lines from the movie "{film_title}" (genre: {genre}).

{char_hint}

Your task: Identify which character is most likely speaking each line below.

Rules:
- Use SHORT character name labels (e.g. "RAM", "BHEEM", "JENNY", "SCOTT")
- If a line is stage direction, narration, or a title card (not dialogue), label it "NARRATOR"
- If a line has multiple speakers (e.g. "Edward. - Yes, sir."), split it as best you can or label the dominant speaker
- Be consistent — use the same name every time for the same character
- Base your decision on: tone, vocabulary, context, who was speaking before/after

Subtitle lines:
{lines_formatted}

Respond ONLY with a valid JSON array, no explanation, no markdown:
[
  {{"index": 0, "speaker": "CHARACTER_NAME", "dialogue": "the dialogue text"}},
  ...
]"""


def _call_gemini(prompt: str) -> Optional[list[dict]]:
    """
    Call Gemini API and parse the JSON response.
    Returns list of {index, speaker, dialogue} or None on failure.
    """
    model = genai.GenerativeModel(MODEL_NAME)

    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,        # low temp for consistent attribution
                    response_mime_type="application/json"
                )
            )
            raw = response.text.strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            return json.loads(raw)

        except json.JSONDecodeError as e:
            print(f"  ⚠️  JSON parse error on attempt {attempt+1}: {e}")
            time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"  ⚠️  Gemini API error on attempt {attempt+1}: {e}")
            time.sleep(RETRY_DELAY)

    return None


def extract_characters(
    subtitles,                        # list[Subtitle] from srt_parser
    film_title: str = "Unknown Film",
    genre: str = "Drama",
    verbose: bool = True
) -> tuple[list[AttributedLine], dict[str, CharacterProfile]]:
    """
    Main MS-2 entry point.

    Args:
        subtitles:   list of Subtitle objects (dialogue only, from srt_parser)
        film_title:  film name for context in the prompt
        genre:       film genre for context
        verbose:     print progress to terminal

    Returns:
        attributed_lines:  list[AttributedLine] — every line with speaker label
        characters:        dict[name → CharacterProfile] — full character roster
    """
    # Filter to dialogue only (exclude songs)
    dialogue_subs = [s for s in subtitles if not s.is_song]

    attributed_lines: list[AttributedLine] = []
    characters: dict[str, CharacterProfile] = {}
    known_characters: list[str] = []

    total_batches = (len(dialogue_subs) + BATCH_SIZE - 1) // BATCH_SIZE

    if verbose:
        print(f"\n🎭 MS-2: Character Extractor")
        print(f"   Film: {film_title} | Genre: {genre}")
        print(f"   Dialogue lines: {len(dialogue_subs)}")
        print(f"   Batches to process: {total_batches} (batch size: {BATCH_SIZE})\n")

    for batch_num in range(total_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx   = min(start_idx + BATCH_SIZE, len(dialogue_subs))
        batch     = dialogue_subs[start_idx:end_idx]

        if verbose:
            print(f"  Processing batch {batch_num+1}/{total_batches} "
                  f"(lines {start_idx}–{end_idx-1})...", end=" ", flush=True)

        # Prepare input for prompt
        batch_input = [{"text": sub.text} for sub in batch]

        # Build and send prompt
        prompt   = _build_attribution_prompt(batch_input, film_title, genre, known_characters)
        results  = _call_gemini(prompt)

        if results is None:
            if verbose:
                print("❌ Failed — labelling as UNKNOWN")
            # Fallback: label everything in this batch as UNKNOWN
            for i, sub in enumerate(batch):
                _process_result(
                    {"index": i, "speaker": "UNKNOWN", "dialogue": sub.text},
                    sub, attributed_lines, characters
                )
            continue

        # Process results
        result_map = {r["index"]: r for r in results if isinstance(r, dict)}

        for i, sub in enumerate(batch):
            result = result_map.get(i, {"index": i, "speaker": "UNKNOWN", "dialogue": sub.text})
            _process_result(result, sub, attributed_lines, characters)

        # Update known characters so next batch is more consistent
        known_characters = list(characters.keys())

        if verbose:
            new_chars = [r.get("speaker", "?") for r in results]
            unique = list(set(new_chars) - {"NARRATOR", "UNKNOWN"})
            print(f"✅  Characters seen: {', '.join(known_characters[:8])}")

        time.sleep(0.3)  # small delay to avoid rate limiting

    # Sort characters by line count
    characters = dict(
        sorted(characters.items(), key=lambda x: x[1].line_count, reverse=True)
    )

    if verbose:
        print(f"\n✅ Extraction complete!")
        print(f"   Total characters found: {len(characters)}")
        print(f"\n   📋 Character Roster:")
        for name, profile in list(characters.items())[:10]:
            print(f"     {name:<20} {profile.line_count:>4} lines  "
                  f"| chunks {profile.first_chunk}→{profile.last_chunk}")

    return attributed_lines, characters


def _process_result(
    result:           dict,
    sub,                          # Subtitle object
    attributed_lines: list,
    characters:       dict
):
    """Helper: add a single attributed line to our data structures."""
    speaker  = str(result.get("speaker", "UNKNOWN")).upper().strip()
    dialogue = result.get("dialogue", sub.text)

    # Add to attributed lines
    attributed_lines.append(AttributedLine(
        original_index = sub.index,
        chunk_id       = sub.scene_chunk,
        start_sec      = sub.start_sec,
        speaker        = speaker,
        dialogue       = dialogue
    ))

    # Update character profile
    if speaker not in characters:
        characters[speaker] = CharacterProfile(name=speaker)
    characters[speaker].update_with_line(dialogue, sub.scene_chunk)


def get_character_summary(characters: dict[str, CharacterProfile]) -> dict:
    """
    Returns a clean summary dict for downstream services and Streamlit display.
    Excludes NARRATOR and UNKNOWN from main roster.
    """
    main_chars = {
        name: {
            "line_count":  p.line_count,
            "first_chunk": p.first_chunk,
            "last_chunk":  p.last_chunk,
            "chunks_active": len(p.chunks_seen),
            "sample_lines": p.lines[:3],        # first 3 lines for display
        }
        for name, p in characters.items()
        if name not in {"NARRATOR", "UNKNOWN", "NARRATOR/UNKNOWN"}
    }
    return main_chars


# ── Quick Test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from srt_parser import parse_srt, assign_scene_chunks, get_dialogue_only

    SRT_FILE = "RRR 2022 JPN UHD en full.srt"

    print("Loading SRT file...")
    subs = parse_srt(SRT_FILE)
    subs = assign_scene_chunks(subs, chunk_minutes=5)

    # For testing, only use first 3 chunks (saves API calls)
    test_subs = [s for s in subs if s.scene_chunk <= 3]
    print(f"Testing with {len(test_subs)} subtitles from first 3 chunks")

    attributed, characters = extract_characters(
        subtitles  = test_subs,
        film_title = "RRR",
        genre      = "Action/Epic",
        verbose    = True
    )

    print("\n--- Sample Attributed Lines ---")
    for line in attributed[:15]:
        print(f"  [{line.chunk_id}] {line.speaker:<15} : {line.dialogue[:60]}")

    summary = get_character_summary(characters)
    print(f"\n--- Character Summary (JSON) ---")
    print(json.dumps(summary, indent=2))
