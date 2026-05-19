"""
api/server.py
--------------
Phase 4: FastAPI REST API wrapping all microservices.

Endpoints:
  POST /api/extract-scenes       MS-1  Scene Extractor
  POST /api/extract-characters   MS-2  Character Extractor
  POST /api/analyze-sentiment    MS-3  Sentiment Analyzer
  POST /api/detect-pacing        MS-4  Pacing Detector
  POST /api/critique             MS-5  Critique Engine
  POST /api/style-transfer       MS-7  Style Transfer
  POST /api/run-pipeline         Full pipeline (MS-1 → MS-5)
  GET  /health                   Health check

Run:
  uvicorn api.server:app --reload
"""

import os
import sys
import json
import tempfile
import time

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# ── Ensure project root is on sys.path ────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from srt_parser import parse_srt, assign_scene_chunks, get_dialogue_only, get_scene_chunk_texts
from services.character_extractor import extract_characters, get_character_summary
from services.sentiment_analyzer import analyze_sentiment, timeline_to_dict
from services.pacing_detector import detect_pacing, report_to_dict as pacing_to_dict
from services.critique_engine import generate_critique, report_to_dict as critique_to_dict
from services.style_transfer import rewrite_scene
import style_transfer_prompts as stp

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Screenplay Doctor API",
    description=(
        "REST API for the AI Screenplay Doctor — an NLP + LLM powered "
        "screenplay analysis platform. Wraps 7 microservices as endpoints."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic Models — Request / Response
# ══════════════════════════════════════════════════════════════════════════════

# ── MS-1: Scene Extraction ────────────────────────────────────────────────────
class SceneExtractionResponse(BaseModel):
    total_subtitles: int
    spoken_lines: int
    song_lines: int
    total_chunks: int
    chunks: dict  # {chunk_id: dialogue_text}


# ── MS-2: Character Extraction ────────────────────────────────────────────────
class CharacterExtractionResponse(BaseModel):
    total_characters: int
    characters: dict  # {name: {line_count, first_chunk, last_chunk, ...}}


# ── MS-3: Sentiment Analysis ─────────────────────────────────────────────────
class SentimentRequest(BaseModel):
    chunks: dict = Field(..., description="Dict of {chunk_id: dialogue_text}")
    film_title: str = "Film"
    use_llm: bool = True
    run_micro: bool = True


class SentimentResponse(BaseModel):
    overall_score: float
    peak_tension_chunk: Optional[int] = None
    peak_joy_chunk: Optional[int] = None
    flat_zones: list = []
    act_boundaries: list = []
    chunks: list = []


# ── MS-4: Pacing Detection ───────────────────────────────────────────────────
class PacingRequest(BaseModel):
    sentiment_data: dict = Field(..., description="Full sentiment timeline dict from MS-3")
    film_title: str = "Film"
    genre: str = "drama"
    run_hero_map: bool = True
    user_climax_chunk: Optional[int] = None


class PacingResponse(BaseModel):
    pacing_score: float
    tension_debt_peak: float
    avg_momentum: float
    curve_deviation: float
    flat_zones: list = []
    climax_chunk: Optional[int] = None
    climax_well_placed: bool = False
    flags: list = []
    hero_journey_map: dict = {}


# ── MS-5: Critique ────────────────────────────────────────────────────────────
class CritiqueRequest(BaseModel):
    sentiment_data: dict = Field(..., description="Full sentiment timeline dict from MS-3")
    pacing_data: dict = Field(..., description="Full pacing report dict from MS-4")
    character_data: Optional[dict] = None
    film_title: str = "Film"
    genre: str = "drama"


class CritiqueResponse(BaseModel):
    screenplay_score: float
    score_breakdown: dict = {}
    overall_verdict: str = ""
    top_3_fixes: list = []
    character_arcs: list = []
    dialogue_critiques: list = []
    plot_issues: list = []
    hero_journey_map: dict = {}
    pacing_flags: list = []


# ── MS-7: Style Transfer ─────────────────────────────────────────────────────
class StyleTransferRequest(BaseModel):
    chunk_text: str = Field(..., description="Dialogue text of the scene to rewrite")
    director: str = Field(
        ..., description="Director style: 'Anurag Kashyap', 'Imtiaz Ali', or 'Mani Ratnam'"
    )


class StyleTransferResponse(BaseModel):
    director: str
    rewritten_scene: str


# ── Full Pipeline ─────────────────────────────────────────────────────────────
class PipelineResponse(BaseModel):
    sentiment: dict
    pacing: dict
    critique: dict
    metadata: dict


# ══════════════════════════════════════════════════════════════════════════════
# Helper: save UploadFile to a temp path and return it
# ══════════════════════════════════════════════════════════════════════════════

async def _save_upload(file: UploadFile) -> str:
    """Save an uploaded file to a temporary path and return the path."""
    suffix = os.path.splitext(file.filename or ".srt")[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=ROOT)
    content = await file.read()
    tmp.write(content)
    tmp.close()
    return tmp.name


# ══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "AI Screenplay Doctor API", "version": "1.0.0"}


# ── MS-1: Scene Extraction ────────────────────────────────────────────────────
@app.post("/api/extract-scenes", response_model=SceneExtractionResponse)
async def extract_scenes(
    file: UploadFile = File(..., description="SRT subtitle file"),
    chunk_minutes: float = 5.0,
):
    """
    **MS-1: Scene Extractor**

    Parses an uploaded `.srt` file, separates dialogue from songs,
    and groups lines into N-minute scene chunks.
    """
    tmp_path = await _save_upload(file)
    try:
        subs = parse_srt(tmp_path)
        subs = assign_scene_chunks(subs, chunk_minutes=chunk_minutes)
        dialogue = get_dialogue_only(subs)
        songs = [s for s in subs if s.is_song]
        chunks = get_scene_chunk_texts(dialogue)

        return SceneExtractionResponse(
            total_subtitles=len(subs),
            spoken_lines=len(dialogue),
            song_lines=len(songs),
            total_chunks=max(chunks.keys()) if chunks else 0,
            chunks={str(k): v for k, v in chunks.items()},
        )
    finally:
        os.unlink(tmp_path)


# ── MS-2: Character Extraction ────────────────────────────────────────────────
@app.post("/api/extract-characters", response_model=CharacterExtractionResponse)
async def api_extract_characters(
    file: UploadFile = File(..., description="SRT subtitle file"),
    film_title: str = "Film",
    genre: str = "Drama",
    max_chunks: int = 36,
):
    """
    **MS-2: Character Extractor**

    Uses Gemini LLM to attribute speaker identity to each dialogue line
    from an SRT file (since SRT files contain no speaker labels).
    """
    tmp_path = await _save_upload(file)
    try:
        subs = parse_srt(tmp_path)
        subs = assign_scene_chunks(subs, chunk_minutes=5)
        test_subs = [s for s in subs if not s.is_song and s.scene_chunk <= max_chunks]

        _, characters = extract_characters(
            subtitles=test_subs,
            film_title=film_title,
            genre=genre,
            verbose=False,
        )
        summary = get_character_summary(characters)

        return CharacterExtractionResponse(
            total_characters=len(summary),
            characters=summary,
        )
    finally:
        os.unlink(tmp_path)


# ── MS-3: Sentiment Analysis ─────────────────────────────────────────────────
@app.post("/api/analyze-sentiment", response_model=SentimentResponse)
async def api_analyze_sentiment(request: SentimentRequest):
    """
    **MS-3: Sentiment & Emotion Analyzer (Two-Layer)**

    Accepts pre-extracted scene chunk texts and runs macro (chunk-level)
    and optional micro (line-level) sentiment analysis using Groq LLMs.
    """
    # Convert string keys back to int (JSON keys are always strings)
    chunk_texts = {int(k): v for k, v in request.chunks.items()}

    timeline = analyze_sentiment(
        chunk_texts=chunk_texts,
        subtitle_map=None,
        film_title=request.film_title,
        use_llm=request.use_llm,
        run_micro=request.run_micro,
        verbose=False,
    )
    return timeline_to_dict(timeline)


# ── MS-4: Pacing Detection ───────────────────────────────────────────────────
@app.post("/api/detect-pacing", response_model=PacingResponse)
async def api_detect_pacing(request: PacingRequest):
    """
    **MS-4: Conflict & Pacing Detector**

    Accepts the sentiment timeline from MS-3 and produces pacing analysis
    including ideal curve deviation, tension debt, momentum spikes,
    flat zone detection, and Hero's Journey mapping.
    """
    report = detect_pacing(
        timeline_dict=request.sentiment_data,
        film_title=request.film_title,
        genre=request.genre,
        run_hero_map=request.run_hero_map,
        user_climax_chunk=request.user_climax_chunk,
        verbose=False,
    )
    return pacing_to_dict(report)


# ── MS-5: Critique Engine ────────────────────────────────────────────────────
@app.post("/api/critique", response_model=CritiqueResponse)
async def api_critique(request: CritiqueRequest):
    """
    **MS-5: Screenplay Critique Engine**

    Accepts sentiment + pacing data and produces a professional critique
    including character arcs, dialogue analysis, plot issues, and an
    overall screenplay score with actionable fixes.
    """
    # Save temp files for the critique engine (it reads from disk)
    sent_path = os.path.join(ROOT, "_tmp_sentiment.json")
    pac_path = os.path.join(ROOT, "_tmp_pacing.json")
    char_path = None

    try:
        with open(sent_path, "w") as f:
            json.dump(request.sentiment_data, f)
        with open(pac_path, "w") as f:
            json.dump(request.pacing_data, f)

        if request.character_data:
            char_path = os.path.join(ROOT, "_tmp_character.json")
            with open(char_path, "w") as f:
                json.dump(request.character_data, f)

        report = generate_critique(
            sentiment_path=sent_path,
            pacing_path=pac_path,
            character_path=char_path,
            film_title=request.film_title,
            genre=request.genre,
            verbose=False,
        )
        return critique_to_dict(report)
    finally:
        for p in [sent_path, pac_path, char_path]:
            if p and os.path.exists(p):
                os.unlink(p)


# ── MS-7: Style Transfer ─────────────────────────────────────────────────────
DIRECTOR_PROMPTS = {
    "Anurag Kashyap": stp.PROMPT_1_ANURAG_KASHYAP,
    "Imtiaz Ali":     stp.PROMPT_2_IMTIAZ_ALI,
    "Mani Ratnam":    stp.PROMPT_3_MANI_RATNAM,
}

@app.post("/api/style-transfer", response_model=StyleTransferResponse)
async def api_style_transfer(request: StyleTransferRequest):
    """
    **MS-7: Style Transfer Engine**

    Rewrites a scene's dialogue in the style of a famous Indian director.
    Available directors: Anurag Kashyap, Imtiaz Ali, Mani Ratnam.
    """
    base_prompt = DIRECTOR_PROMPTS.get(request.director)
    if not base_prompt:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown director '{request.director}'. "
                   f"Choose from: {list(DIRECTOR_PROMPTS.keys())}",
        )

    result = rewrite_scene(base_prompt, request.chunk_text)
    return StyleTransferResponse(
        director=request.director,
        rewritten_scene=result,
    )


# ── Full Pipeline ─────────────────────────────────────────────────────────────
@app.post("/api/run-pipeline", response_model=PipelineResponse)
async def run_full_pipeline(
    file: UploadFile = File(..., description="SRT subtitle file"),
    film_title: str = "Film",
    genre: str = "drama",
    max_chunks: int = 18,
    use_llm: bool = True,
    run_micro: bool = True,
    run_hero_map: bool = True,
):
    """
    **Full Pipeline (MS-1 → MS-3 → MS-4 → MS-5)**

    Runs the entire analysis pipeline on an uploaded SRT file in one call.
    Returns combined sentiment, pacing, and critique results.
    """
    tmp_path = await _save_upload(file)
    t0 = time.time()

    try:
        # ── MS-1: Parse SRT ───────────────────────────────────────────────────
        subs = parse_srt(tmp_path)
        subs = assign_scene_chunks(subs, chunk_minutes=5)
        dialogue = get_dialogue_only(subs)
        all_chunks = get_scene_chunk_texts(dialogue)

        chunks = {k: v for k, v in all_chunks.items() if k <= max_chunks}
        sub_map = {}
        for s in dialogue:
            if s.scene_chunk <= max_chunks:
                sub_map.setdefault(s.scene_chunk, []).append(s)

        # ── MS-3: Sentiment Analysis ──────────────────────────────────────────
        timeline = analyze_sentiment(
            chunk_texts=chunks,
            subtitle_map=sub_map,
            film_title=film_title,
            use_llm=use_llm,
            run_micro=run_micro,
            verbose=False,
        )
        sent_dict = timeline_to_dict(timeline)

        # ── MS-4: Pacing Detection ────────────────────────────────────────────
        pacing_report = detect_pacing(
            timeline_dict=sent_dict,
            film_title=film_title,
            genre=genre,
            run_hero_map=run_hero_map,
            verbose=False,
        )
        pac_dict = pacing_to_dict(pacing_report)

        # ── MS-5: Critique Engine ─────────────────────────────────────────────
        sent_path = os.path.join(ROOT, "_tmp_pipe_sentiment.json")
        pac_path = os.path.join(ROOT, "_tmp_pipe_pacing.json")
        with open(sent_path, "w") as f:
            json.dump(sent_dict, f)
        with open(pac_path, "w") as f:
            json.dump(pac_dict, f)

        critique_report = generate_critique(
            sentiment_path=sent_path,
            pacing_path=pac_path,
            character_path=None,
            film_title=film_title,
            genre=genre,
            verbose=False,
        )
        crit_dict = critique_to_dict(critique_report)

        # Cleanup temp files
        for p in [sent_path, pac_path]:
            if os.path.exists(p):
                os.unlink(p)

        elapsed = time.time() - t0

        return PipelineResponse(
            sentiment=sent_dict,
            pacing=pac_dict,
            critique=crit_dict,
            metadata={
                "film_title": film_title,
                "genre": genre,
                "chunks_analyzed": len(chunks),
                "elapsed_seconds": round(elapsed, 1),
            },
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
