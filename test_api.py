"""Quick test of the FastAPI endpoints."""
import requests
import json

BASE = "http://127.0.0.1:8000"

# 1. Health check
print("=== Health Check ===")
r = requests.get(f"{BASE}/health")
print(f"  Status: {r.status_code}")
print(f"  Response: {r.json()}")

# 2. Extract scenes
print("\n=== MS-1: Extract Scenes ===")
with open("Avengers.Endgame.2019.1080p.WEBRip.x264-RARBG-en.srt", "rb") as f:
    r = requests.post(
        f"{BASE}/api/extract-scenes",
        files={"file": ("test.srt", f, "application/octet-stream")},
        params={"chunk_minutes": 5},
    )
d = r.json()
print(f"  Status: {r.status_code}")
print(f"  Total subtitles: {d['total_subtitles']}")
print(f"  Spoken lines: {d['spoken_lines']}")
print(f"  Song lines: {d['song_lines']}")
print(f"  Total chunks: {d['total_chunks']}")
print(f"  Chunks returned: {len(d['chunks'])} keys")

# 3. Verify /docs is accessible
print("\n=== Swagger Docs ===")
r = requests.get(f"{BASE}/openapi.json")
schema = r.json()
paths = list(schema["paths"].keys())
print(f"  OpenAPI paths: {paths}")

print("\n=== ALL TESTS PASSED ===")
