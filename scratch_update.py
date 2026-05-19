import json
from services.pacing_detector import detect_pacing, report_to_dict
from services.critique_engine import generate_critique, report_to_dict as critique_to_dict

with open('sentiment_output.json') as f:
    tl_dict = json.load(f)

print("Running MS-4 Pacing Detector...")
report = detect_pacing(
    timeline_dict=tl_dict,
    film_title="Avengers: Endgame",
    genre="action",
    run_hero_map=True,
    verbose=True
)
pac_dict = report_to_dict(report)
with open('pacing_output.json', 'w') as f:
    json.dump(pac_dict, f, indent=2)

print("\nRunning MS-5 Critique Engine...")
critique = generate_critique(
    sentiment_path='sentiment_output.json',
    pacing_path='pacing_output.json',
    character_path='character_output.json' if __import__('os').path.exists('character_output.json') else None,
    film_title="Avengers: Endgame",
    genre="action",
    verbose=True
)
with open('critique_output.json', 'w') as f:
    json.dump(critique_to_dict(critique), f, indent=2)
print("\nDone!")
