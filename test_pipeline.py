import run_pipeline
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
temp_path = os.path.join(ROOT, "app_temp.txt")

with open(temp_path, "w", encoding="utf-8") as f:
    f.write("Old Man:\nYou never cheated those women.")

run_pipeline.SRT_FILE = temp_path
run_pipeline.run(max_chunks=18, run_critique=False)
