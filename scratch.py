import sys
sys.path.insert(0, ".")
from srt_parser import parse_plain_text
with open("app_temp.txt", "r") as f:
    text = f.read()
print(repr(text))
subs = parse_plain_text(text)
print(subs)
