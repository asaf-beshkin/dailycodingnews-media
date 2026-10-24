"""Cut one narration file into one take per scene and measure word starts in each take.

Usage (run next to voice2.py and the sherpa models):
  python3 cut_takes.py narration.wav timing.js '[[0],[1],[2],[3,4],...]' takes/
The scene list groups caption-line indexes exactly like the scene letters in LINES.
Writes takes/take_N.wav (24 kHz mono) and takes/takes.json for reel_engine.py (VO_DIR=takes).
"""
import json, os, re, subprocess, sys
import voice2 as v
from scipy.io import wavfile
narr, tim, scenes, out = sys.argv[1], sys.argv[2], json.loads(sys.argv[3]), sys.argv[4]
os.makedirs(out, exist_ok=True)
T = json.loads(re.search(r'const T=(\{.*\});if', open(tim).read()).group(1))
L = T['lines']; dur = T['duration']; takes = []
for i, ls in enumerate(scenes):
    a = L[ls[0]]['t']; b = L[ls[-1]]['end']
    s = max(0, a - 0.12) if i == 0 else (L[ls[0] - 1]['end'] + a) / 2
    e = dur if i == len(scenes) - 1 else (b + L[ls[-1] + 1]['t']) / 2
    p = f'{out}/take_{i}.wav'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', narr, '-ss', f'{s:.3f}', '-to', f'{e:.3f}', '-ac', '1', '-ar', '24000',
                    '-af', 'afade=t=in:d=0.02,areverse,afade=t=in:d=0.04,areverse', p], check=True)
    heard, words, starts = v.asr(p)
    takes.append({"lines": ls, "wav": f"take_{i}.wav", "align": {"words": words, "starts": starts}})
    sr, x = wavfile.read(p); print(i, round(len(x) / sr, 2), heard.lower())
json.dump(takes, open(f'{out}/takes.json', 'w'))
