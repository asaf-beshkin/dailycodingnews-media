"""Measure word timings in a narration and write timing.js for the animation-studio film.

Usage: python3 make_timing.py narration.wav lines.json out/timing.js
lines.json: {"lines": ["caption text of line 0", ...], "spoken": {"AI": "a i", ...}}
"""
import json, re, sys, difflib, subprocess, os
import numpy as np
from scipy.io import wavfile
import voice2 as v

wav, cfgp, outp = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = json.load(open(cfgp)); LINES = cfg["lines"]; SPOKEN = cfg.get("spoken", {})

# 16 kHz mono copy for the recognizer (handles mp3/wav of any rate)
import tempfile; tmp16 = os.path.join(tempfile.gettempdir(), '_asr16.wav')
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', wav, '-ac', '1', '-ar', '16000', tmp16], check=True)
sr, a = wavfile.read(tmp16); a = a.astype(np.float32) / 32768
dur = len(a) / sr
heard, hw, hs = v.asr(tmp16)

def letters(s): return re.sub(r'[^a-z]', '', s.lower())
# recognizer chars with times
achars, atimes = [], []
for i, (w, st) in enumerate(zip(hw, hs)):
    nxt = hs[i + 1] if i + 1 < len(hs) else dur
    L = letters(w)
    for k, ch in enumerate(L): achars.append(ch); atimes.append(st + (nxt - st) * k / max(1, len(L)))
# script words
sw = []
for li, line in enumerate(LINES):
    for w in line.split(): sw.append((li, w, letters(SPOKEN.get(w.strip('.,?!:;'), w))))
schars, owner = [], []
for k, (_, _, L) in enumerate(sw):
    for ch in L: schars.append(ch); owner.append(k)
m = difflib.SequenceMatcher(None, ''.join(schars), ''.join(achars), autojunk=False)
s2a = [None] * len(schars)
for b in m.get_matching_blocks():
    for d in range(b.size): s2a[b.a + d] = b.b + d
starts, pos = [], 0
for k, (_, _, L) in enumerate(sw):
    tt = None
    for d in range(len(L)):
        if s2a[pos + d] is not None: tt = atimes[s2a[pos + d]] - 0.03 * d; break
    starts.append(tt); pos += len(L)
known = [i for i, x in enumerate(starts) if x is not None]
for i in range(len(starts)):
    if starts[i] is None:
        lo = max([j for j in known if j < i], default=None); hi = min([j for j in known if j > i], default=None)
        a0 = starts[lo] if lo is not None else 0.0; a1 = starts[hi] if hi is not None else dur - 0.3
        i0 = lo if lo is not None else -1; i1 = hi if hi is not None else len(starts)
        starts[i] = a0 + (a1 - a0) * (i - i0) / (i1 - i0)
for i in range(1, len(starts)): starts[i] = max(starts[i], starts[i - 1] + 0.08)

# speech envelope (for word ends and the robot's mouth)
fps = 30; hop = sr // fps
env = np.array([np.sqrt(np.mean(a[k:k + hop] ** 2)) if k + hop <= len(a) else 0 for k in range(0, len(a), hop)])
env = env / (np.percentile(env, 98) + 1e-9)
def speech_end(t0, t1):
    i0, i1 = int(t0 * fps), min(len(env), int(t1 * fps) + 1)
    seg = env[i0:i1]; on = np.where(seg > 0.12)[0]
    return t0 + (on[-1] + 1) / fps if len(on) else t1

lines = []
for li, line in enumerate(LINES):
    idx = [k for k, x in enumerate(sw) if x[0] == li]
    words = []
    for j, k in enumerate(idx):
        nxt = starts[k + 1] if k + 1 < len(sw) else dur
        end = min(nxt, starts[k] + 1.0)
        if j == len(idx) - 1: end = min(speech_end(starts[k], nxt), starts[k] + 1.2)
        words.append({"w": sw[k][1], "t": round(starts[k], 3), "end": round(end, 3)})
    lines.append({"t": words[0]["t"], "end": words[-1]["end"], "text": line, "words": words})

# caption chunks: 2-3 words, broken at natural phrases, never ending on a small word
SMALL = set("a an the of to in on at is and or but so like all your my this that these its it be for with as are was by how does where which if each also even they made were then just into can".split())
subs = []
for l in lines:
    ws = l["words"]; groups, cur = [], []
    for wi, w in enumerate(ws):
        cur.append(w)
        punct = w["w"][-1] in '.,?!:'
        if punct or wi == len(ws) - 1: groups.append(cur); cur = []
        elif len(cur) == 3:
            if w["w"].lower() in SMALL: groups.append(cur[:-1]); cur = [w]
            else: groups.append(cur); cur = []
    merged = []
    for g in groups:
        if merged and len(g) == 1 and len(merged[-1]) <= 3 and merged[-1][-1]["w"][-1] not in '.,?!:': merged[-1] = merged[-1] + g
        else: merged.append(g)
    for g in merged:
        subs.append({"t": g[0]["t"], "end": g[-1]["end"] + 0.35, "text": " ".join(w["w"] for w in g), "words": g})
for k in range(len(subs) - 1): subs[k]["end"] = min(subs[k]["end"], subs[k + 1]["t"])

mouth = [round(float(min(1.0, x * 1.1)), 3) for x in env]
T = {"duration": round(dur, 3), "lines": lines, "subtitles": subs, "mouth": {"fps": fps, "open": mouth, "wide": [0.5] * len(mouth)},
     "heard": heard}
os.makedirs(os.path.dirname(outp) or '.', exist_ok=True)
open(outp, 'w').write("(function(){const T=" + json.dumps(T) + ";if(typeof module!=='undefined'&&typeof window==='undefined')module.exports=T;else globalThis.TIMING=T;})();\n")
e, n = v.wer(' '.join(SPOKEN.get(w.strip('.,?!:;'), w) for l in LINES for w in l.split()), heard)
print(f"duration {dur:.2f}s, {len(lines)} lines, {len(subs)} caption chunks, recognizer differs on {e}/{n} words")
for l in lines: print(f"  {l['t']:6.2f}-{l['end']:6.2f}  {l['text']}")
