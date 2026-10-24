"""Free voice pipeline v2: Kokoro via sherpa-onnx (lexicon + espeak pronunciation)
plus word timestamps measured by an offline speech recognizer.

Usage:
  python3 voice2.py synth <lines.json> <outdir> [sid]   # lines.json = list of spoken-text strings
  python3 voice2.py align <wav> -> prints JSON {words:[...], starts:[...]}
"""
import json, os, re, subprocess, sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter

HERE = os.path.dirname(os.path.abspath(__file__))
B = f"{HERE}/sherpa-onnx-v1.13.8-linux-x64-static/bin"
K = f"{HERE}/kokoro-multi-lang-v1_0"
Z = f"{HERE}/sherpa-onnx-zipformer-en-2023-06-26"

def tts(text, out, sid=3, speed=1.0):
    cmd = [f"{B}/sherpa-onnx-offline-tts", f"--kokoro-model={K}/model.onnx", f"--kokoro-voices={K}/voices.bin",
           f"--kokoro-tokens={K}/tokens.txt", f"--kokoro-data-dir={K}/espeak-ng-data", f"--kokoro-dict-dir={K}/dict",
           f"--kokoro-lexicon={K}/lexicon-us-en.txt,{K}/lexicon-zh.txt", f"--sid={sid}",
           f"--kokoro-length-scale={1.0 / speed:.3f}", f"--output-filename={out}", text]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out):
        raise RuntimeError(r.stderr[-800:])

def polish(path):
    """Gentle voice polish: rumble cut, slight presence lift, soft compression, peak normalize."""
    sr, a = wavfile.read(path)
    x = a.astype(np.float64) / (32768.0 if a.dtype == np.int16 else 1.0)
    b, c = butter(2, 80 / (sr / 2), 'high'); x = lfilter(b, c, x)
    b, c = butter(2, [2500 / (sr / 2), 5000 / (sr / 2)], 'band'); x = x + 0.18 * lfilter(b, c, x)
    env = np.abs(x); k = int(0.01 * sr)
    env = np.convolve(env, np.ones(k) / k, mode='same')
    thr = 0.12; gain = np.where(env > thr, (thr + (env - thr) / 2.5) / np.maximum(env, 1e-9), 1.0)
    x = x * gain
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.9
    wavfile.write(path, sr, (x * 32767).astype(np.int16))

def asr(wav):
    cmd = [f"{B}/sherpa-onnx-offline", f"--encoder={Z}/encoder-epoch-99-avg-1.int8.onnx",
           f"--decoder={Z}/decoder-epoch-99-avg-1.onnx", f"--joiner={Z}/joiner-epoch-99-avg-1.int8.onnx",
           f"--tokens={Z}/tokens.txt", wav]
    r = subprocess.run(cmd, capture_output=True, text=True)
    line = [l for l in (r.stdout + r.stderr).splitlines() if l.startswith('{')][-1]
    j = json.loads(line)
    words, starts = [], []
    for tok, ts in zip(j["tokens"], j["timestamps"]):
        if tok.startswith(" ") or not words:
            words.append(tok.strip()); starts.append(ts)
        else:
            words[-1] += tok
    return j["text"], words, starts

def norm(s):
    return re.sub(r"[^a-z' ]+", " ", s.lower()).split()

def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), int); d[:, 0] = range(len(r) + 1); d[0, :] = range(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r), len(h)], len(r)

if __name__ == '__main__':
    if sys.argv[1] == 'synth':
        lines = json.load(open(sys.argv[2])); out = sys.argv[3]; sid = int(sys.argv[4]) if len(sys.argv) > 4 else 3
        os.makedirs(out, exist_ok=True)
        for i, t in enumerate(lines):
            p = f"{out}/line_{i:02d}.wav"; tts(t, p, sid); polish(p)
            sr, a = wavfile.read(p); print(i, round(len(a) / sr, 2))
    elif sys.argv[1] == 'align':
        print(json.dumps(dict(zip(("text", "words", "starts"), asr(sys.argv[2])))))
