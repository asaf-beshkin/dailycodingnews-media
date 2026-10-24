import math, sys, re, subprocess, functools, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import butter, lfilter
from scipy.io import wavfile

W, H, FPS = 1080, 1920, 30
BG0 = (4, 6, 10)
VIO = (56, 189, 248); PINK = (186, 230, 253); MINT = (125, 211, 252)
WHITE = (241, 245, 249); MUTE = (140, 158, 180); YEL = (224, 242, 254); INK = (2, 4, 8)
DEEP = (12, 20, 32)

FD = '/usr/share/fonts/opentype/inter/'
FONTS = {'x': FD + 'InterDisplay-Black.otf', 'b': FD + 'InterDisplay-Bold.otf', 'sb': FD + 'Inter-SemiBold.otf',
         'm': FD + 'Inter-Medium.otf', 'mono': '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf'}
@functools.lru_cache(None)
def F(w, s): return ImageFont.truetype(FONTS[w], s)

# ======================= SCRIPT & TIMELINE =======================
# ===== LINES: (scene letter, caption text). Same scene letter = same continuous voice take. =====
LINES = [
    ("A", "Your brain has about 86 billion neurons. AI copied the idea... using nothing but numbers."),
    ("B", "This is a neural network. Let's teach one to spot a cat."),
    ("C", "A photo goes in as millions of tiny numbers: pixels."),
    ("D", "Each artificial neuron takes in numbers, multiplies each one by a weight, and adds them up."),
    ("D", "If the total is big enough, it fires, passing a signal to the next layer."),
    ("E", "Early layers spot simple edges. Deeper layers combine them into ears, eyes, and whiskers."),
    ("E", "But at first, the weights are random. So it guesses... toaster."),
    ("F", "Now comes the magic: training."),
    ("F", "We measure how wrong it was, then nudge every weight a tiny bit to shrink the error."),
    ("F", "This is called backpropagation. Repeat it millions of times, and the network learns."),
    ("G", "Cat. 98 percent."),
    ("H", "Nobody programmed the rules. It found them... by making mistakes."),
]
SPECIAL = {'bluetooth': 2, '1,600': 5, '2.4': 4, '79': 5, 'wi-fi': 2, 'gigahertz': 3, 'earbuds': 2, 'interference': 4, 'encrypted': 3, 'harald': 2, 'norway': 2, 'devices': 3, 'genius': 2, 'energy': 3, 'microwave': 3, 'radio': 3, 'c++': 3, 'c++-level': 5, 'ai': 2, 'gpu': 3, 'gpus': 3, 'cuda': 2, 'pydantic': 3, 'tiktoken': 2,
           'tokenizers': 4, 'polars': 2, 'python': 2, "you're": 1, "here's": 1, "that's": 1, "isn't": 2,
           "don't": 1, 'rust': 1, 'hugging': 2, 'face': 1, 'secret': 2, 'races': 2, 'pauses': 2, 'package': 2}
def syl(w):
    k = w.lower().strip('.,?!:;')
    if k in SPECIAL: return SPECIAL[k]
    n = len(re.findall(r'[aeiouy]+', k))
    if k.endswith('e') and n > 1 and not k.endswith('le'): n -= 1
    return max(1, n)
VO_DIR = os.environ.get('VO_DIR', 'takes')
import json as _json, difflib as _difflib
# One continuous take per scene; word timings measured by the speech recognizer (voice2.py align).
TAKES = _json.load(open(f'{VO_DIR}/takes.json'))   # [{"lines":[i,...], "wav":"take_0.wav", "align":{...}}]
SPOKEN = {'86': 'eighty six', 'AI': 'a i', 'idea...': 'idea', 'guesses...': 'guesses', 'them...': 'them', '98': 'ninety eight'}
def _letters(s): return re.sub(r'[^a-z]', '', s.lower())
def _load_take(p):
    sr, a = wavfile.read(p); return a.astype(np.float32) / 32767, sr
def _speech_bounds(a, sr):
    fr_ = int(sr / 100); e = np.array([np.sqrt(np.mean(a[k:k + fr_] ** 2)) for k in range(0, len(a), fr_)])
    idx = np.where(e > e.max() * 0.05)[0]
    return idx[0] / 100.0, (idx[-1] + 1) / 100.0
VO, VO_T = [], []
WT, LS, LE, LSC = [None] * len(LINES), [0.0] * len(LINES), [0.0] * len(LINES), [sc for sc, _ in LINES]
GAPS = {10: 0.2, 11: 0.2}
t = 0.5
for ti, tk in enumerate(TAKES):
    a, sr = _load_take(f"{VO_DIR}/{tk['wav']}")
    on, off = _speech_bounds(a, sr)
    first = tk['lines'][0]
    if ti > 0:
        t += 0.15 + GAPS.get(first, 0)
    clip0 = t - on
    VO.append(a); VO_T.append(clip0)
    # characters of the recognizer output, each with a time
    al = tk['align']; achars, atimes = [], []
    for wi, (w, st) in enumerate(zip(al['words'], al['starts'])):
        nxt = al['starts'][wi + 1] if wi + 1 < len(al['starts']) else off
        L = _letters(w)
        for ci, ch in enumerate(L):
            achars.append(ch); atimes.append(st + (nxt - st) * ci / max(1, len(L)))
    # characters of the script words (spoken form)
    swords = []
    for li in tk['lines']:
        for w in LINES[li][1].split():
            swords.append((li, w, _letters(SPOKEN.get(w.strip('.,?!:;'), w))))
    schars, sowner = [], []
    for k, (_, _, L) in enumerate(swords):
        for ch in L: schars.append(ch); sowner.append(k)
    m = _difflib.SequenceMatcher(None, ''.join(schars), ''.join(achars), autojunk=False)
    s2a = [None] * len(schars)
    for blk in m.get_matching_blocks():
        for d in range(blk.size): s2a[blk.a + d] = blk.b + d
    starts, pos = [], 0
    for k, (_, _, L) in enumerate(swords):
        tt = None
        for d in range(len(L)):
            if s2a[pos + d] is not None: tt = atimes[s2a[pos + d]] - (0.03 * d); break
        starts.append(tt); pos += len(L)
    # fill gaps by interpolation, enforce increasing
    known = [i for i, v in enumerate(starts) if v is not None]
    for i in range(len(starts)):
        if starts[i] is None:
            lo = max([j for j in known if j < i], default=None); hi = min([j for j in known if j > i], default=None)
            a0 = starts[lo] if lo is not None else on; a1 = starts[hi] if hi is not None else off - 0.3
            i0 = lo if lo is not None else -1; i1 = hi if hi is not None else len(starts)
            starts[i] = a0 + (a1 - a0) * (i - i0) / (i1 - i0)
    starts[0] = max(starts[0], on)
    for i in range(1, len(starts)): starts[i] = max(starts[i], starts[i - 1] + 0.08)
    for k, (li, w, _) in enumerate(swords):
        st = starts[k]; en = starts[k + 1] if k + 1 < len(swords) else off
        if WT[li] is None: WT[li] = []
        WT[li].append((w, clip0 + st, clip0 + min(en, st + 1.2)))
    for li in tk['lines']:
        LS[li] = WT[li][0][1]
        LE[li] = min(WT[li][-1][2], WT[li][-1][1] + 0.9)
    LE[tk['lines'][-1]] = clip0 + off
    t = clip0 + off
END_CARD = LE[-1] + 0.5
DUR = END_CARD + 2.8
SCN = {}
for i, sc in enumerate(LSC):
    if sc not in SCN: SCN[sc] = [0.0 if i == 0 else LS[i] - 0.4, None]
keys = list(SCN)
for a, b in zip(keys, keys[1:]): SCN[a][1] = SCN[b][0]
SCN[keys[-1]][1] = DUR + 1
def wt(li, key):
    for w, s, e in WT[li]:
        if key.lower() in w.lower(): return s
    raise KeyError(key)

# caption chunks
CHUNKS = []  # (words[(w,s,e)], start, end)
for i, words in enumerate(WT):
    cur, groups = [], []
    SMALL = {'a', 'an', 'the', 'of', 'to', 'in', 'on', 'at', 'is', 'and', 'or', 'but', 'so', 'like', 'all',
             'your', 'my', 'this', 'that', 'these', 'its', 'it', 'be', 'for', 'with', 'as', 'are', 'was', 'by',
             'how', 'does', 'where', 'which', 'if', 'each', 'also', 'even', 'they', 'made', 'were'}
    for wi, w in enumerate(words):
        cur.append(w)
        punct = w[0][-1] in '.,?!:'
        last = wi == len(words) - 1
        if punct or last:
            groups.append(cur); cur = []
        elif len(cur) == 3:
            if w[0].lower() in SMALL:
                groups.append(cur[:-1]); cur = [w]
            else:
                groups.append(cur); cur = []
    # merge single orphan words into the previous chunk of the same phrase when it stays short
    merged = []
    for g in groups:
        if merged and len(g) == 1 and len(merged[-1]) <= 3 and merged[-1][-1][0][-1] not in '.,?!:':
            merged[-1] = merged[-1] + g
        else:
            merged.append(g)
    groups = merged
    if cur: groups.append(cur)
    for gi, g in enumerate(groups):
        s = g[0][1] - 0.05
        if gi + 1 < len(groups): e = groups[gi + 1][0][1] - 0.05
        else:
            e = LE[i] + 0.5
            if i + 1 < len(LS): e = min(e, LS[i + 1] - 0.06)
        CHUNKS.append((g, s, e))

# ======================= HELPERS =======================
def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def eo(x): x = clamp(x); return 1 - (1 - x) ** 3
def eio(x): x = clamp(x); return x * x * (3 - 2 * x)
def eob(x):
    x = clamp(x); c1 = 1.9; c3 = c1 + 1
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2
def pop(t, s, d=0.35): return eob((t - s) / d) if t >= s else 0.0
def fin(t, s, d=0.25): return eo((t - s) / d) if t >= s else 0.0
def lerpc(a, b, f): return tuple(int(a[i] + (b[i] - a[i]) * f) for i in range(len(a)))

def hgrad(w, h, c1, c2):
    x = np.linspace(0, 1, w)[None, :]; y = np.linspace(0, 1, h)[:, None]
    f = np.clip(x * 0.8 + y * 0.2, 0, 1)
    arr = np.zeros((h, w, 4), np.uint8)
    for i in range(3): arr[..., i] = (c1[i] + (c2[i] - c1[i]) * f).astype(np.uint8)
    arr[..., 3] = 255
    return Image.fromarray(arr, 'RGBA')
def solid(w, h, c): return Image.new('RGBA', (w, h), c if len(c) == 4 else c + (255,))

@functools.lru_cache(None)
def tsp(s, font, fill=WHITE, stroke=0, grad=False):
    l, tp, r, b = font.getbbox(s, stroke_width=stroke)
    w, h = int(r - l + 6), int(b - tp + 6)
    if grad:
        m = Image.new('L', (w, h), 0); ImageDraw.Draw(m).text((3 - l, 3 - tp), s, font=font, fill=255)
        im = hgrad(w, h, VIO, PINK); im.putalpha(m); return im
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((3 - l, 3 - tp), s, font=font, fill=fill, stroke_width=stroke, stroke_fill=INK)
    return im

@functools.lru_cache(None)
def card(w, h, style='glass', r=38, bw=4):
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle([0, 0, w - 1, h - 1], radius=r, fill=(255, 255, 255, 20))
    m = Image.new('L', (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], radius=r, outline=255, width=bw)
    g = {'grad': lambda: hgrad(w, h, VIO, PINK), 'mint': lambda: solid(w, h, MINT),
         'glass': lambda: solid(w, h, (255, 255, 255, 60)), 'vio': lambda: solid(w, h, VIO)}[style]()
    im.paste(g, (0, 0), m)
    return im

@functools.lru_cache(None)
def pill(text, font, style='grad', tc=INK):
    tw_ = font.getlength(text); h = int(font.size * 1.7); w = int(tw_ + font.size * 1.4)
    if style == 'grad': base = hgrad(w, h, VIO, PINK)
    else: base = solid(w, h, {'mint': MINT, 'white': WHITE, 'yel': YEL, 'dark': DEEP}[style])
    m = Image.new('L', (w, h), 0); ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], radius=h // 2, fill=255)
    base.putalpha(m)
    ImageDraw.Draw(base).text((w / 2, h / 2), text, font=font, fill=tc, anchor='mm')
    return base

def put(fr, sp, cx, cy, alpha=1.0, scale=1.0, rot=0.0):
    if alpha <= 0.01 or scale <= 0.02: return
    if rot: sp = sp.rotate(rot, resample=Image.BICUBIC, expand=True)
    if abs(scale - 1) > 0.004:
        sp = sp.resize((max(1, int(sp.width * scale)), max(1, int(sp.height * scale))), Image.BILINEAR)
    if alpha < 0.995:
        sp = sp.copy(); sp.putalpha(sp.getchannel('A').point(lambda v, al=alpha: int(v * al)))
    x, y = int(cx - sp.width / 2), int(cy - sp.height / 2)
    l, tp = max(0, -x), max(0, -y); r, b = min(sp.width, W - x), min(sp.height, H - y)
    if r <= l or b <= tp: return
    if (l, tp, r, b) != (0, 0, sp.width, sp.height): sp = sp.crop((l, tp, r, b))
    fr.alpha_composite(sp, dest=(x + l, y + tp))

def canvas(w, h): im = Image.new('RGBA', (w, h), (0, 0, 0, 0)); return im, ImageDraw.Draw(im)

# ======================= ICONS (drawn 2x, downsampled) =======================
def icon(fn, S, *a):
    im, d = canvas(2 * S, 2 * S); fn(d, 2 * S, *a)
    return im.resize((S, S), Image.LANCZOS)

def ic_net(d, Z, t):
    cols = [(0.14, 3, MINT), (0.5, 4, VIO), (0.86, 3, PINK)]
    nodes = [[(Z * x, Z * (0.5 + (k - (n - 1) / 2) * 0.22)) for k in range(n)] for x, n, _ in cols]
    e = 0
    for L in range(2):
        for a in nodes[L]:
            for b in nodes[L + 1]:
                d.line([a, b], fill=(255, 255, 255, 55), width=int(Z * 0.008))
                p = (t * 0.9 + e * 0.137) % 1
                px, py = a[0] + (b[0] - a[0]) * p, a[1] + (b[1] - a[1]) * p
                r = Z * 0.012
                d.ellipse([px - r, py - r, px + r, py + r], fill=MINT + (230,)); e += 1
    for L, (x, n, c) in enumerate(cols):
        for k, (nx, ny) in enumerate(nodes[L]):
            r = Z * (0.055 + 0.008 * math.sin(t * 4 + k + L))
            d.ellipse([nx - r, ny - r, nx + r, ny + r], fill=DEEP + (255,), outline=c + (255,), width=int(Z * 0.014))
            r2 = r * 0.45; d.ellipse([nx - r2, ny - r2, nx + r2, ny + r2], fill=c + (255,))

def ic_wheel(d, Z, ang):
    c = Z / 2; R = Z * 0.42; w = int(Z * 0.075)
    d.ellipse([c - R, c - R, c + R, c + R], outline=MINT, width=w)
    for k in range(3):
        a = math.radians(ang + 90 + k * 120)
        d.line([(c, c), (c + math.cos(a) * R * 0.95, c + math.sin(a) * R * 0.95)], fill=MINT, width=int(w * 0.8))
    r = Z * 0.11; d.ellipse([c - r, c - r, c + r, c + r], fill=MINT)

def ic_gear(d, Z, ang, color):
    c = Z / 2; n = 9; pts = []
    for k in range(n * 4):
        a = math.radians(ang) + k * 2 * math.pi / (n * 4)
        R = Z * (0.46 if k % 4 in (1, 2) else 0.36)
        pts.append((c + math.cos(a) * R, c + math.sin(a) * R))
    d.polygon(pts, fill=color)
    r = Z * 0.14; d.ellipse([c - r, c - r, c + r, c + r], fill=(0, 0, 0, 0))

def ic_chip(d, Z, glow):
    m = Z * 0.2; L, R = m, Z - m
    for k in range(5):
        p = L + (R - L) * (k + 0.5) / 5; s = Z * 0.025
        for (x0, y0, x1, y1) in [(p - s, L - Z * 0.09, p + s, L), (p - s, R, p + s, R + Z * 0.09),
                                 (L - Z * 0.09, p - s, L, p + s), (R, p - s, R + Z * 0.09, p + s)]:
            d.rectangle([x0, y0, x1, y1], fill=MUTE)
    d.rounded_rectangle([L, L, R, R], radius=Z * 0.06, fill=DEEP + (255,), outline=VIO, width=int(Z * 0.02))
    i0, i1 = Z * 0.33, Z * 0.67
    col = lerpc(PINK, MINT, glow)
    d.rounded_rectangle([i0, i0, i1, i1], radius=Z * 0.03, outline=col, width=int(Z * 0.018))
    d.text((Z / 2, Z / 2), "GPU", font=F('x', int(Z * 0.11)), fill=WHITE, anchor='mm')

def ic_nogc(d, Z):
    c = Z / 2
    bx0, bx1, by0, by1 = Z * 0.33, Z * 0.67, Z * 0.36, Z * 0.72
    d.polygon([(bx0, by0), (bx1, by0), (bx1 - Z * 0.03, by1), (bx0 + Z * 0.03, by1)], outline=WHITE, width=int(Z * 0.025))
    d.line([(Z * 0.28, Z * 0.32), (Z * 0.72, Z * 0.32)], fill=WHITE, width=int(Z * 0.03))
    d.line([(Z * 0.45, Z * 0.27), (Z * 0.55, Z * 0.27)], fill=WHITE, width=int(Z * 0.03))
    for x in (0.44, 0.5, 0.56):
        d.line([(Z * x, Z * 0.43), (Z * x, Z * 0.65)], fill=WHITE, width=int(Z * 0.018))
    R = Z * 0.44; w = int(Z * 0.06)
    d.ellipse([c - R, c - R, c + R, c + R], outline=PINK, width=w)
    a = math.radians(45); d.line([(c - math.cos(a) * R, c - math.sin(a) * R), (c + math.cos(a) * R, c + math.sin(a) * R)], fill=PINK, width=w)

def ic_shield(d, Z, ck):
    pts = [(Z * 0.5, Z * 0.08), (Z * 0.86, Z * 0.2), (Z * 0.82, Z * 0.55), (Z * 0.5, Z * 0.92), (Z * 0.18, Z * 0.55), (Z * 0.14, Z * 0.2)]
    d.polygon(pts, fill=(110, 231, 183, 50), outline=MINT, width=int(Z * 0.035))
    if ck > 0:
        a, b, c = (Z * 0.32, Z * 0.5), (Z * 0.45, Z * 0.63), (Z * 0.7, Z * 0.36)
        if ck < 0.5:
            f = ck / 0.5; d.line([a, (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)], fill=WHITE, width=int(Z * 0.05))
        else:
            f = (ck - 0.5) / 0.5
            d.line([a, b, (b[0] + (c[0] - b[0]) * f, b[1] + (c[1] - b[1]) * f)], fill=WHITE, width=int(Z * 0.05), joint='curve')

def ic_speed(d, Z, v):
    c = (Z / 2, Z * 0.58); R = Z * 0.42; w = int(Z * 0.06)
    d.arc([c[0] - R, c[1] - R, c[0] + R, c[1] + R], 160, 380, fill=(255, 255, 255, 50), width=w)
    end = 160 + 220 * v
    steps = 24
    for k in range(steps):
        a0 = 160 + (end - 160) * k / steps; a1 = 160 + (end - 160) * (k + 1) / steps + 0.5
        d.arc([c[0] - R, c[1] - R, c[0] + R, c[1] + R], a0, a1, fill=lerpc(MINT, PINK, k / steps), width=w)
    a = math.radians(end)
    d.line([c, (c[0] + math.cos(a) * R * 0.8, c[1] + math.sin(a) * R * 0.8)], fill=WHITE, width=int(Z * 0.03))
    r = Z * 0.05; d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=WHITE)

# ======================= BACKGROUND =======================
SW, SH = 135, 240
_gx = np.linspace(0, 1, SW)[None, :]; _gy = np.linspace(0, 1.78, SH)[:, None]
BLOBS = [(VIO, 0.13, 0.45), (MINT, 0.06, 0.40), ((20, 90, 140), 0.12, 0.38), ((30, 70, 120), 0.10, 0.5)]
def aurora(t):
    acc = np.zeros((SH, SW, 3), np.float32) + np.array(BG0, np.float32)
    pos = [(0.25 + 0.18 * math.sin(t * 0.21), 0.35 + 0.15 * math.cos(t * 0.17)),
           (0.8 + 0.15 * math.sin(t * 0.19 + 2), 1.35 + 0.18 * math.cos(t * 0.23 + 1)),
           (0.15 + 0.12 * math.cos(t * 0.15), 1.55 + 0.12 * math.sin(t * 0.2)),
           (0.75 + 0.2 * math.cos(t * 0.13 + 4), 0.75 + 0.2 * math.sin(t * 0.16))]
    for (c, inten, r), (px, py) in zip(BLOBS, pos):
        g = np.exp(-((_gx - px) ** 2 + (_gy - py) ** 2) / (r * r))[..., None]
        acc += g * np.array(c, np.float32) * inten
    im = Image.fromarray(np.clip(acc, 0, 255).astype(np.uint8), 'RGB').resize((W, H), Image.BICUBIC)
    return im.convert('RGBA')
rng = np.random.default_rng(7)
PART = [(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(8, 30), rng.uniform(1.5, 3.5), rng.uniform(0, 6.28)) for _ in range(45)]
def particles(fr, t):
    d = ImageDraw.Draw(fr)
    for x, y, sp, r, ph in PART:
        yy = (y - t * sp) % H; xx = x + 12 * math.sin(t * 0.5 + ph)
        b = 0.5 + 0.5 * math.sin(t * 2 + ph)
        c = lerpc((22, 32, 46), (150, 200, 235), b)
        d.ellipse([xx - r, yy - r, xx + r, yy + r], fill=c + (255,))

# ======================= SCENES =======================
# Cinematic episode: "Tech Legends, Ep. 1" — Linus Torvalds, Linux and Git.
# Everything is drawn from scratch: an original silhouette character (no real likeness), no logos or mascots.
from PIL import ImageFilter, ImageChops
_R = np.random.default_rng(1991)
MONO = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
@functools.lru_cache(None)
def FM(s): return ImageFont.truetype(MONO, s)

def vgrad(w, h, top, bot):
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    arr = np.empty((h, w, 4), np.uint8)
    for i in range(3):
        arr[..., i] = (top[i] * (1 - y) + bot[i] * y).astype(np.uint8)
    arr[..., 3] = 255
    return Image.fromarray(arr, 'RGBA')

def glow(w, h, cx, cy, r, color, alpha, power=2.0):
    yy, xx = np.ogrid[0:h, 0:w]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
    a = np.clip(1 - d, 0, 1) ** power * alpha
    arr = np.zeros((h, w, 4), np.uint8); arr[..., 0] = color[0]; arr[..., 1] = color[1]; arr[..., 2] = color[2]
    arr[..., 3] = (a * 255).astype(np.uint8)
    return Image.fromarray(arr, 'RGBA')

def rimlit(mask, fill=(5, 8, 12), rim=MINT, width=9, strength=1.7, top_bias=True):
    S = mask.size
    body = Image.new('RGBA', S, fill + (255,)); body.putalpha(mask)
    er = mask.filter(ImageFilter.MinFilter(width))
    edge = ImageChops.subtract(mask, er).filter(ImageFilter.GaussianBlur(3))
    if top_bias:
        g = Image.linear_gradient('L').rotate(180).resize(S)
        edge = ImageChops.multiply(edge, g)
    edge = edge.point(lambda v: min(255, int(v * strength)))
    r = Image.new('RGBA', S, rim + (0,)); r.putalpha(edge)
    body.alpha_composite(r)
    return body

def cam(img, z, cx, cy):
    if z <= 1.0005: return img
    cw, ch = W / z, H / z
    x0 = min(max(cx - cw / 2, 0), W - cw); y0 = min(max(cy - ch / 2, 0), H - ch)
    return img.resize((W, H), Image.BILINEAR, box=(x0, y0, x0 + cw, y0 + ch))

def ease_span(t, a, b): return eio((t - a) / max(1e-3, b - a))

# ---------- the character: original silhouette seen from behind, lit by the monitor
def person_mask(S):
    m = Image.new('L', (S, S), 0); d = ImageDraw.Draw(m); c = S / 2
    d.rounded_rectangle([c - 0.44 * S, 0.50 * S, c + 0.44 * S, 1.3 * S], radius=int(0.2 * S), fill=255)
    d.polygon([(c - 0.22 * S, 0.53 * S), (c + 0.22 * S, 0.53 * S), (c + 0.085 * S, 0.37 * S), (c - 0.085 * S, 0.37 * S)], fill=255)
    d.ellipse([c - 0.125 * S, 0.13 * S, c + 0.125 * S, 0.43 * S], fill=255)
    d.ellipse([c - 0.152 * S, 0.255 * S, c - 0.1 * S, 0.335 * S], fill=255)
    d.ellipse([c + 0.1 * S, 0.255 * S, c + 0.152 * S, 0.335 * S], fill=255)
    rr = np.random.default_rng(5)
    for i in range(46):
        a = rr.uniform(math.pi * 1.02, math.pi * 1.98)
        x = c + math.cos(a) * 0.118 * S; y = 0.28 * S + math.sin(a) * 0.142 * S
        s = rr.uniform(0.018, 0.04) * S
        d.ellipse([x - s, y - s, x + s, y + s], fill=255)
    return m.filter(ImageFilter.GaussianBlur(1.5))
PERSON = rimlit(person_mask(820))

FLAKES = [(_R.uniform(0, W), _R.uniform(0, H), _R.uniform(40, 150), _R.uniform(1.4, 6.5), _R.uniform(0, 6.28)) for _ in range(190)]
def snow_layer(t, amt=1.0):
    im = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for x, y, sp, r, ph in FLAKES:
        yy = (y + t * sp) % H; xx = (x + 26 * math.sin(t * 0.7 + ph) + t * 9) % W
        a = int(255 * min(1, 0.3 + r / 7) * amt)
        d.ellipse([xx - r, yy - r, xx + r, yy + r], fill=(226, 238, 250, a))
    return im

# ---------- exterior: Helsinki at night, 1991
def build_exterior():
    sky = vgrad(W, H, (8, 16, 32), (2, 4, 9))
    sky.alpha_composite(glow(W, H, 800, 300, 580, (110, 170, 225), 0.22, 2.4))
    d = ImageDraw.Draw(sky)
    for i in range(150):
        x, y = _R.uniform(0, W), _R.uniform(0, 950); r = _R.uniform(0.6, 1.9); v = int(_R.uniform(90, 230))
        d.ellipse([x - r, y - r, x + r, y + r], fill=(v, min(255, v + 15), 255, 255))
    d.ellipse([755, 255, 845, 345], fill=(226, 240, 252, 255))
    far = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(far)
    x = -30
    while x < W + 30:
        bw = int(_R.integers(70, 160)); bh = int(_R.integers(240, 600)); top = 1260 - bh; col = (11, 19, 31, 255)
        d.rectangle([x, top, x + bw, 1750], fill=col)
        if _R.random() < 0.3: d.rectangle([x + bw // 2 - 3, top - 60, x + bw // 2 + 3, top], fill=col)
        for wy in range(top + 18, 1240, 24):
            for wx in range(x + 10, x + bw - 14, 20):
                if _R.random() < 0.15:
                    v = _R.uniform(0.35, 1.0)
                    d.rectangle([wx, wy, wx + 8, wy + 11], fill=(int(60 * v) + 20, int(120 * v) + 30, int(180 * v) + 40, 255))
        x += bw + int(_R.integers(2, 16))
    far = far.filter(ImageFilter.GaussianBlur(2.6))
    near = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(near)
    d.rectangle([150, 430, 930, 1760], fill=(13, 21, 32, 255))
    d.rectangle([132, 412, 948, 446], fill=(18, 27, 40, 255))
    d.rectangle([132, 402, 948, 414], fill=(170, 192, 214, 255))           # snow on the roof edge
    for i in range(1600):
        xx, yy = _R.uniform(150, 930), _R.uniform(446, 1760); v = int(_R.uniform(-5, 6))
        d.point((xx, yy), fill=(13 + v, 21 + v, 32 + v, 255))
    cols = [235, 387, 540, 693, 845]; rows = [560, 770, 980, 1190, 1400]
    for ry in rows:
        for cx_ in cols:
            x0, y0, x1, y1 = cx_ - 50, ry - 70, cx_ + 50, ry + 70
            hero = (cx_ == 540 and ry == 770)
            if hero:
                win = vgrad(100, 140, (165, 222, 252), (85, 150, 205)); near.paste(win, (x0, y0))
                pm = person_mask(150).resize((96, 96), Image.LANCZOS)
                body = Image.new('RGBA', (96, 96), (10, 16, 26, 255)); body.putalpha(pm)
                near.alpha_composite(body, (x0 + 4, y1 - 70))
            else:
                lit = _R.random() < 0.18
                d.rectangle([x0, y0, x1, y1], fill=(36, 62, 92, 255) if lit else (8, 13, 20, 255))
            d.rectangle([x0, y0, x1, y1], outline=(24, 33, 46, 255), width=6)
            d.line([(cx_, y0), (cx_, y1)], fill=(24, 33, 46, 255), width=5)
            d.line([(x0, ry - 10), (x1, ry - 10)], fill=(24, 33, 46, 255), width=5)
            d.rectangle([x0 - 8, y1 + 4, x1 + 8, y1 + 14], fill=(176, 196, 216, 255))     # snow on the sill
    near.alpha_composite(glow(W, H, 540, 770, 360, (120, 195, 245), 0.36, 2.0))
    near.alpha_composite(vgrad(W, 300, (34, 48, 62), (16, 23, 32)), (0, 1650))
    d = ImageDraw.Draw(near)
    d.rectangle([96, 1110, 108, 1700], fill=(9, 13, 20, 255))
    d.polygon([(84, 1112), (152, 1092), (160, 1104), (92, 1124)], fill=(9, 13, 20, 255))
    near.alpha_composite(glow(W, H, 150, 1108, 300, (150, 210, 245), 0.42, 2.0))
    cone = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(cone).polygon([(140, 1110), (160, 1110), (420, 1720), (-120, 1720)], fill=(150, 205, 240, 34))
    near.alpha_composite(cone.filter(ImageFilter.GaussianBlur(14)))
    near.alpha_composite(glow(W, H, 540, 1700, 700, (60, 100, 140), 0.25, 1.4))
    return sky, far, near
EXT = build_exterior()

def exterior(t, z, fx=540, fy=770, snow=1.0):
    sky, far, near = EXT
    img = cam(sky, 1 + (z - 1) * 0.15, fx, fy).copy()
    img.alpha_composite(cam(far, 1 + (z - 1) * 0.45, fx, fy))
    img.alpha_composite(cam(near, z, fx, fy))
    img.alpha_composite(snow_layer(t, snow))
    return img

# ---------- interior rooms (1991 CRT, 2005 flat screen)
def build_interior(year):
    im = vgrad(W, H, (10, 15, 23), (4, 6, 10)); d = ImageDraw.Draw(im)
    wx0, wy0, wx1, wy1 = 60, 230, 330, 640
    win = vgrad(wx1 - wx0, wy1 - wy0, (12, 26, 46), (6, 12, 22)); wd = ImageDraw.Draw(win)
    for i in range(9):
        bx = int(_R.integers(-20, 250)); bh = int(_R.integers(90, 230)); bw = int(_R.integers(40, 90))
        wd.rectangle([bx, 410 - bh, bx + bw, 410], fill=(8, 14, 24, 255))
        for k in range(6):
            wx_ = bx + int(_R.integers(4, max(5, bw - 10))); wy_ = 410 - int(_R.integers(12, bh))
            if _R.random() < 0.5: wd.rectangle([wx_, wy_, wx_ + 6, wy_ + 8], fill=(90, 150, 200, 255))
    for i in range(60):
        x, y, r = _R.uniform(0, 270), _R.uniform(0, 410), _R.uniform(1, 2.6)
        wd.ellipse([x - r, y - r, x + r, y + r], fill=(205, 226, 246, 255))
    im.paste(win, (wx0, wy0))
    d.rectangle([wx0 - 14, wy0 - 14, wx1 + 14, wy1 + 14], outline=(24, 32, 44), width=14)
    d.line([((wx0 + wx1) // 2, wy0), ((wx0 + wx1) // 2, wy1)], fill=(24, 32, 44), width=10)
    d.line([(wx0, (wy0 + wy1) // 2), (wx1, (wy0 + wy1) // 2)], fill=(24, 32, 44), width=10)
    beam = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(beam).polygon([(wx0, wy1), (wx1, wy1), (640, 1110), (240, 1110)], fill=(120, 170, 220, 30))
    im.alpha_composite(beam.filter(ImageFilter.GaussianBlur(16)))
    if year == 1991:
        d.rectangle([820, 300, 1050, 930], fill=(9, 13, 19))
        for sy in (450, 610, 770, 930):
            d.rectangle([820, sy - 8, 1050, sy], fill=(17, 23, 31))
            x = 832
            while x < 1035:
                bw = int(_R.integers(14, 30)); bh = int(_R.integers(90, 140))
                c = int(_R.integers(13, 26)); d.rectangle([x, sy - 8 - bh, x + bw, sy - 8], fill=(c, c + 6, c + 14))
                x += bw + 3
    else:
        cal = (730, 150, 1020, 450)
        d.rectangle(cal, fill=(20, 27, 37)); d.rectangle([cal[0], cal[1], cal[2], cal[1] + 64], fill=(28, 50, 72))
        d.text(((cal[0] + cal[2]) / 2, cal[1] + 32), "APRIL 2005", font=F('x', 30), fill=WHITE, anchor='mm')
    d.rectangle([0, 1000, W, 1080], fill=(16, 21, 28)); d.rectangle([0, 1080, W, H], fill=(7, 9, 13))
    if year == 1991:
        d.rounded_rectangle([290, 450, 790, 990], radius=36, fill=(26, 32, 40))
        d.rounded_rectangle([318, 478, 762, 900], radius=26, fill=(14, 18, 24))
        scr = (350, 510, 730, 868)
        d.polygon([(400, 990), (680, 990), (720, 1010), (360, 1010)], fill=(20, 25, 32))
        d.rounded_rectangle([250, 1020, 830, 1070], radius=10, fill=(22, 27, 34))
    else:
        d.rounded_rectangle([170, 470, 910, 920], radius=14, fill=(20, 24, 30))
        scr = (192, 492, 888, 898)
        d.rectangle([510, 920, 570, 990], fill=(18, 22, 28)); d.rounded_rectangle([420, 985, 660, 1005], radius=6, fill=(18, 22, 28))
        d.rounded_rectangle([230, 1020, 850, 1060], radius=8, fill=(22, 27, 34))
    sc = ((scr[0] + scr[2]) / 2, (scr[1] + scr[3]) / 2)
    im.alpha_composite(glow(W, H, sc[0], sc[1], 640, (90, 160, 220), 0.30, 1.8))
    im.alpha_composite(glow(W, H, sc[0], 1040, 540, (110, 180, 235), 0.22, 2.0))
    return im, scr
ROOM91, SCR91 = build_interior(1991)
ROOM05, SCR05 = build_interior(2005)

def scanlines(w, h, gap=4, a=46):
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for y in range(0, h, gap): d.line([(0, y), (w, y)], fill=(0, 0, 0, a))
    return im
SCAN91 = scanlines(SCR91[2] - SCR91[0], SCR91[3] - SCR91[1])

CRT_LINES = ["> NEW MESSAGE", "", "Hi everyone!", "I'm making a free", "operating system.", "It's just a hobby.", "Nothing big."]
HOBBY_LINE, HOBBY_TXT = 5, "just a hobby"

def crt_text(w, h, font, lh, x0, y0, chars, hi=0.0, strike=0.0, cursor_t=0.0):
    im = vgrad(w, h, (8, 22, 32), (4, 12, 20)); d = ImageDraw.Draw(im)
    left = chars; cur = None
    for i, line in enumerate(CRT_LINES):
        if left <= 0 and cur is None: cur = (x0, y0 + i * lh); break
        shown = line[:max(0, int(left))]; left -= len(line) + 1
        y = y0 + i * lh
        if i == HOBBY_LINE and hi > 0 and HOBBY_TXT in shown:
            px = x0 + font.getlength(line[:line.index(HOBBY_TXT)]); pw = font.getlength(HOBBY_TXT)
            d.rounded_rectangle([px - 8, y - 6, px + pw + 8, y + lh - 4], radius=8, fill=(40, 110, 160, int(200 * hi)))
            if strike > 0:
                d.line([(px - 6, y + lh * 0.45), (px - 6 + (pw + 12) * strike, y + lh * 0.45)], fill=WHITE + (255,), width=max(3, lh // 9))
        d.text((x0, y), shown, font=font, fill=MINT if i else (150, 200, 230))
        if left <= 0 and cur is None: cur = (x0 + font.getlength(shown), y)
    if cur and int(cursor_t * 2.5) % 2 == 0:
        d.rectangle([cur[0] + 3, cur[1] + 4, cur[0] + font.size * 0.55, cur[1] + lh - 8], fill=MINT)
    return im

TYPE_A0 = None   # set after timeline
def type_chars(t, a, b):
    total = sum(len(l) + 1 for l in CRT_LINES)
    return total * clamp((t - a) / max(0.1, b - a))

def room91(t, chars, z=1.0, fx=540, fy=900, typing=1.0):
    img = ROOM91.copy()
    w, h = SCR91[2] - SCR91[0], SCR91[3] - SCR91[1]
    scr = crt_text(w, h, FM(22), 34, 22, 22, chars, cursor_t=t); scr.alpha_composite(SCAN91)
    img.paste(scr, (SCR91[0], SCR91[1]))
    dy = typing * (3 * math.sin(t * 15) + 2 * math.sin(t * 9.3)); dx = typing * 2 * math.sin(t * 4.1)
    img.alpha_composite(PERSON, (int(540 - 410 - 40 + dx), int(880 + dy)))
    return cam(img, z, fx, fy)

def crt_close(t, chars, hi=0.0, strike=0.0, z=1.0):
    img = Image.new('RGBA', (W, H), (4, 6, 9, 255)); d = ImageDraw.Draw(img)
    d.rounded_rectangle([20, 230, 1060, 1270], radius=70, fill=(24, 30, 38))
    d.rounded_rectangle([60, 270, 1020, 1230], radius=52, fill=(12, 16, 22))
    sw, sh = 880, 880
    scr = crt_text(sw, sh, FM(54), 92, 50, 70, chars, hi=hi, strike=strike, cursor_t=t)
    scr.alpha_composite(scanlines(sw, sh, 6, 50))
    mask = Image.new('L', (sw, sh), 0); ImageDraw.Draw(mask).rounded_rectangle([0, 0, sw - 1, sh - 1], radius=44, fill=255)
    img.paste(scr, (100, 310), mask)
    img.alpha_composite(glow(W, H, 540, 750, 620, (90, 160, 220), 0.18, 1.6))
    return cam(img, z, 540, 760)

# ---------- globe at night
CONT = [(50, 15, 14, 22), (63, 20, 9, 11), (60, 90, 18, 55), (35, 98, 18, 33), (22, 78, 12, 10), (8, 18, 27, 20),
        (-15, 27, 18, 13), (25, 46, 10, 14), (5, 112, 7, 14), (-25, 134, 11, 17), (48, -100, 16, 32), (62, -110, 10, 30),
        (25, -101, 9, 12), (-14, -58, 22, 14), (72, -40, 9, 14), (36, 138, 6, 5), (54, -3, 4, 3)]
def build_globe_tex():
    Wt, Ht = 1024, 512
    lon = np.linspace(-180, 180, Wt, endpoint=False)[None, :]; lat = np.linspace(90, -90, Ht)[:, None]
    land = np.full((Ht, Wt), -1.0)
    for la, lo, rla, rlo in CONT:
        dlo = (lon - lo + 180) % 360 - 180
        land = np.maximum(land, 1 - ((lat - la) / rla) ** 2 - (dlo / rlo) ** 2)
    rl, ra = np.radians(lon), np.radians(lat)
    nz = np.sin(rl * 7 + 1.3) * np.cos(ra * 9) + 0.6 * np.sin(rl * 13 + ra * 11) + 0.4 * np.sin(ra * 17 - rl * 5)
    m = (land + 0.22 * nz) > 0.1
    tex = np.zeros((Ht, Wt, 3), np.float32); tex[:] = (3, 8, 16); tex[m] = (13, 24, 38)
    rr = np.random.default_rng(3)
    city = m & (rr.random((Ht, Wt)) < 0.07) & (np.abs(lat) < 68)
    lights = np.zeros((Ht, Wt), np.float32); lights[city] = rr.uniform(0.35, 1.0, (Ht, Wt))[city]
    li = Image.fromarray((lights * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))
    lights = np.maximum(lights, np.asarray(li).astype(np.float32) / 255 * 1.6)
    tex += lights[..., None] * np.array([150, 205, 245], np.float32)
    ys, xs = np.where(city)
    pick = rr.choice(len(ys), 60, replace=False)
    targets = [(90 - ys[i] / (Ht - 1) * 180, xs[i] / Wt * 360 - 180) for i in pick]
    return np.clip(tex, 0, 255).astype(np.uint8), targets
GTEX, GTARGETS = build_globe_tex()
GR = 420
_gy, _gx = np.mgrid[-GR:GR, -GR:GR].astype(np.float32) + 0.5
_vx = _gx / GR; _vy = -_gy / GR; _r2 = _vx ** 2 + _vy ** 2
_vz = np.sqrt(np.clip(1 - _r2, 0, 1))
PHI = math.radians(30)
_Yg = _vy * math.cos(PHI) + _vz * math.sin(PHI); _Zg = -_vy * math.sin(PHI) + _vz * math.cos(PHI)
_lat = np.degrees(np.arcsin(np.clip(_Yg, -1, 1))); _lonb = np.degrees(np.arctan2(_vx, _Zg))
_vi = ((90 - _lat) / 180 * 511).astype(int).clip(0, 511)
_shade = (0.3 + 0.7 * _vz)[..., None]
_galpha = (np.clip((1 - np.sqrt(_r2)) * GR, 0, 1) * 255).astype(np.uint8)
def globe_img(lon0):
    u = (((_lonb + lon0 + 180) % 360) / 360 * 1024).astype(int) % 1024
    col = GTEX[_vi, u].astype(np.float32) * _shade
    arr = np.zeros((2 * GR, 2 * GR, 4), np.uint8); arr[..., :3] = col.astype(np.uint8); arr[..., 3] = _galpha
    return Image.fromarray(arr, 'RGBA')
def gproj(lat, lon, lon0, h=1.0):
    la, lo = math.radians(lat), math.radians(lon - lon0)
    X, Y, Z = math.cos(la) * math.sin(lo), math.sin(la), math.cos(la) * math.cos(lo)
    vy = Y * math.cos(PHI) - Z * math.sin(PHI); vz = Y * math.sin(PHI) + Z * math.cos(PHI)
    vis = vz > 0 or (X * X + vy * vy) * h * h > 1.0
    return X * h * GR, -vy * h * GR, vis
STARS = vgrad(W, H, (4, 8, 16), (1, 2, 5))
_ds = ImageDraw.Draw(STARS)
for i in range(260):
    x, y, r = _R.uniform(0, W), _R.uniform(0, H), _R.uniform(0.5, 1.8); v = int(_R.uniform(80, 220))
    _ds.ellipse([x - r, y - r, x + r, y + r], fill=(v, min(255, v + 15), 255, 255))
ATMO = glow(2 * GR + 240, 2 * GR + 240, GR + 120, GR + 120, GR + 110, (90, 170, 235), 0.55, 1.2)
FIN = (60.17, 24.94)

def globe_scene(t, t0, arcs_t=None, dim=0.0, cy=760, scale=1.0):
    img = STARS.copy()
    lon0 = 14 + 3.2 * (t - t0)
    img.alpha_composite(ATMO, (540 - GR - 120, cy - GR - 120))
    g = globe_img(lon0); img.alpha_composite(g, (540 - GR, cy - GR))
    ov = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    fx, fy, fv = gproj(FIN[0], FIN[1], lon0)
    if arcs_t is not None:
        for i, (la, lo) in enumerate(GTARGETS):
            st = arcs_t + i * 0.045
            p = clamp((t - st) / 0.75)
            if p <= 0: continue
            dlo = (lo - FIN[1] + 180) % 360 - 180
            dist = math.hypot(la - FIN[0], dlo)
            pts = []
            for k in range(int(28 * p) + 1):
                s = k / 28
                h = 1 + 0.16 * math.sin(math.pi * s) * min(1, dist / 50)
                x, y, v = gproj(FIN[0] + (la - FIN[0]) * s, FIN[1] + dlo * s, lon0, h)
                if v: pts.append((540 + x, cy + y))
            if len(pts) > 1: d.line(pts, fill=(150, 215, 250, 200), width=3)
            if p >= 1:
                x, y, v = gproj(la, lo, lon0)
                if v:
                    r = 6 + 3 * math.sin(t * 6 + i)
                    d.ellipse([540 + x - r, cy + y - r, 540 + x + r, cy + y + r], fill=(220, 240, 255, 230))
    if fv:
        r = 11 + 5 * math.sin(t * 5)
        d.ellipse([540 + fx - r * 2.2, cy + fy - r * 2.2, 540 + fx + r * 2.2, cy + fy + r * 2.2], outline=(200, 235, 255, 200), width=3)
        d.ellipse([540 + fx - r, cy + fy - r, 540 + fx + r, cy + fy + r], fill=(235, 248, 255, 255))
    img.alpha_composite(ov)
    if dim > 0: img.alpha_composite(Image.new('RGBA', (W, H), (0, 0, 0, int(170 * dim))))
    if scale != 1.0: img = cam(img, scale, 540, cy)
    return img

# ---------- phone in hand
def build_bokeh():
    im = vgrad(W, H, (6, 12, 22), (2, 4, 8)); lay = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    for i in range(80):
        x, y, r = _R.uniform(-50, W + 50), _R.uniform(100, H - 200), _R.uniform(20, 95)
        c = [(56, 140, 200), (125, 200, 245), (40, 90, 150), (180, 220, 250)][i % 4]
        d.ellipse([x - r, y - r, x + r, y + r], fill=c + (int(_R.uniform(35, 90)),))
    im.alpha_composite(lay.filter(ImageFilter.GaussianBlur(14)))
    return im
BOKEH = build_bokeh()
def build_phone():
    pw, ph = 420, 860
    m = Image.new('L', (pw, ph), 0); ImageDraw.Draw(m).rounded_rectangle([0, 0, pw - 1, ph - 1], radius=62, fill=255)
    ph_img = rimlit(m, fill=(14, 18, 24), width=7, strength=1.2, top_bias=False)
    scr = vgrad(pw - 36, ph - 36, (20, 46, 72), (6, 14, 26)); d = ImageDraw.Draw(scr)
    d.text(((pw - 36) / 2, 80), "20:41", font=F('x', 64), fill=WHITE, anchor='mm')
    cols = [(56, 189, 248), (125, 211, 252), (186, 230, 253), (40, 110, 170)]
    for r_ in range(4):
        for c_ in range(4):
            x0 = 30 + c_ * 88; y0 = 210 + r_ * 112
            d.rounded_rectangle([x0, y0, x0 + 64, y0 + 64], radius=16, fill=cols[(r_ * 3 + c_) % 4])
    msk = Image.new('L', scr.size, 0); ImageDraw.Draw(msk).rounded_rectangle([0, 0, scr.size[0] - 1, scr.size[1] - 1], radius=48, fill=255)
    ph_img.paste(scr, (18, 18), msk)
    return ph_img
PHONE = build_phone()
def build_hand():
    S = (900, 900); m = Image.new('L', S, 0); d = ImageDraw.Draw(m)
    d.rounded_rectangle([260, 520, 700, 980], radius=150, fill=255)                 # palm
    d.rounded_rectangle([170, 380, 300, 640], radius=60, fill=255)                  # thumb
    for i, y in enumerate([230, 330, 430, 530]):
        d.rounded_rectangle([600, y, 760, y + 92], radius=46, fill=255)             # fingers wrapping the side
    m = m.filter(ImageFilter.GaussianBlur(2))
    return rimlit(m, fill=(8, 11, 16), width=11, strength=1.4)
HAND = build_hand()
def phone_scene(t, t0):
    img = cam(BOKEH, 1.0 + 0.05 * (t - t0), 540, 900).copy()
    fy = 6 * math.sin(t * 1.6)
    img.alpha_composite(glow(W, H, 540, 840, 520, (110, 190, 245), 0.30, 1.6))
    hand = HAND
    img.alpha_composite(hand, (540 - 470, int(880 + fy)))
    ph = PHONE.rotate(-5, resample=Image.BICUBIC, expand=True)
    img.alpha_composite(ph, (int(540 - ph.width / 2), int(820 - ph.height / 2 + fy)))
    fing = Image.new('L', (900, 900), 0); d = ImageDraw.Draw(fing)
    for y in [230, 330, 430, 530]: d.rounded_rectangle([600, y, 760, y + 92], radius=46, fill=255)
    fing = rimlit(fing.filter(ImageFilter.GaussianBlur(2)), fill=(8, 11, 16), width=11, strength=1.4)
    img.alpha_composite(fing, (540 - 470, int(880 + fy)))
    return img

# ---------- server hall (supercomputers): real 3D perspective, dolly forward
def _build_fog():
    f = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    f.alpha_composite(glow(W, H, 540, 860, 900, (70, 120, 160), 0.30, 1.2))
    f.alpha_composite(glow(W, H, 540, 1500, 900, (40, 80, 120), 0.35, 1.0))
    return f
FOG = _build_fog()
def _build_shafts():
    lay = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    for k, (x0, w0) in enumerate(((420, 40), (540, 60), (660, 40))):
        d.polygon([(x0 - w0 / 2, 0), (x0 + w0 / 2, 0), (540 + (x0 - 540) * 2.6 + 160, H), (540 + (x0 - 540) * 2.6 - 160, H)], fill=(170, 215, 245, 22))
    return lay.filter(ImageFilter.GaussianBlur(40))
SHAFTS = _build_shafts()
def corridor(t, t0):
    im = Image.new('RGBA', (W, H), (3, 6, 10, 255)); d = ImageDraw.Draw(im)
    f, vx, vy = 900, 540, 860
    cz = (t - t0) * 2.4
    def P(X, Y, Z): return (vx + f * X / Z, vy - f * Y / Z)
    d.polygon([P(-1.25, -1, 0.5), P(1.25, -1, 0.5), P(1.25, -1, 60), P(-1.25, -1, 60)], fill=(9, 14, 21))
    d.polygon([P(-1.25, 1.3, 0.5), P(1.25, 1.3, 0.5), P(1.25, 1.3, 60), P(-1.25, 1.3, 60)], fill=(5, 8, 12))
    sp = 0.66; base = math.floor(cz / sp)
    for k in range(48, -2, -1):
        z = 0.75 + k * sp - (cz % sp)
        if z < 0.35: continue
        idx = base + k
        fogk = clamp(z / 26)
        face = lerpc((15, 22, 31), (20, 42, 62), fogk); edge = lerpc((34, 50, 66), (28, 54, 78), fogk)
        lz0, lz1 = z, z + 0.6
        d.polygon([P(-0.08, 1.29, lz0), P(0.08, 1.29, lz0), P(0.08, 1.29, lz0 + 0.35), P(-0.08, 1.29, lz0 + 0.35)],
                  fill=lerpc((190, 225, 250), (60, 110, 150), fogk))
        for s in (-1, 1):
            X = s * 1.2
            d.polygon([P(X, -1, lz0), P(X, 1.1, lz0), P(X, 1.1, lz1), P(X, -1, lz1)], fill=face, outline=edge)
            for r_ in range(7):
                Y = -0.78 + r_ * 0.27
                for c_ in range(3):
                    Z = lz0 + 0.12 + c_ * 0.17
                    h = (idx * 131 + r_ * 17 + c_ * 7 + (s > 0) * 53 + int(t * 7 + (idx + r_) * 0.37)) % 11
                    if h < 5:
                        x, y = P(X - s * 0.002, Y, Z); rr = max(1.0, f * 0.011 / Z)
                        col = (125, 211, 252) if h < 4 else (230, 245, 255)
                        d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=lerpc(col, (40, 80, 110), fogk * 0.7))
    im.alpha_composite(glow(W, H, vx, vy, 560, (90, 160, 210), 0.42, 1.5))
    im.alpha_composite(FOG)
    shafts = SHAFTS.copy(); sa = 0.75 + 0.25 * math.sin(t * 0.9)
    shafts.putalpha(shafts.getchannel('A').point(lambda v, k=sa: int(v * k)))
    im.alpha_composite(shafts)
    return im

# ---------- computer cutaway and car (what an operating system is)
def computer_scene(t, t0):
    img = vgrad(W, H, (8, 14, 24), (2, 4, 8)); img.alpha_composite(glow(W, H, 540, 620, 640, (70, 140, 200), 0.25, 1.6))
    lay = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    e = clamp((t - t0) / 0.6)
    a = int(255 * e)
    d.rounded_rectangle([120, 560, 600, 920], radius=18, fill=(20, 40, 60, int(120 * e)), outline=MINT + (a,), width=5)
    d.rounded_rectangle([150, 590, 570, 890], radius=10, fill=(30, 70, 100, int(150 * e)))
    d.rectangle([320, 920, 400, 980], outline=MINT + (a,), width=4)
    d.rounded_rectangle([140, 1010, 580, 1060], radius=10, outline=MINT + (a,), width=4)
    d.rounded_rectangle([660, 560, 940, 1060], radius=16, fill=(20, 40, 60, int(120 * e)), outline=MINT + (a,), width=5)
    core = (730, 720, 870, 860)
    pul = 0.6 + 0.4 * math.sin(t * 5)
    lay.alpha_composite(glow(W, H, 800, 790, 230, (90, 190, 250), 0.55 * e * pul, 1.4))
    d = ImageDraw.Draw(lay)
    d.rounded_rectangle(core, radius=14, fill=(56, 189, 248, a), outline=WHITE + (a,), width=4)
    d.text((800, 790), "OS", font=F('x', 56), fill=INK + (a,), anchor='mm')
    paths = [[(730, 790), (620, 790), (620, 740), (570, 740)], [(800, 860), (800, 1035), (580, 1035)], [(800, 720), (800, 600)]]
    for pi_, pts in enumerate(paths):
        d.line(pts, fill=(125, 211, 252, int(150 * e)), width=3)
        segs = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
        L = sum(math.dist(p, q) for p, q in segs)
        for k in range(3):
            s = ((t - t0) * 0.7 + k / 3 + pi_ * 0.2) % 1 * L
            for p, q in segs:
                l = math.dist(p, q)
                if s <= l:
                    x = p[0] + (q[0] - p[0]) * s / l; y = p[1] + (q[1] - p[1]) * s / l
                    d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=WHITE + (a,)); break
                s -= l
    img.alpha_composite(lay)
    img.alpha_composite(vgrad(W, 220, (10, 18, 28), (3, 5, 8)), (0, 1100))
    refl = lay.crop((0, 560, W, 1100)).transpose(Image.FLIP_TOP_BOTTOM)
    refl.putalpha(refl.getchannel('A').point(lambda v: int(v * 0.16)))
    img.alpha_composite(refl.crop((0, 0, W, 220)), (0, 1100))
    return img

def build_car():
    S = (960, 420); m = Image.new('L', S, 0); d = ImageDraw.Draw(m)
    d.polygon([(30, 300), (40, 240), (150, 200), (300, 112), (540, 100), (690, 186), (880, 214), (930, 255), (935, 305)], fill=255)
    d.rounded_rectangle([30, 250, 935, 330], radius=30, fill=255)
    for cx_ in (220, 740):
        d.ellipse([cx_ - 82, 248, cx_ + 82, 412], fill=255)
    m = m.filter(ImageFilter.GaussianBlur(1.6))
    car = rimlit(m, fill=(8, 12, 18), width=9, strength=1.6)
    d = ImageDraw.Draw(car)
    d.polygon([(320, 128), (530, 118), (640, 190), (300, 196)], fill=(18, 34, 52, 255))
    d.line([(470, 120), (470, 194)], fill=(8, 12, 18, 255), width=8)
    for cx_ in (220, 740):
        d.ellipse([cx_ - 52, 278, cx_ + 52, 382], fill=(16, 22, 30, 255), outline=(60, 90, 120, 255), width=4)
    return car
CAR = build_car()
def car_scene(t, t0):
    img = vgrad(W, H, (8, 14, 24), (2, 4, 8)); img.alpha_composite(glow(W, H, 540, 700, 700, (60, 120, 180), 0.22, 1.6))
    x0, y0 = 60 + int(-12 * ease_span(t, t0, t0 + 3)), 640
    beam = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(beam).polygon([(x0 + 930, y0 + 250), (W + 300, y0 + 120), (W + 300, y0 + 420)], fill=(170, 220, 250, 40))
    img.alpha_composite(beam.filter(ImageFilter.GaussianBlur(18)))
    img.alpha_composite(vgrad(W, 600, (14, 24, 36), (3, 5, 8)), (0, y0 + 412))
    refl = CAR.transpose(Image.FLIP_TOP_BOTTOM); refl.putalpha(refl.getchannel('A').point(lambda v: int(v * 0.14)))
    img.alpha_composite(refl, (x0, y0 + 412))
    img.alpha_composite(CAR, (x0, y0))
    e = clamp((t - t0) / 0.5); pul = 0.7 + 0.3 * math.sin(t * 6)
    img.alpha_composite(glow(W, H, x0 + 800, y0 + 225, 210, (100, 200, 255), 0.75 * e * pul, 1.3))
    put(img, icon(ic_gear, 120, t * 120, WHITE + (255,)), x0 + 800, y0 + 225, e)
    ImageDraw.Draw(img).ellipse([x0 + 912, y0 + 236, x0 + 940, y0 + 262], fill=(230, 245, 255, 255))
    return img

# ---------- time jump, code history, calendar, terminal, dictionary
def tunnel(t, t0, cy=760):
    img = vgrad(W, H, (4, 8, 14), (1, 2, 4)); lay = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    rr = np.random.default_rng(9)
    for i in range(110):
        a = rr.uniform(0, 2 * math.pi); sp = rr.uniform(500, 1400); ph = rr.uniform(0, 1)
        r0 = ((t - t0) * sp / 900 + ph) % 1
        r_in, r_out = 60 + r0 * 1100, 60 + r0 * 1100 + 60 + r0 * 240
        al = int(220 * r0)
        d.line([(540 + math.cos(a) * r_in, cy + math.sin(a) * r_in), (540 + math.cos(a) * r_out, cy + math.sin(a) * r_out)],
               fill=(150, 210, 245, al), width=max(1, int(1 + 3 * r0)))
    img.alpha_composite(lay.filter(ImageFilter.GaussianBlur(1.2)))
    img.alpha_composite(glow(W, H, 540, cy, 420, (70, 140, 200), 0.35, 1.6))
    return img

NODES = 40
_br = np.random.default_rng(4)
NODE_BRANCH = [(_br.random() < 0.22) for _ in range(NODES)]
NODE_FALL = [_br.uniform(0.6, 1.4) for _ in range(NODES)]
def history_scene(t, t0, tbreak):
    img = vgrad(W, H, (6, 12, 20), (2, 4, 8)); lay = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    scroll = (t - t0) * 120; cy = 760
    br = clamp((t - tbreak) / 1.6) if t >= tbreak else 0.0
    pos = []
    for i in range(NODES):
        x = 120 + i * 170 - scroll
        fall = 0.0
        if t >= tbreak:
            tt = max(0.0, t - tbreak - (i % 7) * 0.05); fall = 900 * tt * tt * NODE_FALL[i]
        pos.append((x, cy + fall, (cy - 150 if NODE_BRANCH[i] else cy) + fall))
    col = lerpc(MINT, (60, 74, 90), br)
    al = int(255 * (1 - 0.7 * br))
    for i in range(NODES - 1):
        x0, y0, b0 = pos[i]; x1, y1, b1 = pos[i + 1]
        if x1 < -100 or x0 > W + 100: continue
        d.line([(x0, y0), (x1, y1)], fill=col + (al,), width=6)
        if NODE_BRANCH[i]:
            d.line([(x0, y0), (x0 + 85, b0)], fill=col + (al,), width=5); d.line([(x0 + 85, b0), (x1, y1)], fill=col + (al,), width=5)
            d.ellipse([x0 + 85 - 16, b0 - 16, x0 + 85 + 16, b0 + 16], fill=INK + (al,), outline=col + (al,), width=5)
    for x, y, b in pos:
        if -60 < x < W + 60:
            d.ellipse([x - 22, y - 22, x + 22, y + 22], fill=INK + (al,), outline=col + (al,), width=6)
    img.alpha_composite(lay)
    if 0 <= t - tbreak < 0.5:
        arr = np.asarray(img).copy(); rr = np.random.default_rng(int(t * 60))
        for k in range(7):
            y = int(rr.uniform(300, 1200)); hgt = int(rr.uniform(10, 50)); sh = int(rr.uniform(-40, 40))
            arr[y:y + hgt] = np.roll(arr[y:y + hgt], sh, axis=1)
        img = Image.fromarray(arr, 'RGBA')
    return img

CODE_ROWS = []
_cr = np.random.default_rng(8)
for i in range(80):
    ind = int(_cr.integers(0, 4)); segs = []; x = 24 + ind * 34
    for k in range(int(_cr.integers(1, 5))):
        w_ = int(_cr.integers(30, 150)); segs.append((x, w_, int(_cr.integers(0, 4)))); x += w_ + 14
    CODE_ROWS.append(segs)
def lcd_code(w, h, t, speed=260):
    im = vgrad(w, h, (6, 16, 26), (3, 9, 16)); d = ImageDraw.Draw(im)
    lh = 26; off = (t * speed) % (lh * len(CODE_ROWS))
    cols = [MINT, VIO, (90, 130, 160), WHITE]
    for k in range(h // lh + 2):
        idx = int((off // lh) + k) % len(CODE_ROWS); y = k * lh - (off % lh) + 10
        for x, w_, c in CODE_ROWS[idx]:
            if x + w_ < w - 20: d.rounded_rectangle([x, y, x + w_, y + 12], radius=6, fill=cols[c])
    return im

def calendar_marks(img, t, t0):
    cal = (730, 150, 1020, 450); d = ImageDraw.Draw(img)
    cw, chh = (cal[2] - cal[0] - 20) / 7, 46
    for day in range(1, 31):
        k = day + 3; r_, c_ = divmod(k, 7)
        x = cal[0] + 10 + c_ * cw + cw / 2; y = cal[1] + 92 + r_ * chh
        d.text((x, y), str(day), font=F('sb', 22), fill=(170, 190, 210), anchor='mm')
        if 3 <= day <= 7:
            st = t0 + 0.25 + (day - 3) * 0.42
            p = clamp((t - st) / 0.18)
            if p > 0:
                s = 15
                d.line([(x - s, y - s), (x - s + 2 * s * p, y - s + 2 * s * p)], fill=MINT, width=4)
                if p >= 1: d.line([(x + s, y - s), (x - s, y + s)], fill=MINT, width=4)

def room05(t, t0, z=1.0):
    img = ROOM05.copy()
    w, h = SCR05[2] - SCR05[0], SCR05[3] - SCR05[1]
    img.paste(lcd_code(w, h, t), (SCR05[0], SCR05[1]))
    calendar_marks(img, t, t0)
    dy = 3.5 * math.sin(t * 22) + 2 * math.sin(t * 13); dx = 2 * math.sin(t * 6)
    img.alpha_composite(PERSON, (int(540 - 410 - 40 + dx), int(880 + dy)))
    return cam(img, z, 540, 760)

def terminal(t, ttype, tdict):
    img = Image.new('RGBA', (W, H), (4, 6, 9, 255)); d = ImageDraw.Draw(img)
    d.rounded_rectangle([40, 250, 1040, 1250], radius=24, fill=(20, 24, 30))
    scr = vgrad(960, 960, (6, 16, 26), (3, 9, 16)); sd = ImageDraw.Draw(scr)
    up = 200 * ease_span(t, tdict, tdict + 0.5)
    y = 300 - up
    sd.text((50, y), "$", font=FM(140), fill=(90, 130, 160))
    n = int(clamp((t - ttype) / 0.45) * 3)
    sd.text((170, y), "git"[:n], font=FM(140), fill=MINT)
    cx_ = 170 + FM(140).getlength("git"[:n])
    if int(t * 2.5) % 2 == 0: sd.rectangle([cx_ + 8, y + 20, cx_ + 70, y + 150], fill=MINT)
    img.paste(scr, (60, 270))
    img.alpha_composite(glow(W, H, 540, 700, 600, (90, 160, 220), 0.16, 1.6))
    if t >= tdict:
        cd = card(900, 400, 'glass', r=36).copy(); cdd = ImageDraw.Draw(cd)
        cdd.text((56, 40), "git", font=F('x', 110), fill=WHITE)
        cdd.text((60, 185), "noun  ·  British slang", font=F('m', 36), fill=MUTE)
        cdd.text((60, 255), "an annoying person", font=F('x', 58), fill=MINT)
        e = eo((t - tdict) / 0.5)
        put(img, cd, 540, 900 + 160 * (1 - e), e)
    return img

def name_morph(t, t0, tflip, tgit, tpill):
    img = vgrad(W, H, (6, 12, 20), (2, 4, 8)); img.alpha_composite(glow(W, H, 540, 640, 560, (70, 140, 200), 0.28, 1.6))
    f = F('x', 210)
    pre = tsp("Linu", f, grad=True)
    s_spr = tsp("s", f, grad=True); x_spr = tsp("x", f, grad=True)
    tw_ = pre.width + max(s_spr.width, x_spr.width)
    x0 = 540 - tw_ / 2; y = 640
    a0 = fin(t, t0, 0.35)
    put(img, pre, x0 + pre.width / 2, y, a0)
    p = clamp((t - tflip) / 0.45)
    lx = x0 + pre.width + s_spr.width / 2
    if p < 0.5:
        sc = 1 - p * 2; spr = s_spr
    else:
        sc = (p - 0.5) * 2; spr = x_spr
    if sc > 0.02:
        sq = spr.resize((spr.width, max(1, int(spr.height * sc))), Image.LANCZOS)
        put(img, sq, lx - 6 + (spr.width - s_spr.width) / 2, y, a0)
    if t >= tgit:
        put(img, tsp("Git", F('x', 150), WHITE), 540, 900, fin(t, tgit, 0.3), pop(t, tgit, 0.35))
    if t >= tpill:
        put(img, pill("Named after himself", F('x', 42), 'white', INK), 540, 1080, fin(t, tpill), pop(t, tpill))
    return img

# ---------- a whole city of programmers
FAC_COLS, FAC_ROWS = 9, 15
_fr = np.random.default_rng(12)
FAC_RANK = _fr.permutation(FAC_COLS * FAC_ROWS)
FAC_VAR = _fr.uniform(0.55, 1.0, FAC_COLS * FAC_ROWS)
FAC_HEAD = _fr.random(FAC_COLS * FAC_ROWS) < 0.5
CENTER_WIN = (FAC_ROWS // 2) * FAC_COLS + FAC_COLS // 2
def facade(t, t0, t1):
    img = Image.new('RGBA', (W, H), (8, 13, 20, 255)); d = ImageDraw.Draw(img)
    cw, chh = W / FAC_COLS, H / FAC_ROWS
    n = 1 + (FAC_COLS * FAC_ROWS - 1) * ease_span(t, t0 + 0.4, t1)
    for i in range(FAC_COLS * FAC_ROWS):
        r_, c_ = divmod(i, FAC_COLS)
        x0 = c_ * cw + cw * 0.17; y0 = r_ * chh + chh * 0.16; x1 = x0 + cw * 0.66; y1 = y0 + chh * 0.66
        lit = (i == CENTER_WIN) or (FAC_RANK[i] < n)
        if lit:
            v = FAC_VAR[i] * (0.92 + 0.08 * math.sin(t * 3 + i))
            d.rectangle([x0, y0, x1, y1], fill=lerpc((30, 60, 90), (170, 225, 252), v))
            if FAC_HEAD[i] or i == CENTER_WIN:
                hx = (x0 + x1) / 2; hy = y1 - chh * 0.16
                d.ellipse([hx - cw * 0.08, hy - chh * 0.2, hx + cw * 0.08, hy], fill=(10, 16, 26))
                d.rounded_rectangle([hx - cw * 0.2, hy - chh * 0.04, hx + cw * 0.2, y1], radius=8, fill=(10, 16, 26))
        else:
            d.rectangle([x0, y0, x1, y1], fill=(11, 17, 26))
        d.rectangle([x0, y0, x1, y1], outline=(20, 28, 40), width=3)
    r_, c_ = divmod(CENTER_WIN, FAC_COLS)
    fx, fy = c_ * cw + cw / 2, r_ * chh + chh / 2
    z = 1 + 3.6 * (1 - eo((t - t0) / max(0.1, t1 - t0)))
    return cam(img, z, fx, fy)


# ======================= COMPILER EPISODE SHOTS =======================
BUGTXT = "HOW TECH WORKS"

# ---------- modern room with a laptop
def build_room_now():
    im = vgrad(W, H, (10, 15, 23), (4, 6, 10)); d = ImageDraw.Draw(im)
    wx0, wy0, wx1, wy1 = 60, 230, 330, 640
    win = vgrad(wx1 - wx0, wy1 - wy0, (12, 26, 46), (6, 12, 22)); wd = ImageDraw.Draw(win)
    rr = np.random.default_rng(21)
    for i in range(9):
        bx = int(rr.integers(-20, 250)); bh = int(rr.integers(90, 260)); bw = int(rr.integers(40, 90))
        wd.rectangle([bx, 410 - bh, bx + bw, 410], fill=(8, 14, 24, 255))
        for k in range(10):
            wx_ = bx + int(rr.integers(4, max(5, bw - 10))); wy_ = 410 - int(rr.integers(12, bh))
            if rr.random() < 0.5: wd.rectangle([wx_, wy_, wx_ + 6, wy_ + 8], fill=(90, 150, 200, 255))
    im.paste(win, (wx0, wy0))
    d.rectangle([wx0 - 14, wy0 - 14, wx1 + 14, wy1 + 14], outline=(24, 32, 44), width=14)
    d.line([((wx0 + wx1) // 2, wy0), ((wx0 + wx1) // 2, wy1)], fill=(24, 32, 44), width=10)
    # plant on the shelf (right)
    d.rectangle([800, 760, 1050, 780], fill=(17, 23, 31))
    d.polygon([(890, 760), (960, 760), (950, 680), (900, 680)], fill=(14, 20, 28))
    for a in range(-60, 70, 18):
        ang = math.radians(a - 90); L = 120 + (a % 3) * 25
        d.line([(925, 685), (925 + math.cos(ang) * L, 685 + math.sin(ang) * L)], fill=(16, 26, 36), width=12)
    d.rectangle([0, 1000, W, 1080], fill=(16, 21, 28)); d.rectangle([0, 1080, W, H], fill=(7, 9, 13))
    d.rounded_rectangle([230, 540, 850, 930], radius=16, fill=(22, 26, 32))
    scr = (250, 560, 830, 910)
    d.polygon([(200, 930), (880, 930), (930, 975), (150, 975)], fill=(24, 29, 36))
    im.alpha_composite(glow(W, H, 540, 735, 640, (90, 160, 220), 0.32, 1.8))
    im.alpha_composite(glow(W, H, 540, 1000, 540, (110, 180, 235), 0.22, 2.0))
    return im, scr
ROOMN, SCRN = build_room_now()

CODE = ["if hungry:", "    order_pizza()", "else:", "    keep_coding()", "", 'print("Hello!")']
def code_screen(w, h, font, lh, x0, y0, chars, t, qmarks=0.0):
    im = vgrad(w, h, (8, 20, 32), (4, 11, 19)); d = ImageDraw.Draw(im)
    left = chars; cur = None
    cols = [VIO, MINT, VIO, MINT, MINT, WHITE]
    for i, line in enumerate(CODE):
        shown = line[:max(0, int(left))]; left -= len(line) + 1
        y = y0 + i * lh
        d.text((x0, y), shown, font=font, fill=cols[i])
        if left <= 0 and cur is None: cur = (x0 + font.getlength(shown), y)
    if cur and int(t * 2.5) % 2 == 0:
        d.rectangle([cur[0] + 3, cur[1] + 4, cur[0] + font.size * 0.55, cur[1] + lh - 8], fill=MINT)
    if qmarks > 0:
        rr = np.random.default_rng(3)
        for k in range(14):
            x, y = rr.uniform(0.1, 0.9) * w, rr.uniform(0.1, 0.9) * h
            yy = y - 40 * qmarks * rr.uniform(0.5, 1.5)
            s = int(font.size * rr.uniform(1.2, 2.2))
            d.text((x, yy), "?", font=F('x', s), fill=WHITE + (int(220 * min(1, qmarks * 1.5)),), anchor='mm')
    return im
CODE_TOTAL = sum(len(l) + 1 for l in CODE)

def room_now(t, chars, z=1.0, fx=540, fy=880, typing=1.0):
    img = ROOMN.copy()
    w, h = SCRN[2] - SCRN[0], SCRN[3] - SCRN[1]
    img.paste(code_screen(w, h, FM(28), 46, 26, 26, chars, t), (SCRN[0], SCRN[1]))
    dy = typing * (3 * math.sin(t * 15) + 2 * math.sin(t * 9.3)); dx = typing * 2 * math.sin(t * 4.1)
    img.alpha_composite(PERSON, (int(540 - 410 - 40 + dx), int(900 + dy)))
    return cam(img, z, fx, fy)

def laptop_close(t, chars, qmarks=0.0, z=1.0):
    img = Image.new('RGBA', (W, H), (4, 6, 9, 255)); d = ImageDraw.Draw(img)
    d.rounded_rectangle([30, 300, 1050, 1200], radius=34, fill=(22, 26, 32))
    sw, sh = 960, 840
    img.paste(code_screen(sw, sh, FM(62), 104, 50, 60, chars, t, qmarks), (60, 330))
    img.alpha_composite(glow(W, H, 540, 750, 620, (90, 160, 220), 0.16, 1.6))
    return cam(img, z, 540, 750)

# ---------- streams of 1s and 0s (horizontal, light blue)
_br2 = np.random.default_rng(31)
def _binrow(n): return ''.join('01'[int(v)] for v in _br2.integers(0, 2, n))
ROWS = []
for i in range(44):
    depth = _br2.uniform(0.25, 1.0)
    f = FM(int(18 + 22 * depth))
    txt = _binrow(140)
    col = lerpc((40, 80, 120), (170, 225, 252), depth)
    sp = Image.new('RGBA', (int(f.getlength(txt)) + 10, int(f.size * 1.3)), (0, 0, 0, 0))
    ImageDraw.Draw(sp).text((0, 0), txt, font=f, fill=col + (255,))
    if depth < 0.5: sp = sp.filter(ImageFilter.GaussianBlur(1.6))
    ROWS.append((sp, depth, _br2.uniform(60, 260) * (0.5 + depth), _br2.uniform(0, 2000)))
def binary_void(t, t0, chip=1.0, cy=760):
    img = vgrad(W, H, (4, 9, 16), (1, 3, 6))
    for i, (sp, depth, speed, ph) in enumerate(ROWS):
        y = 140 + i * 40
        off = (ph + (t - t0) * speed) % (sp.width - W)
        a = 0.35 + 0.65 * depth
        if 1210 < y < 1460: a *= 0.25
        crop = sp.crop((int(sp.width - W - off), 0, int(sp.width - off), sp.height))
        if a < 0.99:
            crop = crop.copy(); crop.putalpha(crop.getchannel('A').point(lambda v, al=a: int(v * al)))
        img.alpha_composite(crop, (0, int(y)))
    if chip > 0.01:
        img.alpha_composite(glow(W, H, 540, cy, 360, (90, 180, 240), 0.55 * chip, 1.5))
        g = 0.5 + 0.5 * math.sin(t * 5)
        put(img, icon(ic_chip, 400, g), 540, cy, chip, 0.8 + 0.2 * chip)
    return img

def binary_wall(t, t0, t1):
    img = vgrad(W, H, (4, 9, 16), (1, 3, 6))
    n = int(4 + 34 * ease_span(t, t0, t1 + 0.6))
    for k in range(n):
        sp, depth, speed, ph = ROWS[k % len(ROWS)]
        y = 1180 - k * 34
        if y < 120: break
        crop = sp.crop((int(ph % 600), 0, int(ph % 600) + 900, sp.height))
        put(img, crop, 540, y, 0.55 + 0.45 * (1 - k / 40))
    small = PERSON.resize((300, 300), Image.LANCZOS)
    img.alpha_composite(glow(W, H, 540, 1150, 380, (90, 160, 220), 0.3, 1.6))
    img.alpha_composite(small, (540 - 150, 1180))
    # clock
    cx, cy, r = 880, 330, 70
    d = ImageDraw.Draw(img)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=MINT, width=6)
    for k, (L, sp_) in enumerate(((0.75, 6.0), (0.5, 0.7))):
        ang = (t - t0) * sp_ * 2 * math.pi - math.pi / 2
        d.line([(cx, cy), (cx + math.cos(ang) * r * L, cy + math.sin(ang) * r * L)], fill=WHITE, width=6 - k)
    return img

# ---------- the compiler machine
WORDS = ["print", "hello", "if", "repeat", "add", "save", "open", "show"]
def compiler_machine(t, t0, tm):
    img = vgrad(W, H, (8, 14, 24), (2, 4, 8)); img.alpha_composite(glow(W, H, 540, 640, 680, (70, 140, 200), 0.22, 1.6))
    d = ImageDraw.Draw(img)
    by = 1010
    d.rectangle([0, by, W, by + 22], fill=(22, 30, 40))
    off = ((t - t0) * 160) % 60
    for x in range(-60, W + 60, 60): d.rectangle([x + off, by + 8, x + off + 30, by + 14], fill=(40, 60, 80))
    img.alpha_composite(vgrad(W, 300, (12, 20, 30), (3, 5, 8)), (0, by + 22))
    m = ease_span(t, tm - 0.3, tm + 0.4)
    mx0, mx1 = 360, 720
    for i in range(12):
        x = -160 + ((t - t0) * 160 + i * 150) % 1400
        if x > mx0 - 40 and m > 0.5: continue
        word = WORDS[i % len(WORDS)]
        put(img, pill(word, F('x', 40), 'grad', INK), x, by - 40, 1.0)
    if m > 0.01:
        for i in range(12):
            x = mx1 + ((t - t0) * 160 + i * 110) % 600
            put(img, pill(_binrow(4), FM(34), 'white', INK), x + 60, by - 40, m * clamp(1 - (x - 900) / 300))
        lay = Image.new('RGBA', (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay)
        a = int(255 * m)
        ld.rounded_rectangle([mx0, 560, mx1, by - 4], radius=30, fill=(16, 34, 52, int(230 * m)), outline=MINT + (a,), width=6)
        ld.rectangle([mx0 - 10, by - 110, mx0 + 20, by - 4], fill=(10, 18, 28, a)); ld.rectangle([mx1 - 20, by - 110, mx1 + 10, by - 4], fill=(10, 18, 28, a))
        img.alpha_composite(lay)
        img.alpha_composite(glow(W, H, 540, 760, 260, (90, 190, 250), 0.6 * m * (0.75 + 0.25 * math.sin(t * 6)), 1.4))
        put(img, icon(ic_gear, 150, t * 140, WHITE + (255,)), 540, 740, m)
        put(img, tsp("COMPILER", F('x', 44), WHITE), 540, 900, m)
    return img

# ---------- translating a book
def book_scene(t, t0):
    img = vgrad(W, H, (8, 14, 24), (2, 4, 8)); img.alpha_composite(glow(W, H, 540, 760, 640, (80, 150, 210), 0.28, 1.6))
    lay = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    d.polygon([(100, 560), (535, 600), (535, 1080), (100, 1040)], fill=(176, 202, 222, 255))
    d.polygon([(545, 600), (980, 560), (980, 1040), (545, 1080)], fill=(186, 210, 228, 255))
    d.rectangle([533, 596, 547, 1082], fill=(120, 150, 175, 255))
    sweep = ease_span(t, t0 + 0.3, t0 + 2.6)
    for side, (xa, xb, ya, yb) in enumerate(((130, 505, 605, 640), (575, 950, 640, 605))):
        for k in range(15):
            fx = 0.15 + 0.85 * ((k * 37) % 10) / 10
            x0 = xa; x1 = xa + (xb - xa) * (0.55 + 0.45 * fx)
            y0 = ya + (yb - ya) * 0.0 + 30 * k; y1 = yb + 30 * k
            ym = (y0 + y1) / 2
            done = ((side * 0.5 + (k / 15) * 0.5) < sweep)
            if done:
                d.text((x0, ym - 13), _binrow(int((x1 - x0) / 15)), font=FM(22), fill=(30, 100, 150, 255))
            else:
                d.line([(x0, y0 + 1), (x1, y0 + 1 + (y1 - y0) * (x1 - x0) / (xb - xa))], fill=(55, 75, 95, 255), width=8)
    img.alpha_composite(lay)
    sx = 100 + 880 * sweep
    if 0 < sweep < 1:
        img.alpha_composite(glow(W, H, sx, 820, 240, (150, 220, 255), 0.6, 1.4))
    return img

# ---------- 1952 machine room
def woman_mask(S):
    m = Image.new('L', (S, S), 0); d = ImageDraw.Draw(m); c = S / 2
    d.rounded_rectangle([c - 0.40 * S, 0.52 * S, c + 0.40 * S, 1.3 * S], radius=int(0.2 * S), fill=255)
    d.polygon([(c - 0.19 * S, 0.55 * S), (c + 0.19 * S, 0.55 * S), (c + 0.075 * S, 0.40 * S), (c - 0.075 * S, 0.40 * S)], fill=255)
    d.ellipse([c - 0.115 * S, 0.17 * S, c + 0.115 * S, 0.45 * S], fill=255)
    d.ellipse([c - 0.15 * S, 0.14 * S, c + 0.15 * S, 0.34 * S], fill=255)      # full, rounded hairstyle
    d.ellipse([c - 0.17 * S, 0.24 * S, c - 0.08 * S, 0.40 * S], fill=255)
    d.ellipse([c + 0.08 * S, 0.24 * S, c + 0.17 * S, 0.40 * S], fill=255)
    return m.filter(ImageFilter.GaussianBlur(1.5))
WOMAN = rimlit(woman_mask(760))

def build_room52():
    im = vgrad(W, H, (9, 14, 21), (3, 5, 8)); d = ImageDraw.Draw(im)
    lamps = []
    for x0 in (20, 260, 820):
        d.rectangle([x0, 260, x0 + 220, 1020], fill=(18, 24, 32)); d.rectangle([x0, 260, x0 + 220, 280], fill=(26, 34, 44))
        for r_ in range(9):
            for c_ in range(5):
                lamps.append((x0 + 30 + c_ * 40, 330 + r_ * 44))
    reels = []
    d.rectangle([520, 300, 780, 1020], fill=(16, 22, 30))
    for ry in (440, 700):
        for rx in (590, 710):
            d.ellipse([rx - 52, ry - 52, rx + 52, ry + 52], fill=(10, 14, 20), outline=(60, 80, 100), width=5)
            reels.append((rx, ry))
    d.rectangle([0, 1020, W, H], fill=(7, 9, 13))
    d.rectangle([150, 980, 930, 1080], fill=(20, 26, 34))
    im.alpha_composite(glow(W, H, 540, 700, 700, (80, 140, 200), 0.18, 1.8))
    return im, lamps, reels
ROOM52, LAMPS52, REELS52 = build_room52()
def room52(t, t0, on=0.2, spin=1.0, z=1.0):
    img = ROOM52.copy(); d = ImageDraw.Draw(img)
    for i, (x, y) in enumerate(LAMPS52):
        lit = ((i * 37 + int(t * 5 + i * 0.13)) % 10) < 10 * on
        d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=(170, 225, 252) if lit else (30, 44, 58))
    for k, (rx, ry) in enumerate(REELS52):
        ang = (t - t0) * spin * (2.2 if k % 2 else -2.0)
        for s in range(3):
            a = ang + s * 2 * math.pi / 3
            d.line([(rx, ry), (rx + math.cos(a) * 44, ry + math.sin(a) * 44)], fill=(90, 120, 150), width=6)
        d.ellipse([rx - 10, ry - 10, rx + 10, ry + 10], fill=(120, 160, 190))
    img.alpha_composite(WOMAN, (540 - 380, 1010))
    return cam(img, z, 540, 900)

def arithmetic_panel(t, t0):
    img = vgrad(W, H, (6, 10, 16), (2, 3, 6)); d = ImageDraw.Draw(img)
    d.rounded_rectangle([90, 520, 990, 1000], radius=30, fill=(18, 24, 32), outline=(40, 56, 72), width=6)
    txt = "2 + 2 = 4"
    n = int(clamp((t - t0) / 0.9) * len(txt))
    f = F('x', 150)
    x = 540 - f.getlength(txt) / 2
    lay = Image.new('RGBA', (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay)
    ld.text((x, 760), txt[:n], font=f, fill=(170, 225, 252, 255), anchor='lm')
    img.alpha_composite(lay.filter(ImageFilter.GaussianBlur(6))); img.alpha_composite(lay)
    return img

# ---------- AI EPISODE SHOTS ----------
BUGTXT = "HOW TECH WORKS"
QUESTION = "Is my PHP updated?"
ANSWER = ["Sun", "light", " bounces", " off", " tiny", " air", " bits", ",", " and", " blue", " light", " scatters", " the", " most", "."]

def bubble(text, font, fill, tc, pad=28, r=34, maxw=760):
    tw_ = min(maxw, font.getlength(text)); h = int(font.size * 1.6 + pad)
    im = Image.new('RGBA', (int(tw_ + pad * 2), h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, im.width - 1, h - 1], radius=r, fill=fill)
    d.text((pad, h / 2), text, font=font, fill=tc, anchor='lm')
    return im

def chat_close(t, qchars, achars, dots=False, z=1.0):
    img = Image.new('RGBA', (W, H), (4, 6, 9, 255)); d = ImageDraw.Draw(img)
    d.rounded_rectangle([30, 250, 1050, 1250], radius=34, fill=(22, 26, 32))
    scr = vgrad(960, 940, (8, 18, 30), (4, 10, 18)); sd = ImageDraw.Draw(scr)
    sd.rounded_rectangle([40, 820, 920, 900], radius=40, fill=(18, 32, 48))
    sd.text((80, 860), "Ask anything", font=F('m', 34), fill=(90, 120, 150), anchor='lm')
    img.paste(scr, (60, 280))
    f = F('sb', 46)
    q = QUESTION[:int(qchars)]
    if q:
        b = bubble(q, f, VIO + (255,), INK)
        img.alpha_composite(b, (990 - b.width, 360))
    if dots or achars > 0:
        if achars <= 0:
            b = Image.new('RGBA', (170, 90), (0, 0, 0, 0)); bd = ImageDraw.Draw(b)
            bd.rounded_rectangle([0, 0, 169, 89], radius=34, fill=(30, 46, 64, 255))
            for k in range(3):
                a = 0.4 + 0.6 * max(0, math.sin(t * 7 - k * 0.8))
                bd.ellipse([38 + k * 36 - 11, 34, 38 + k * 36 + 11, 56], fill=lerpc((60, 80, 100), MINT, a))
            img.alpha_composite(b, (90, 500))
        else:
            n = int(achars); txt = ''.join(ANSWER[:n])
            words = txt.split(' '); lines, cur = [], ''
            for w_ in words:
                if f.getlength((cur + ' ' + w_).strip()) > 760 and cur: lines.append(cur); cur = w_
                else: cur = (cur + ' ' + w_).strip()
            lines.append(cur)
            hb = int(len(lines) * 64 + 48)
            b = Image.new('RGBA', (840, hb), (0, 0, 0, 0)); bd = ImageDraw.Draw(b)
            bd.rounded_rectangle([0, 0, 839, hb - 1], radius=34, fill=(30, 46, 64, 255))
            for i, ln in enumerate(lines): bd.text((30, 24 + i * 64), ln, font=f, fill=WHITE)
            img.alpha_composite(b, (90, 500))
    img.alpha_composite(glow(W, H, 540, 750, 620, (90, 160, 220), 0.14, 1.6))
    return cam(img, z, 540, 760)

# ===== EPISODE START (replace this whole block for a new reel; keep bug(), scA..scH and SCENEFN, HITS) =====
# ---------- Neural networks episode: "How a neural network learns"
# Everything drawn from scratch: an original stylised cat, no logos. Light blue + black only.
from scipy.spatial import cKDTree

_DOTS = {}
def dot(r, col=MINT, blur=0):
    k = (int(r), tuple(int(c) for c in col), int(blur))
    r, col, blur = k
    if k not in _DOTS:
        S = int(2 * r + 4 * blur + 6); im = Image.new('RGBA', (S, S), (0, 0, 0, 0))
        ImageDraw.Draw(im).ellipse([S / 2 - r, S / 2 - r, S / 2 + r, S / 2 + r], fill=col + (255,))
        if blur > 0: im = im.filter(ImageFilter.GaussianBlur(blur))
        _DOTS[k] = im
    return _DOTS[k]

def bug(fr, t):
    if 0.8 < t < END_CARD:
        a = fin(t, 0.8, 0.6) * (1 - eio((t - (END_CARD - 0.4)) / 0.4))
        put(fr, tsp(BUGTXT, F('sb', 26), MUTE), 160, 228, a)

def bg_dark(cy=820, k=0.22):
    img = vgrad(W, H, (6, 11, 19), (1, 3, 6)); img.alpha_composite(glow(W, H, 540, cy, 700, (60, 130, 190), k, 1.5))
    return img
BG_NN = bg_dark()

# ---------- 3D brain made of neurons (hook)
_bb = np.random.default_rng(86)
def _in_brain(x, y, z, k=1.0):
    cer = (z / (0.95 * k)) ** 2 + ((y - 0.1) / (0.62 * k)) ** 2 + (x / (0.68 * k)) ** 2 < 1 and y > -0.38 + 0.18 * z
    cbl = (x / (0.45 * k)) ** 2 + ((y + 0.42) / (0.2 * k)) ** 2 + ((z + 0.62) / (0.3 * k)) ** 2 < 1
    stem = (x / (0.12 * k)) ** 2 + ((y + 0.62) / (0.3 * k)) ** 2 + ((z + 0.28) / (0.12 * k)) ** 2 < 1
    return cer or cbl or stem
_pts = []
while len(_pts) < 620:
    x, y, z = _bb.uniform(-1, 1, 3)
    if abs(x) < 0.035 or not _in_brain(x, y, z): continue
    if _in_brain(x, y, z, 0.8) and _bb.random() < 0.8: continue
    _pts.append((x, y, z))
BRAIN = np.array(_pts)
BPAIRS = np.array(sorted(cKDTree(BRAIN).query_pairs(0.135)))
WAVES = [(0.2, 17), (1.1, 211), (2.0, 340), (2.7, 90), (3.5, 400), (4.4, 150), (5.2, 30), (6.0, 260)]
NUMSTR = [f"{v:.2f}" for v in _bb.uniform(-0.99, 0.99, 60)]

def project(P, yaw, pitch, R, cx, cy):
    c, s = math.cos(yaw), math.sin(yaw)
    X = P[:, 0] * c - P[:, 2] * s; Z = P[:, 0] * s + P[:, 2] * c; Y = P[:, 1]
    cp, sp = math.cos(pitch), math.sin(pitch)
    Y2 = Y * cp - Z * sp; Z2 = Y * sp + Z * cp
    k = 1 / (1 + 0.38 * Z2)
    return cx + X * R * k, cy - Y2 * R * k, Z2, k

def brain_scene(t, t0, tnum, R=470, cy=820):
    img = BG_NN.copy()
    yaw = math.pi / 2 + 0.35 * math.sin(0.25 * (t - t0)) + 0.05 * (t - t0); pitch = 0.12
    px, py, pz, pk = project(BRAIN, yaw, pitch, R, 540, cy)
    act = np.zeros(len(BRAIN))
    for ts, src in WAVES:
        if t < ts: continue
        d = np.linalg.norm(BRAIN - BRAIN[src], axis=1)
        act += np.exp(-((d - 0.9 * (t - ts)) / 0.07) ** 2) * math.exp(-(t - ts) * 0.6)
    act = np.clip(act, 0, 1)
    lay = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    for i, j in BPAIRS:
        a = (act[i] + act[j]) / 2; depth = (pz[i] + pz[j]) / 2
        al = int((26 + 200 * a) * (0.45 + 0.55 * (1 - (depth + 1) / 2)))
        d.line([(px[i], py[i]), (px[j], py[j])], fill=lerpc((70, 130, 180), (230, 248, 255), a) + (al,), width=2 if a < 0.4 else 3)
    img.alpha_composite(lay)
    order = np.argsort(-pz)
    numa = fin(t, tnum, 0.6)
    for i in order:
        depth = (pz[i] + 1) / 2
        r = 3 + 4 * pk[i] + 5 * act[i]; blur = 2.5 * depth
        col = lerpc((90, 160, 210), (235, 248, 255), act[i])
        put(img, dot(r, col, blur), px[i], py[i], (0.5 + 0.5 * (1 - depth)) * (1 - 0.6 * numa))
        if act[i] > 0.5: put(img, dot(r * 3, MINT, r * 1.5), px[i], py[i], 0.35 * act[i])
    if numa > 0:
        for k in range(len(NUMSTR)):
            i = order[-1 - k * 4] if k * 4 < len(order) else order[-1]
            put(img, tsp(NUMSTR[k], F('sb', 26), (200, 230, 250)), px[i], py[i], numa * (0.5 + 0.5 * (1 - (pz[i] + 1) / 2)))
    return img

# ---------- original stylised cat (photo-like card and a flat icon)
def cat_photo(S):
    im = vgrad(S, S, (40, 80, 115), (10, 22, 36)); d = ImageDraw.Draw(im)
    d.ellipse([S * 0.1, S * 0.62, S * 0.9, S * 1.25], fill=(26, 52, 76))
    hx, hy, hr = S * 0.5, S * 0.52, S * 0.27
    d.polygon([(hx - hr * 0.95, hy - hr * 0.35), (hx - hr * 0.75, hy - hr * 1.35), (hx - hr * 0.2, hy - hr * 0.85)], fill=(120, 170, 205))
    d.polygon([(hx + hr * 0.95, hy - hr * 0.35), (hx + hr * 0.75, hy - hr * 1.35), (hx + hr * 0.2, hy - hr * 0.85)], fill=(120, 170, 205))
    d.ellipse([hx - hr, hy - hr * 0.95, hx + hr, hy + hr * 0.85], fill=(140, 185, 215))
    for sx in (-1, 1):
        ex = hx + sx * hr * 0.4
        d.ellipse([ex - hr * 0.17, hy - hr * 0.3, ex + hr * 0.17, hy + hr * 0.02], fill=(225, 245, 255))
        d.ellipse([ex - hr * 0.06, hy - hr * 0.26, ex + hr * 0.06, hy - hr * 0.02], fill=(8, 16, 24))
        for k in (-1, 0, 1):
            d.line([(hx + sx * hr * 0.3, hy + hr * 0.35), (hx + sx * hr * 1.25, hy + hr * (0.25 + 0.14 * k))], fill=(220, 240, 252), width=max(2, S // 160))
    d.polygon([(hx - hr * 0.1, hy + hr * 0.15), (hx + hr * 0.1, hy + hr * 0.15), (hx, hy + hr * 0.28)], fill=(30, 50, 70))
    im = im.filter(ImageFilter.GaussianBlur(S / 300))
    return im
CATP = cat_photo(600)
CATG = np.asarray(CATP.convert('L').resize((24, 24), Image.BILINEAR)).astype(np.float32) / 255
def photo_card(img_, w=640):
    pad = 20; c = Image.new('RGBA', (w + 2 * pad, w + 2 * pad), (0, 0, 0, 0)); d = ImageDraw.Draw(c)
    d.rounded_rectangle([0, 0, c.width - 1, c.height - 1], radius=28, fill=(18, 28, 40, 255), outline=(150, 200, 235, 255), width=4)
    m = Image.new('L', (w, w), 0); ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, w - 1], radius=18, fill=255)
    c.paste(img_.resize((w, w)), (pad, pad), m)
    return c
CAT_CARD = photo_card(CATP, 420)

# ---------- 3D layered network
LAYERS = [5, 7, 7, 5, 3]
_nr = np.random.default_rng(5)
NODES = []
for l, n in enumerate(LAYERS):
    for i in range(n):
        NODES.append((l, i, ((l - (len(LAYERS) - 1) / 2) * 0.5, (i - (n - 1) / 2) * 0.27, _nr.uniform(-0.12, 0.12))))
NODEP = np.array([p for _, _, p in NODES])
EDGES = []
for a, (la, ia, _) in enumerate(NODES):
    for b, (lb, ib, _) in enumerate(NODES):
        if lb == la + 1: EDGES.append((a, b, la, _nr.uniform(-1, 1)))

def net3d(t, yaw=0.0, cy=800, R=760, fire_t=None, back_t=None, train=0.0, alpha=1.0, dim=0.0):
    img = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    px, py, pz, pk = project(NODEP, yaw, 0.12, R, 540, cy)
    lay = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    nl = len(LAYERS)
    for ei, (a, b, l, w) in enumerate(EDGES):
        if train > 0: w = w + 0.35 * train * math.sin(ei * 1.7 + t * 9) * (1 if back_t is not None and t > back_t + (nl - 2 - l) * 0.3 else 0)
        col = (125, 211, 252) if w > 0 else (60, 95, 130)
        bk = 0.0
        if back_t is not None:
            p = (t - back_t - (nl - 2 - l) * 0.3) / 0.3
            bk = math.exp(-((p - 0.5) / 0.35) ** 2) if -0.5 < p < 1.5 else 0
        al = int((40 + 90 * abs(w) + 120 * bk) * alpha * (1 - dim))
        d.line([(px[a], py[a]), (px[b], py[b])], fill=lerpc(col, (240, 250, 255), bk) + (al,), width=1 + int(2.5 * abs(w)) + int(2 * bk))
    img.alpha_composite(lay)
    pul = Image.new('RGBA', (W, H), (0, 0, 0, 0)); pd = ImageDraw.Draw(pul)
    for ei, (a, b, l, w) in enumerate(EDGES):
        if fire_t is not None and ei % 2 == 0:
            p = (t - fire_t - l * 0.35) / 0.35
            if 0 <= p <= 1:
                x = px[a] + (px[b] - px[a]) * p; y = py[a] + (py[b] - py[a]) * p
                pd.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(235, 248, 255, int(230 * alpha)))
        if back_t is not None and ei % 3 == 0:
            p = 1 - (t - back_t - (nl - 2 - l) * 0.3) / 0.3
            if 0 <= p <= 1:
                x = px[a] + (px[b] - px[a]) * p; y = py[a] + (py[b] - py[a]) * p
                pd.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(255, 255, 255, int(240 * alpha)))
    img.alpha_composite(pul.filter(ImageFilter.GaussianBlur(1.2)))
    img.alpha_composite(pul)
    for idx in np.argsort(-pz):
        l = NODES[idx][0]
        on = 0.0
        if fire_t is not None: on = clamp((t - fire_t - l * 0.35) / 0.2) * (0.6 + 0.4 * math.sin(idx * 2.3) ** 2)
        depth = (pz[idx] + 1) / 2; r = 16 * pk[idx]
        put(img, dot(r * 2.6, MINT, r * 1.4), px[idx], py[idx], (0.15 + 0.5 * on) * alpha)
        put(img, dot(r + 3, (12, 22, 34), 0), px[idx], py[idx], alpha)
        put(img, dot(r, lerpc((70, 130, 180), (235, 248, 255), on), 1.5 * depth), px[idx], py[idx], alpha)
    return img

# ---------- shots
def shot_A(t):
    img = brain_scene(t, 0, wt(0, 'numbers') - 0.4)
    tn = wt(0, '86') - 0.1
    put(img, pill("86 billion neurons", F('x', 52), 'grad', INK), 540, 330, fin(t, tn) * (1 - fin(t, wt(0, 'copied') - 0.2)), pop(t, tn))
    if t > wt(0, 'copied') - 0.2:
        put(img, pill("AI = numbers", F('x', 52), 'white', INK), 540, 330, fin(t, wt(0, 'numbers') - 0.3), pop(t, wt(0, 'numbers') - 0.3))
    return img

def shot_B(t):
    t0 = SCN['B'][0]
    img = BG_NN.copy()
    yaw = -0.35 + 0.12 * (t - t0)
    img.alpha_composite(net3d(t, yaw, 780, R=430, fire_t=t0 + 0.4, alpha=fin(t, t0, 0.5)))
    put(img, pill("Neural network", F('x', 50), 'grad', INK), 540, 330, fin(t, t0 + 0.2), pop(t, t0 + 0.2))
    tc = wt(1, 'cat') - 0.35
    if t > tc: put(img, CAT_CARD, 540, 1180 - 30 * fin(t, tc, 0.5), fin(t, tc, 0.4), 0.55)
    return img

def shot_C(t):
    t0 = SCN['C'][0]; tp = wt(2, 'millions') - 0.1
    img = BG_NN.copy()
    z = 1.0 if t < tp else 1.0 + 7.0 * eio((t - tp) / 2.6)
    pix = clamp((t - tp + 0.3) / 0.8)
    S = 700; cx, cy = 540, 800
    if pix < 0.02:
        put(img, photo_card(CATP, 640), cx, cy, fin(t, t0, 0.3))
        return img
    cell = S / 24 * z
    fx, fy = 7.5, 10.0
    lay = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    for r in range(24):
        for c in range(24):
            x0 = cx + (c - fx) * cell; y0 = cy + (r - fy) * cell
            if x0 > W or y0 > H or x0 + cell < 0 or y0 + cell < 0: continue
            v = CATG[r, c]
            col = lerpc((8, 18, 30), (225, 245, 255), v)
            gap = max(1, int(cell * 0.06 * pix))
            d.rectangle([x0 + gap, y0 + gap, x0 + cell - gap, y0 + cell - gap], fill=col + (255,))
            if cell > 90:
                d.text((x0 + cell / 2, y0 + cell / 2), f"{v:.2f}", font=F('x', int(cell * 0.24)), fill=(INK if v > 0.55 else WHITE) + (int(255 * clamp((cell - 90) / 60)),), anchor='mm')
    img.alpha_composite(lay)
    put(img, pill("pixels = numbers", F('x', 48), 'grad', INK), 540, 330, fin(t, wt(2, 'pixels') - 0.2), pop(t, wt(2, 'pixels') - 0.2))
    return img

def wire(d, a, b, col, w):
    d.line([a, b], fill=col, width=w)
INS = [(0.9, 0.8), (0.2, -0.5), (0.7, 1.2)]
def shot_D(t):
    t0 = SCN['D'][0]
    img = BG_NN.copy(); d = ImageDraw.Draw(img)
    ti, tw, ts, tth, tf = wt(3, 'takes') - 0.2, wt(3, 'multiplies') - 0.1, wt(3, 'adds') - 0.1, LS[4] - 0.1, wt(4, 'fires') - 0.1
    N = (640, 780); ys = [520, 780, 1040]
    fire = clamp((t - tf) / 0.25)
    img.alpha_composite(glow(W, H, N[0], N[1], 300, (120, 200, 250), 0.2 + 0.6 * fire * math.exp(-max(0, t - tf) * 1.5), 1.5))
    d = ImageDraw.Draw(img)
    for k, ((x, w), y) in enumerate(zip(INS, ys)):
        a = fin(t, ti + k * 0.12, 0.3)
        if a <= 0: continue
        wcol = (125, 211, 252) if w > 0 else (80, 120, 160)
        d.line([(230, y), N], fill=wcol + (int(255 * a),), width=int(4 + 4 * abs(w)))
        put(img, dot(62, (14, 26, 40)), 190, y, a)
        put(img, tsp(f"{x:.1f}", F('x', 52), WHITE), 190, y, a)
        if t > tw:
            mx, my = (230 + N[0]) / 2, (y + N[1]) / 2 - 40
            put(img, pill(f"× {w:+.1f}", F('x', 38), 'dark', MINT), mx, my, fin(t, tw + k * 0.15, 0.25), pop(t, tw + k * 0.15))
        if t > tw + 0.3 and k < 3:
            put(img, tsp(f"= {x * w:+.2f}", F('sb', 32), MUTE), 260, y + 70, fin(t, tw + 0.3 + k * 0.15, 0.25))
    put(img, dot(118, lerpc((20, 40, 60), (200, 238, 255), fire)), N[0], N[1], fin(t, t0, 0.3))
    put(img, dot(100, (10, 20, 32)), N[0], N[1], fin(t, t0, 0.3) * (1 - 0.6 * fire))
    tot = sum(x * w for x, w in INS)
    lab = "Σ" if t < ts + 0.4 else f"{tot:.2f}"
    put(img, tsp(lab, F('x', 70 if t < ts + 0.4 else 56), WHITE), N[0], N[1], fin(t, t0 + 0.2))
    if t > tth:
        bx, by = 900, 620; a = fin(t, tth, 0.3); h = 320
        dd = ImageDraw.Draw(img)
        dd.rounded_rectangle([bx - 26, by, bx + 26, by + h], radius=26, outline=(140, 180, 210, int(255 * a)), width=4)
        lvl = clamp(tot / 2.0) * eo((t - tth) / 0.8)
        dd.rounded_rectangle([bx - 18, by + h - h * lvl, bx + 18, by + h - 8], radius=18, fill=MINT + (int(255 * a),))
        ty = by + h - h * 0.5
        dd.line([(bx - 50, ty), (bx + 50, ty)], fill=WHITE + (int(255 * a),), width=5)
        put(img, tsp("threshold", F('sb', 28), MUTE), bx, by + h + 40, a)
    if t > tf:
        p = clamp((t - tf - 0.1) / 0.6)
        dd = ImageDraw.Draw(img)
        dd.line([(N[0] + 118, N[1]), (N[0] + 118 + 300 * p, N[1])], fill=WHITE + (255,), width=8)
        put(img, dot(14, WHITE, 3), N[0] + 118 + 300 * p, N[1], 1 - clamp((p - 0.9) / 0.1) * 0)
        put(img, pill("fires!", F('x', 50), 'white', INK), N[0], N[1] - 200, fin(t, tf), pop(t, tf))
    put(img, pill("1 artificial neuron", F('x', 46), 'grad', INK), 540, 330, fin(t, t0 + 0.2), pop(t, t0 + 0.2))
    return img

def edge_tile(kind, S=150):
    im, d = canvas(S, S)
    d.rounded_rectangle([0, 0, S - 1, S - 1], radius=16, fill=(12, 22, 34, 255), outline=(90, 140, 180, 255), width=3)
    w = 8
    if kind == 0: d.line([(30, S - 30), (S - 30, 30)], fill=MINT, width=w)
    elif kind == 1: d.line([(S / 2, 25), (S / 2, S - 25)], fill=MINT, width=w)
    elif kind == 2: d.line([(25, S / 2), (S - 25, S / 2)], fill=MINT, width=w)
    elif kind == 3: d.arc([30, 30, S - 30, S - 30], 200, 340, fill=MINT, width=w)
    elif kind == 4:
        d.polygon([(S * 0.2, S * 0.8), (S * 0.5, S * 0.18), (S * 0.8, S * 0.8)], outline=MINT, width=w)
    elif kind == 5:
        d.ellipse([S * 0.25, S * 0.3, S * 0.75, S * 0.7], outline=MINT, width=w); d.ellipse([S * 0.43, S * 0.38, S * 0.57, S * 0.62], fill=MINT)
    elif kind == 6:
        for k in (-1, 0, 1): d.line([(S * 0.15, S * 0.5 + k * 22), (S * 0.85, S * 0.5 + k * 30)], fill=MINT, width=6)
    return im
TILES = [edge_tile(k) for k in range(7)]
def bars(img, items, t0, t, y0=980, win=None, bw=520):
    for i, (lab, p) in enumerate(items):
        y = y0 + i * 110; g = eo((t - t0 - 0.1 * i) / 0.6)
        put(img, tsp(lab, F('x', 48), WHITE if i == 0 else MUTE), 260, y, fin(t, t0 + 0.1 * i), 1)
        dd = ImageDraw.Draw(img)
        dd.rounded_rectangle([360, y - 26, 360 + bw, y + 26], radius=26, fill=(20, 34, 50, 255))
        if g > 0.02: dd.rounded_rectangle([360, y - 26, 360 + max(52, bw * p * g), y + 26], radius=26, fill=(WHITE if (win == i) else (MINT if i == 0 else (60, 110, 150))) + (255,))
        put(img, tsp(f"{round(p * 100 * g)}%", F('x', 40), WHITE), 360 + bw + 70, y, fin(t, t0 + 0.1 * i))

def shot_E(t):
    t0 = SCN['E'][0]; tr = LS[6] - 0.15
    img = BG_NN.copy()
    if t < tr:
        te, tp = wt(5, 'edges') - 0.3, wt(5, 'ears') - 0.3
        img.alpha_composite(net3d(t, 0.2 * math.sin(t * 0.3), 780, R=480, fire_t=t0 + 0.1, alpha=0.28))
        put(img, tsp("layer by layer", F('x', 64), WHITE), 540, 780, fin(t, t0 + 0.1, 0.3) * (1 - fin(t, te, 0.25)))
        put(img, pill("Early layers: edges", F('x', 42), 'dark', WHITE), 540, 420, fin(t, te), pop(t, te))
        for k in range(4): put(img, TILES[k], 540 + (k - 1.5) * 190, 580, fin(t, te + 0.1 * k), 0.9 + 0.1 * pop(t, te + 0.1 * k))
        if t > tp - 0.2:
            dd = ImageDraw.Draw(img)
            for k in range(4):
                for j in range(3):
                    a = int(120 * fin(t, tp - 0.2, 0.4)); dd.line([(540 + (k - 1.5) * 190, 660), (540 + (j - 1) * 260, 880)], fill=MINT + (a,), width=3)
        put(img, pill("Deeper layers: parts", F('x', 42), 'dark', WHITE), 540, 830, fin(t, tp), pop(t, tp))
        for j, (k, lab) in enumerate(((4, 'ears'), (5, 'eyes'), (6, 'whiskers'))):
            st = wt(5, lab) - 0.15
            put(img, TILES[k], 540 + (j - 1) * 260, 990, fin(t, st), 1.0 + 0.2 * pop(t, st))
            put(img, tsp(lab, F('sb', 34), MUTE), 540 + (j - 1) * 260, 1100, fin(t, st))
        return img
    img.alpha_composite(net3d(t, 0.3 * math.sin(t * 0.4), 640, R=400, fire_t=tr + 0.1, train=0.0, alpha=1.0))
    put(img, CAT_CARD, 150, 640, fin(t, tr, 0.3), 0.3)
    put(img, pill("weights: random", F('x', 40), 'dark', WHITE), 540, 330, fin(t, tr + 0.1), pop(t, tr + 0.1))
    tg = wt(6, 'toaster') - 0.3
    if t > tg - 0.6:
        bars(img, [("toaster", 0.61), ("cat", 0.12), ("banana", 0.09)], tg - 0.6, t, y0=1010, win=0 if t > tg else None)
    if t > tg: put(img, tsp("?!", F('x', 90), WHITE), 960, 1010, fin(t, tg, 0.15), pop(t, tg))
    return img

LOSS = [0.88 * math.exp(-k / 16) + 0.03 + 0.03 * math.sin(k * 1.9) * math.exp(-k / 30) for k in range(120)]
def shot_F(t):
    t0 = SCN['F'][0]; tm = wt(8, 'measure') - 0.2; tn = wt(8, 'nudge') - 0.2; tb = LS[9] - 0.15; trp = wt(9, 'Repeat') - 0.2
    img = BG_NN.copy()
    put(img, pill("Training", F('x', 54), 'grad', INK), 540, 330, fin(t, t0 + 0.1), pop(t, t0 + 0.1))
    if t < tb:
        img.alpha_composite(net3d(t, -0.2 + 0.05 * (t - t0), 740, R=430, back_t=tn if t > tn else None, train=1.0 if t > tn else 0.0))
        if t > tm:
            put(img, pill("error: 0.88", F('x', 52), 'white', INK), 540, 1150, fin(t, tm), pop(t, tm))
        if t > tn: put(img, tsp("← nudge every weight", F('sb', 38), MUTE), 540, 1240, fin(t, tn + 0.2))
        return img
    img.alpha_composite(net3d(t, 0.1 * math.sin(t), 600, R=320, back_t=tb + ((t - tb) // 0.9) * 0.9, train=1.0, alpha=0.9))
    put(img, pill("backpropagation", F('x', 50), 'white', INK), 540, 420, fin(t, tb), pop(t, tb))
    gx0, gy0, gw, gh = 140, 870, 800, 230
    dd = ImageDraw.Draw(img)
    dd.line([(gx0, gy0), (gx0, gy0 + gh), (gx0 + gw, gy0 + gh)], fill=(90, 130, 170), width=4)
    p = clamp((t - trp) / (SCN['F'][1] - trp - 0.2))
    n = max(2, int(120 * p))
    pts = [(gx0 + gw * k / 119, gy0 + gh - gh * LOSS[k] / 0.95) for k in range(n)]
    if t > trp: dd.line(pts, fill=MINT, width=7, joint='curve')
    put(img, tsp("error", F('sb', 30), MUTE), gx0 + 50, gy0 - 30, fin(t, tb + 0.3))
    att = int(10 ** (6 * p)) if t > trp else 1
    put(img, tsp(f"attempt {att:,}", F('x', 44), WHITE), 540, gy0 + gh + 55, fin(t, trp))
    return img

def shot_G(t):
    t0 = SCN['G'][0]
    img = BG_NN.copy()
    img.alpha_composite(net3d(t, 0.15 * math.sin(t * 0.5), 640, R=400, fire_t=t0 - 0.2, alpha=0.9))
    put(img, CAT_CARD, 150, 640, fin(t, t0 - 0.1, 0.3), 0.3)
    bars(img, [("cat", 0.98), ("toaster", 0.01), ("banana", 0.01)], t0, t, y0=1010, win=0)
    img.alpha_composite(glow(W, H, 540, 1010, 360, (150, 220, 255), 0.25 * fin(t, t0 + 0.6, 0.3), 1.6))
    put(img, pill("Learned!", F('x', 54), 'grad', INK), 540, 330, fin(t, t0 + 0.5), pop(t, t0 + 0.5))
    return img

def shot_H(t):
    t0 = SCN['H'][0]
    img = brain_scene(t - t0 + 1.5, 0, 1e9, R=420 - 80 * ease_span(t, t0, DUR), cy=760)
    tm = wt(11, 'mistakes') - 0.2
    put(img, tsp("by making mistakes", F('x', 72), WHITE), 540, 1100, fin(t, tm, 0.4) * (1 - fin(t, END_CARD - 0.1, 0.3)))
    if t >= END_CARD:
        put(img, pill("Save this for later", F('x', 50), 'grad', INK), 540, 1100, fin(t, END_CARD), pop(t, END_CARD, 0.4))
        put(img, tsp("Follow for more", F('sb', 34), MUTE), 540, 1200, fin(t, END_CARD + 0.3))
    return img

def scA(fr, t): fr.alpha_composite(shot_A(t))
def scB(fr, t): fr.alpha_composite(shot_B(t))
def scC(fr, t): fr.alpha_composite(shot_C(t))
def scD(fr, t): fr.alpha_composite(shot_D(t))
def scE(fr, t): fr.alpha_composite(shot_E(t))
def scF(fr, t): fr.alpha_composite(shot_F(t))
def scG(fr, t): fr.alpha_composite(shot_G(t))
def scH(fr, t): fr.alpha_composite(shot_H(t))
SCENEFN = {'A': scA, 'B': scB, 'C': scC, 'D': scD, 'E': scE, 'F': scF, 'G': scG, 'H': scH}
# deep low hits in the music on the big moments (times in seconds)
def HITS(): return [0.35, wt(6, 'toaster') - 0.05, LS[10] - 0.05]

# ===== EPISODE END =====



# ======================= CAPTIONS, WIPES, FX =======================
_capc = {}
def cap_sprite(ci, active):
    key = (ci, active)
    if key in _capc: return _capc[key]
    words = CHUNKS[ci][0]; f = F('x', 80); stroke = 8
    space = f.getlength(' '); maxw = 960
    lines, cur, cw = [], [], 0
    for j, (w, s, e) in enumerate(words):
        ww = f.getlength(w)
        if cur and cw + space + ww > maxw: lines.append(cur); cur, cw = [], 0
        cw += (space if cur else 0) + ww; cur.append((j, w, ww))
    lines.append(cur)
    lh = 96; hgt = lh * len(lines) + 30
    im, d = canvas(maxw + 60, hgt)
    for li, ln in enumerate(lines):
        tot = sum(x[2] for x in ln) + space * (len(ln) - 1); x = (maxw + 60 - tot) / 2
        for j, w, ww in ln:
            d.text((x, 10 + li * lh), w, font=f, fill=MINT if j == active else WHITE, stroke_width=stroke, stroke_fill=INK)
            x += ww + space
    if len(_capc) > 400: _capc.clear()
    _capc[key] = im; return im
def captions(fr, t):
    for ci, (words, s, e) in enumerate(CHUNKS):
        if s <= t < e:
            act = 0
            for j, (w, ws, we) in enumerate(words):
                if ws <= t: act = j
            a = fin(t, s, 0.08) * (1 - eio((t - (e - 0.1)) / 0.1))
            put(fr, cap_sprite(ci, act), 540, 1330, a, 0.82 + 0.18 * pop(t, s, 0.22))

_band = None
def band():
    global _band
    if _band is None:
        g = hgrad(520, H, VIO, PINK)
        im = Image.new('RGBA', (520 + 500, H), (0, 0, 0, 0)); im.paste(g, (250, 0))
        im = im.transform(im.size, Image.AFFINE, (1, -0.26, 0, 0, 1, 0), resample=Image.BILINEAR)
        _band = im
    return _band
TRANS = [SCN[k][0] for k in keys[1:]]
def wipes(fr, t):
    for ts in TRANS:
        p = (t - (ts - 0.22)) / 0.44
        if 0 <= p <= 1:
            b = band(); x = -b.width + (W + b.width * 2) * eio(p)
            put(fr, b, x, H / 2, 0.95)
PBAR = None
def progress(fr, t):
    global PBAR
    if PBAR is None: PBAR = hgrad(W, 8, VIO, PINK)
    w = int(W * t / DUR)
    if w > 0: fr.alpha_composite(PBAR.crop((0, 0, w, 8)), (0, 0))

SHAKES = []
_yy, _xx = np.ogrid[0:H, 0:W]
_rv = np.sqrt(((_xx - W / 2) / (W * 0.62)) ** 2 + ((_yy - H / 2) / (H * 0.62)) ** 2)
VIG = (1 - 0.6 * np.clip((_rv - 0.45) / 0.75, 0, 1) ** 1.6).astype(np.float32)[..., None]
_gr = np.random.default_rng(77)
GRAIN = [_gr.normal(0, 6.5, (H, W)).astype(np.float32)[..., None] for _ in range(6)]
def darkness(t):
    k = min(1.0, t / 0.35, max(0.0, (DUR - t) / 0.8))
    for b in TRANS:
        k = min(k, clamp(abs(t - b) / 0.16))
    return k
def post(img, t):
    small = img.resize((W // 4, H // 4), Image.BILINEAR)
    br = np.clip(np.asarray(small).astype(np.float32) - 150, 0, None) * 1.5
    bl = Image.fromarray(br.clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(9)).resize((W, H), Image.BILINEAR)
    a = np.asarray(img).astype(np.float32) + np.asarray(bl).astype(np.float32) * 0.6
    a = lens(a) * VIG * darkness(t) + GRAIN[int(t * 24) % 6]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
def _mote_sprite(r):
    im = Image.new('RGBA', (4 * r, 4 * r), (0, 0, 0, 0))
    ImageDraw.Draw(im).ellipse([r, r, 3 * r, 3 * r], fill=(200, 230, 255, 120))
    return im.filter(ImageFilter.GaussianBlur(r * 0.6))
_mr = np.random.default_rng(12)
MOTES = [(_mr.uniform(0, W), _mr.uniform(0, H), int(_mr.uniform(5, 18)), _mr.uniform(6, 22), _mr.uniform(0, 6.28)) for _ in range(26)]
MOTE_SPR = {r: _mote_sprite(r) for r in range(5, 19)}
def motes(fr, t):
    for x, y, r, sp, ph in MOTES:
        yy = (y - t * sp) % H; xx = x + 30 * math.sin(t * 0.3 + ph)
        put(fr, MOTE_SPR[r], xx, yy, 0.35 + 0.35 * math.sin(t * 0.8 + ph) ** 2)
_cm = np.clip((_rv - 0.35) / 0.6, 0, 1).astype(np.float32)
def lens(a):
    sh = 4
    r = np.roll(a[..., 0], sh, axis=1); b = np.roll(a[..., 2], -sh, axis=1)
    a[..., 0] = a[..., 0] * (1 - _cm) + r * _cm; a[..., 2] = a[..., 2] * (1 - _cm) + b * _cm
    return a
def handheld(img, t):
    ox = 7 * math.sin(t * 1.27) + 4 * math.sin(t * 2.9 + 1) + 1.5 * math.sin(t * 7.1)
    oy = 6 * math.cos(t * 1.05) + 4 * math.sin(t * 3.3 + 2) + 1.5 * math.cos(t * 6.3)
    z = 1.035; cw, ch = W / z, H / z
    x0 = (W - cw) / 2 + ox; y0 = (H - ch) / 2 + oy
    return img.resize((W, H), Image.BILINEAR, box=(x0, y0, x0 + cw, y0 + ch))
def frame(t):
    fr = Image.new('RGBA', (W, H), BG0 + (255,))
    for k in keys:
        s, e = SCN[k]
        if s <= t < e + 0.02: SCENEFN[k](fr, t)
    motes(fr, t)
    fr = handheld(fr, t)
    out = post(fr.convert('RGB'), t).convert('RGBA')
    bug(out, t); captions(out, t); progress(out, t)
    return out.convert('RGB')

# ======================= AUDIO =======================
SR = 44100
def mtof(m): return 440 * 2 ** ((m - 69) / 12)
def bp(x, lo, hi, o=2):
    b, a = butter(o, [lo / (SR / 2), hi / (SR / 2)], 'band'); return lfilter(b, a, x)
def lp(x, f, o=2):
    b, a = butter(o, f / (SR / 2)); return lfilter(b, a, x)
def hp(x, f, o=2):
    b, a = butter(o, f / (SR / 2), 'high'); return lfilter(b, a, x)
def add(buf, sig, t0, gain=1.0, pan=0.0):
    i = int(t0 * SR)
    if i >= len(buf): return
    n = min(len(sig), len(buf) - i)
    if sig.ndim == 1:
        buf[i:i + n, 0] += sig[:n] * gain * (1 - max(0, pan))
        buf[i:i + n, 1] += sig[:n] * gain * (1 + min(0, pan))
    else:
        buf[i:i + n] += sig[:n] * gain


def make_audio(path):
    from scipy.signal import resample_poly
    from scipy.ndimage import uniform_filter1d
    n = int(DUR * SR); tn = np.arange(n) / SR; rng_ = np.random.default_rng(5)
    voice = np.zeros(n)
    for i, a in enumerate(VO):
        v = resample_poly(a, 147, 80); st = int(VO_T[i] * SR); k = min(len(v), n - st)
        voice[st:st + k] += v[:k]
    act = np.abs(voice) > 1e-3
    voice *= (10 ** (-18 / 20)) / np.sqrt(np.mean(voice[act] ** 2))
    CH = [[45, 48, 52, 59], [41, 45, 48, 55], [48, 52, 55, 59], [43, 47, 50, 57]]
    pad = np.zeros((n, 2)); seg = 6.0
    for b in range(int(DUR / seg) + 2):
        notes = CH[b % 4]; t0 = b * seg
        if t0 >= DUR: break
        L = int((seg + 3) * SR); tt = np.arange(L) / SR
        env = np.minimum(1, tt / 2.2) * np.where(tt > seg, np.exp(-(tt - seg) * 1.2), 1)
        for side, det in ((0, -0.04), (1, 0.04)):
            s = np.zeros(L)
            for m in notes:
                f = mtof(m) * 2 ** (det / 12)
                s += np.sin(2 * np.pi * f * tt) + 0.22 * np.sin(4 * np.pi * f * tt) + 0.06 * np.sin(6 * np.pi * f * tt)
            s *= env / len(notes) * (1 + 0.15 * np.sin(2 * np.pi * 0.15 * tt + side * 1.5))
            i0 = int(t0 * SR); k = min(L, n - i0)
            if k > 0: pad[i0:i0 + k, side] += s[:k]
    pad = np.stack([lp(pad[:, 0], 850), lp(pad[:, 1], 850)], 1)
    bells = np.zeros((n, 2)); pent = [57, 60, 62, 64, 67, 69]; tb = 2.0
    while tb < DUR - 2.5:
        m = pent[rng_.integers(len(pent))]; L = int(3.5 * SR); tt = np.arange(L) / SR
        s = (np.sin(2 * np.pi * mtof(m) * tt) + 0.15 * np.sin(2 * np.pi * mtof(m) * 2.01 * tt)) * np.exp(-tt * 1.3) * np.minimum(1, tt / 0.02)
        add(bells, s, tb, 1.0, rng_.uniform(-0.5, 0.5)); tb += rng_.uniform(2.5, 4.0)
    music = pad * 0.5 + bells * 0.10
    music *= (10 ** (-30 / 20)) / np.sqrt(np.mean(music ** 2))
    ve = uniform_filter1d(np.abs(voice), int(0.3 * SR))
    music *= (1 - 0.35 * np.clip(ve / (ve.max() * 0.3), 0, 1))[:, None]
    sfx = np.zeros((n, 2))
    def boom(Lb=1.8):
        tt = np.arange(int(Lb * SR)) / SR
        f = 36 + 46 * np.exp(-tt * 5); ph = 2 * np.pi * np.cumsum(f) / SR
        return np.sin(ph) * np.exp(-tt * 2.0) * np.minimum(1, tt / 0.01)
    for ts in HITS():
        add(sfx, boom(), ts, 0.18)
    for a0_, a1_, rate in ():
        tk = a0_
        while tk < a1_:
            Lc = int(0.03 * SR); tt = np.arange(Lc) / SR
            c = bp(rng_.standard_normal(Lc), 1800, 6000) * np.exp(-tt * 160)
            add(sfx, c, tk, 0.03, pan=float(rng_.uniform(-0.3, 0.3))); tk += float(rng_.uniform(0.06, 0.15)) / rate
    for ts in TRANS:
        L = int(0.9 * SR); tt = np.arange(L) / SR
        add(sfx, lp(rng_.standard_normal(L), 1500) * np.sin(np.pi * tt / 0.9) ** 2, ts - 0.5, 0.025)
    mix = music + sfx + voice[:, None]
    mix *= np.minimum(1, tn / 0.6)[:, None] * np.clip((DUR - tn) / 1.5, 0, 1)[:, None]
    pk = np.max(np.abs(mix))
    if pk > 0.95: mix *= 0.95 / pk
    for name, x in (("voice", voice), ("music", music)):
        print(name, "rms dBFS", round(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12), 1))
    wavfile.write(path, SR, (mix * 32767).astype(np.int16))

def print_script():
    for i, (sc, text) in enumerate(LINES):
        m, s = divmod(LS[i], 60)
        print(f"{int(m)}:{s:05.2f}-{LE[i]:5.2f}  {text}")
    print("DUR", DUR)

if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'script': print_script()
    elif cmd == 'stills':
        for ts in sys.argv[3:]:
            frame(float(ts)).save(f"{sys.argv[2]}/s_{float(ts):05.2f}.png")
    elif cmd == 'audio': make_audio(sys.argv[2])
    elif cmd == 'render':
        out, wav = sys.argv[2], sys.argv[3]
        p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                              '-r', str(FPS), '-i', '-', '-i', wav, '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
                              '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', out],
                             stdin=subprocess.PIPE)
        for i in range(int(DUR * FPS)):
            p.stdin.write(frame(i / FPS).tobytes())
        p.stdin.close(); p.wait(); print("done", p.returncode)
