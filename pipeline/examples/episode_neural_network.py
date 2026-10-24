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
