# ---------- PHP2Shell episode: "The flaw behind 7 out of 10 websites"
# All drawn from scratch: original silhouettes, no logos, no real sites. Light blue + black only.

def glitch(img, amt, seed):
    if amt <= 0.02: return img
    r = np.random.default_rng(seed)
    a = np.asarray(img).copy()
    for _ in range(int(5 + 16 * amt)):
        y = int(r.integers(0, H - 60)); h = int(r.integers(6, 70)); dx = int(r.integers(-110, 110) * amt)
        a[y:y + h] = np.roll(a[y:y + h], dx, axis=1)
    ghost = np.roll(a, int(14 * amt), axis=1).astype(np.float32)
    ghost[..., 0] *= 0.35; ghost[..., 1] *= 0.8
    out = np.maximum(a.astype(np.float32), ghost * 0.55 * amt)
    out[..., 3] = 255
    return Image.fromarray(out.clip(0, 255).astype(np.uint8), 'RGBA')

def flash(img, amt):
    if amt > 0.01: img.alpha_composite(Image.new('RGBA', (W, H), (200, 235, 255, int(150 * clamp(amt)))))

def hood_mask(S):
    m = Image.new('L', (S, S), 0); d = ImageDraw.Draw(m); c = S / 2
    d.rounded_rectangle([c - 0.47 * S, 0.56 * S, c + 0.47 * S, 1.35 * S], radius=int(0.24 * S), fill=255)
    d.polygon([(c - 0.3 * S, 0.66 * S), (c + 0.3 * S, 0.66 * S), (c + 0.22 * S, 0.3 * S), (c, 0.1 * S), (c - 0.22 * S, 0.3 * S)], fill=255)
    d.ellipse([c - 0.21 * S, 0.12 * S, c + 0.21 * S, 0.6 * S], fill=255)
    return m.filter(ImageFilter.GaussianBlur(1.5))
HOOD = rimlit(hood_mask(760), fill=(4, 7, 11), width=11, strength=1.8)

# ---------- shot A: the hook — the night globe, then 7 of 10 websites light up
def site_icon(S, lit):
    im, d = canvas(S, int(S * 0.78))
    col = MINT if lit else (40, 60, 80)
    d.rounded_rectangle([2, 2, S - 3, int(S * 0.78) - 3], radius=int(S * 0.08), fill=(10, 18, 28, 255), outline=col + (255,), width=max(3, S // 30))
    d.rectangle([2, 2, S - 3, int(S * 0.17)], fill=col + (90 if lit else 60,))
    for k in range(3): d.ellipse([S * 0.06 + k * S * 0.07, S * 0.06, S * 0.1 + k * S * 0.07, S * 0.1], fill=col + (255,))
    for k in range(3): d.rounded_rectangle([S * 0.1, S * (0.3 + k * 0.13), S * (0.9 - 0.2 * (k == 2)), S * (0.36 + k * 0.13)], radius=S // 40, fill=col + (150 if lit else 80,))
    return im
SITE_ON, SITE_OFF = site_icon(200, True), site_icon(200, False)

def shot_hook(t):
    t0 = 0.0; t7 = wt(0, 'seven') - 0.15
    if t < t7:
        img = globe_scene(t, t0, arcs_t=0.6, cy=820, scale=1.0 + 0.06 * ease_span(t, 0, t7))
        tf = wt(0, 'flaws')
        g = clamp(1 - abs(t - tf - 0.15) / 0.35)
        img = glitch(img, g, int(t * 30))
        return img
    img = globe_scene(t, t0, arcs_t=0.6, dim=0.75, cy=820, scale=1.12)
    for i in range(10):
        r, c = divmod(i, 5)
        x = 540 + (c - 2) * 190; y = 640 + r * 230
        ti = t7 + 0.08 * i
        on = i < 7 and t >= wt(0, 'websites') - 0.35 + 0.07 * i
        put(img, SITE_ON if on else SITE_OFF, x, y, fin(t, ti, 0.25), 0.82 * (0.85 + 0.15 * pop(t, ti, 0.3)))
        if on: img.alpha_composite(glow(W, H, x, y, 120, (110, 200, 250), 0.18, 1.6))
    put(img, pill("7 out of 10 websites", F('x', 54), 'grad', INK), 540, 1010, fin(t, wt(0, 'websites') - 0.1), pop(t, wt(0, 'websites') - 0.1))
    return img

# ---------- shot B: the engine nobody sees, then a page being built
def big_word(s, size):
    f = F('x', size); w_ = int(f.getlength(s)) + 80; h_ = int(size * 1.4)
    im, d = canvas(w_, h_); d.text((w_ / 2, h_ / 2), s, font=f, fill=(225, 244, 255, 255), anchor='mm')
    halo = im.filter(ImageFilter.GaussianBlur(18)); out = Image.new('RGBA', im.size, (0, 0, 0, 0))
    hb = Image.new('RGBA', im.size, MINT + (0,)); hb.putalpha(halo.getchannel('A').point(lambda v: min(255, int(v * 1.6))))
    out.alpha_composite(hb); out.alpha_composite(im)
    return out
PHPW = big_word("PHP", 260)

def page_build(t, t0):
    img = vgrad(W, H, (6, 12, 20), (2, 4, 8)); img.alpha_composite(glow(W, H, 540, 760, 640, (70, 140, 200), 0.24, 1.6))
    fr_ = card(820, 1000, 'grad', r=36); put(img, fr_, 540, 780, fin(t, t0, 0.3), 0.96 + 0.04 * fin(t, t0, 0.4))
    d = ImageDraw.Draw(img)
    for k in range(3): d.ellipse([170 + k * 34, 312, 192 + k * 34, 334], fill=MINT)
    d.rounded_rectangle([290, 304, 900, 342], radius=19, fill=(20, 36, 52))
    d.text((310, 323), "a-website.example", font=F('m', 26), fill=MUTE, anchor='lm')
    blocks = [((170, 380, 910, 470), 'bar'), ((170, 500, 910, 780), 'img'), ((170, 810, 760, 850), 'ln'), ((170, 875, 880, 915), 'ln'),
              ((170, 940, 700, 980), 'ln'), ((170, 1030, 470, 1110), 'btn'), ((540, 1030, 910, 1110), 'ln2')]
    for i, (bx, kind) in enumerate(blocks):
        st = t0 + 0.35 + i * 0.22; p = eo((t - st) / 0.45)
        if p <= 0: continue
        dy = -260 * (1 - p)
        x0, y0, x1, y1 = bx; y0 += dy; y1 += dy
        lay = Image.new('RGBA', (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay)
        a = int(255 * min(1, p * 1.5))
        if kind == 'img':
            ld.rounded_rectangle([x0, y0, x1, y1], radius=20, fill=(24, 60, 90, a))
            ld.polygon([(x0 + 60, y1 - 40), (x0 + 260, y0 + 90), (x0 + 420, y1 - 40)], fill=(56, 140, 200, a))
            ld.polygon([(x0 + 330, y1 - 40), (x0 + 520, y0 + 130), (x1 - 60, y1 - 40)], fill=(125, 211, 252, a))
            ld.ellipse([x1 - 150, y0 + 40, x1 - 80, y0 + 110], fill=(224, 242, 254, a))
        elif kind == 'btn':
            ld.rounded_rectangle([x0, y0, x1, y1], radius=40, fill=MINT + (a,))
        elif kind == 'bar':
            ld.rounded_rectangle([x0, y0, x1, y1], radius=18, fill=(30, 60, 90, a))
            ld.rounded_rectangle([x0 + 30, y0 + 30, x0 + 230, y0 + 60], radius=12, fill=WHITE + (a,))
        else:
            ld.rounded_rectangle([x0, y0, x1, y1], radius=14, fill=(110, 150, 185, int(a * 0.7)))
        img.alpha_composite(lay)
    return img

def shot_php(t):
    t0 = SCN['B'][0]; tb = wt(1, 'builds') - 0.25
    if t < tb:
        img = corridor(t, t0 - 2.0)
        a = fin(t, wt(1, 'PHP') - 0.2, 0.3)
        s = 0.85 + 0.15 * pop(t, wt(1, 'PHP') - 0.2, 0.5) + 0.03 * (t - t0)
        img.alpha_composite(glow(W, H, 540, 760, 420, (100, 190, 250), 0.35 * a, 1.6))
        put(img, PHPW, 540, 760, a, s)
        put(img, tsp("the engine behind the page", F('sb', 40), MUTE), 540, 940, fin(t, wt(1, 'never') - 0.1))
        return img
    return page_build(t, tb)

# ---------- shot C: a stranger, a lock that opens without a key
def _metal(w, h, top, bot, streak=True):
    im = vgrad(w, h, top, bot)
    if streak:
        a = np.asarray(im).astype(np.float32)
        x = np.linspace(0, 1, w)[None, :]
        band = np.exp(-((x - 0.28) / 0.07) ** 2) * 70 + np.exp(-((x - 0.7) / 0.18) ** 2) * 25
        a[..., :3] += band[..., None]
        rr = np.random.default_rng(4); a[..., :3] += rr.normal(0, 3, (h, w, 1))
        im = Image.fromarray(a.clip(0, 255).astype(np.uint8), 'RGBA')
    return im
_LOCK_BODY = None
def _lock_body():
    global _LOCK_BODY
    if _LOCK_BODY is None:
        bw, bh = 440, 320
        m = Image.new('L', (bw, bh), 0); ImageDraw.Draw(m).rounded_rectangle([0, 0, bw - 1, bh - 1], radius=54, fill=255)
        body = _metal(bw, bh, (120, 150, 175), (22, 34, 48)); body.putalpha(m)
        d = ImageDraw.Draw(body)
        d.rounded_rectangle([3, 3, bw - 4, bh - 4], radius=52, outline=(210, 232, 248, 150), width=3)
        d.line([(60, 14), (bw - 60, 14)], fill=(235, 246, 255, 170), width=4)
        for k in range(4):
            d.ellipse([28 + k * 0, 0, 0, 0], fill=None) if False else None
        for (x, y) in ((36, 36), (bw - 36, 36), (36, bh - 36), (bw - 36, bh - 36)):
            d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=(40, 58, 76, 255), outline=(170, 200, 225, 200), width=2)
        _LOCK_BODY = body
    return _LOCK_BODY
def padlock(o, glowk=0.0):
    S = 640; im = Image.new('RGBA', (S, S), (0, 0, 0, 0))
    cx = S / 2; lift = 70 * o
    sm = Image.new('L', (S, S), 0); sd = ImageDraw.Draw(sm)
    sd.arc([cx - 140, 60 - lift, cx + 140, 340 - lift], 180, 360, fill=255, width=46)
    sd.line([(cx - 117, 200 - lift), (cx - 117, 330 - lift)], fill=255, width=46)
    sd.line([(cx + 117, 200 - lift), (cx + 117, 330 - lift)], fill=255, width=46)
    chrome = _metal(S, S, (230, 242, 252), (70, 96, 120)); chrome.putalpha(sm)
    if o > 0: chrome = chrome.rotate(28 * o, resample=Image.BICUBIC, center=(cx + 117, 330))
    im.alpha_composite(chrome)
    sh = Image.new('RGBA', (460, 60), (0, 0, 0, 0)); ImageDraw.Draw(sh).ellipse([0, 0, 459, 59], fill=(0, 0, 0, 150))
    im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(12)), (int(cx - 230), 586))
    im.alpha_composite(_lock_body(), (int(cx - 220), 290))
    d = ImageDraw.Draw(im)
    kc = lerpc((20, 34, 48), (230, 248, 255), glowk)
    d.ellipse([cx - 36, 388, cx + 36, 460], fill=(8, 14, 20, 255)); d.polygon([(cx - 20, 440), (cx + 20, 440), (cx + 32, 530), (cx - 32, 530)], fill=(8, 14, 20, 255))
    d.ellipse([cx - 28, 396, cx + 28, 452], fill=kc + (255,)); d.polygon([(cx - 14, 440), (cx + 14, 440), (cx + 24, 522), (cx - 24, 522)], fill=kc + (255,))
    if glowk > 0.05:
        g = glow(S, S, cx, 440, 160, (170, 225, 255), 0.6 * glowk, 1.6); im.alpha_composite(g)
    return im

def pw_field(t, tt):
    im, d = canvas(720, 120)
    d.rounded_rectangle([0, 0, 719, 119], radius=60, fill=(12, 22, 34, 255), outline=(90, 140, 180, 255), width=4)
    d.text((50, 60), "password", font=F('m', 34), fill=MUTE, anchor='lm')
    for k in range(6): d.ellipse([280 + k * 52, 46, 308 + k * 52, 74], fill=WHITE)
    if int(t * 2.5) % 2 == 0: d.rectangle([600, 30, 606, 90], fill=MINT)
    return im

def shot_stranger(t):
    t0 = SCN['C'][0]
    img = corridor(14.0 + 0.4 * (t - t0), 14.0)
    img.alpha_composite(Image.new('RGBA', (W, H), (0, 0, 0, 120)))
    tpw = wt(3, 'password') - 0.3; tlog = wt(3, 'logging') - 0.05
    o = eob((t - tlog) / 0.45) if t > tlog else 0.0
    gk = clamp((t - wt(2, 'take')) / 1.0)
    img.alpha_composite(glow(W, H, 540, 640, 460, (110, 200, 250), 0.25 + 0.35 * gk + 0.4 * clamp(1 - (t - tlog) / 0.8) * (t > tlog), 1.5))
    z = 1.0 + 0.08 * ease_span(t, t0, SCN['C'][1])
    put(img, padlock(o, gk), 540, 640, fin(t, t0 + 0.2, 0.4), 0.95 * z)
    if t >= tpw:
        f = pw_field(t, tpw)
        put(img, f, 540, 1020, fin(t, tpw, 0.25) * (1 - fin(t, tlog + 0.3, 0.3)), 0.9 + 0.1 * pop(t, tpw))
        if t > wt(3, 'password') + 0.35:
            d = ImageDraw.Draw(img); p = clamp((t - wt(3, 'password') - 0.35) / 0.25) * (1 - fin(t, tlog + 0.3, 0.3))
            if p > 0: d.line([(190, 1020), (190 + 700 * clamp(p * 1.3), 1020)], fill=WHITE + (int(255 * p),), width=8)
    if t >= tlog:
        put(img, pill("No password. No login.", F('x', 50), 'grad', INK), 540, 1020, fin(t, tlog + 0.35), pop(t, tlog + 0.35))
    sx = 540 + 10 * math.sin(t * 0.7)
    put(img, HOOD, sx, 1380 + 4 * math.sin(t * 1.3), fin(t, t0 + 0.1, 0.6), 1.0 + 0.04 * ease_span(t, t0, SCN['C'][1]))
    img = glitch(img, clamp(1 - abs(t - tlog - 0.05) / 0.3), int(t * 31))
    flash(img, clamp(1 - (t - tlog) / 0.35) if t >= tlog else 0)
    return img

# ---------- shot D: 9.2 out of 10, then the quote
def gauge(v, S=640):
    im, d = canvas(2 * S, 2 * S); Z = 2 * S
    c = (Z / 2, Z * 0.55); R = Z * 0.42; w = int(Z * 0.06)
    d.arc([c[0] - R, c[1] - R, c[0] + R, c[1] + R], 150, 390, fill=(255, 255, 255, 40), width=w)
    end = 150 + 240 * v; steps = 40
    for k in range(steps):
        a0 = 150 + (end - 150) * k / steps; a1 = 150 + (end - 150) * (k + 1) / steps + 0.6
        if a1 > a0: d.arc([c[0] - R, c[1] - R, c[0] + R, c[1] + R], a0, a1, fill=lerpc((56, 140, 200), (230, 248, 255), k / steps), width=w)
    for k in range(11):
        a = math.radians(150 + 24 * k)
        d.line([(c[0] + math.cos(a) * R * 0.8, c[1] + math.sin(a) * R * 0.8), (c[0] + math.cos(a) * R * 0.88, c[1] + math.sin(a) * R * 0.88)], fill=(140, 170, 200, 200), width=8)
    return im.resize((S, S), Image.LANCZOS)

def shot_severity(t):
    t0 = SCN['D'][0]; tv = wt(4, '9.2') - 0.35; tq = LS[5] - 0.25
    img = vgrad(W, H, (6, 10, 18), (1, 3, 6))
    v = 0.92 * eo((t - tv) / 1.1) if t > tv else 0.0
    hot = clamp((v - 0.6) / 0.32)
    img.alpha_composite(glow(W, H, 540, 700, 620, lerpc((50, 110, 170), (150, 220, 255), hot), 0.18 + 0.3 * hot, 1.5))
    gy = 700 if t < tq else 700 - 230 * eo((t - tq) / 0.6)
    gs = 1.0 if t < tq else 1.0 - 0.35 * eo((t - tq) / 0.6)
    g = gauge(v)
    num = tsp(f"{10 * v:.1f}", F('x', 170), WHITE)
    lay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    put(lay, g, 540, gy, fin(t, t0, 0.3), gs)
    put(lay, num, 540, gy + 50 * gs, fin(t, tv, 0.2), gs)
    put(lay, tsp("out of 10", F('sb', 44), MUTE), 540, gy + 170 * gs, fin(t, tv + 0.3), gs)
    shake = 9 * clamp(1 - (t - (tv + 1.1)) / 0.4) if t > tv + 1.1 else 0
    img.alpha_composite(lay, (int(shake * math.sin(t * 90)), int(shake * math.cos(t * 77))))
    tc = tv + 1.1
    if t >= tc:
        put(img, pill("CRITICAL", F('x', 64), 'white', INK), 540, gy + 290 * gs, fin(t, tc, 0.15) * (0.8 + 0.2 * math.sin(t * 9)), pop(t, tc, 0.3) * gs)
        flash(img, clamp(1 - (t - tc) / 0.3) * 0.7)
    if t >= tq:
        c = card(900, 420, 'grad', r=40).copy(); cd = ImageDraw.Draw(c)
        cd.text((70, 80), "“", font=F('x', 150), fill=MINT, anchor='lm')
        cd.text((450, 190), "One of the biggest", font=F('x', 64), fill=WHITE, anchor='mm')
        cd.text((450, 270), "holes in years", font=F('x', 64), fill=WHITE, anchor='mm')
        cd.text((450, 360), "a well-known security researcher", font=F('sb', 32), fill=MUTE, anchor='mm')
        put(img, c, 540 + 60 * (1 - eo((t - tq - 0.2) / 0.5)), 960, fin(t, tq + 0.2, 0.35))
    return img

# ---------- shot E: what it means for you — a phone leaks data, then lies to you
def scr_login(w, h):
    im = vgrad(w, h, (16, 34, 54), (6, 14, 26)); d = ImageDraw.Draw(im)
    d.rounded_rectangle([20, 60, w - 20, 110], radius=25, fill=(10, 22, 36)); d.text((44, 85), "shop.example", font=F('m', 24), fill=MUTE, anchor='lm')
    d.text((w / 2, 230), "Welcome back", font=F('x', 40), fill=WHITE, anchor='mm')
    for k, lab in enumerate(["you@email.com", "••••••••"]):
        d.rounded_rectangle([34, 300 + k * 110, w - 34, 380 + k * 110], radius=20, fill=(10, 22, 36), outline=(60, 100, 140), width=3)
        d.text((60, 340 + k * 110), lab, font=F('m', 28), fill=(170, 200, 225), anchor='lm')
    d.rounded_rectangle([34, 540, w - 34, 620], radius=40, fill=MINT); d.text((w / 2, 580), "Log in", font=F('x', 32), fill=INK, anchor='mm')
    for k in range(4): d.rounded_rectangle([34, 680 + k * 34, w - 34 - 80 * (k % 2), 696 + k * 34], radius=8, fill=(40, 70, 100))
    return im
def scr_scam(w, h):
    im = vgrad(w, h, (30, 70, 110), (8, 20, 36)); d = ImageDraw.Draw(im)
    d.rounded_rectangle([20, 60, w - 20, 110], radius=25, fill=(10, 22, 36)); d.text((44, 85), "shop.example", font=F('m', 24), fill=MUTE, anchor='lm')
    d.text((w / 2, 260), "YOU WON!", font=F('x', 72), fill=WHITE, anchor='mm')
    d.text((w / 2, 340), "Claim your prize now", font=F('sb', 30), fill=YEL, anchor='mm')
    d.rounded_rectangle([60, 430, w - 60, 520], radius=45, fill=WHITE); d.text((w / 2, 475), "Enter card details", font=F('x', 28), fill=INK, anchor='mm')
    for k in range(6):
        x = 40 + (k * 61) % (w - 80); y = 600 + (k * 97) % 180
        d.polygon([(x, y - 16), (x + 6, y - 4), (x + 18, y), (x + 6, y + 4), (x, y + 16), (x - 6, y + 4), (x - 18, y), (x - 6, y - 4)], fill=PINK)
    return im
PW_, PH_ = 440, 900
def phone_with(screen):
    m = Image.new('L', (PW_, PH_), 0); ImageDraw.Draw(m).rounded_rectangle([0, 0, PW_ - 1, PH_ - 1], radius=64, fill=255)
    ph = rimlit(m, fill=(14, 18, 24), width=7, strength=1.2, top_bias=False)
    msk = Image.new('L', screen.size, 0); ImageDraw.Draw(msk).rounded_rectangle([0, 0, screen.size[0] - 1, screen.size[1] - 1], radius=50, fill=255)
    ph.paste(screen, (18, 18), msk)
    return ph
PH_LOGIN = phone_with(scr_login(PW_ - 36, PH_ - 36)); PH_SCAM = phone_with(scr_scam(PW_ - 36, PH_ - 36))
_fm = Image.new('L', (900, 900), 0); _fd = ImageDraw.Draw(_fm)
for _y in [250, 350, 450, 550]: _fd.rounded_rectangle([610, _y, 770, _y + 92], radius=46, fill=255)
FINGERS = rimlit(_fm.filter(ImageFilter.GaussianBlur(2)), fill=(8, 11, 16), width=11, strength=1.4)
LEAKS = ["password", "email", "home address", "card number", "messages", "login token"]

def shot_leak(t):
    t0 = SCN['E'][0]; tl = wt(6, 'leak') - 0.1; ts = LS[7] - 0.1
    img = cam(BOKEH, 1.0 + 0.04 * (t - t0), 540, 900).copy()
    img.alpha_composite(Image.new('RGBA', (W, H), (0, 0, 0, 90)))
    fy = 6 * math.sin(t * 1.6)
    img.alpha_composite(glow(W, H, 540, 800, 520, (110, 190, 245), 0.3, 1.6))
    ph = PH_LOGIN if t < ts + 0.15 else PH_SCAM
    img.alpha_composite(HAND, (540 - 470, int(900 + fy)))
    put(img, ph.rotate(-4, resample=Image.BICUBIC, expand=True), 540, 800 + fy, fin(t, t0, 0.3), 0.98 + 0.02 * ease_span(t, t0, SCN['E'][1]))
    img.alpha_composite(FINGERS, (540 - 470, int(900 + fy)))
    if t >= tl:
        for i, lab in enumerate(LEAKS):
            st = tl + i * 0.22; p = clamp((t - st) / 2.4)
            if p <= 0: continue
            side = -1 if i % 2 == 0 else 1
            x = 540 + side * (60 + 290 * eo(p * 1.6)); y = 470 + i * 105 - 40 * p
            put(img, pill(lab, F('sb', 40), 'dark', WHITE), x, y, fin(t, st, 0.2) * (1 - clamp((p - 0.7) / 0.3)), 0.7 + 0.3 * pop(t, st))
    g = clamp(1 - abs(t - ts - 0.15) / 0.3)
    img = glitch(img, g, int(t * 29))
    return img

# ---------- shot F: good news (patched servers), bad news (forgotten sites)
def rack(p, done):
    im, d = canvas(260, 520)
    d.rounded_rectangle([0, 0, 259, 519], radius=18, fill=(12, 20, 30, 255), outline=(50, 80, 110, 255), width=4)
    for k in range(8):
        y = 30 + k * 52; d.rounded_rectangle([20, y, 239, y + 36], radius=8, fill=(18, 30, 44, 255))
        d.ellipse([200, y + 12, 212, y + 24], fill=(MINT if done or (k + int(p * 20)) % 3 == 0 else (40, 70, 90)) + (255,))
    d.rounded_rectangle([20, 450, 239, 480], radius=15, fill=(20, 36, 52, 255))
    d.rounded_rectangle([20, 450, 20 + 219 * p, 480], radius=15, fill=(MINT if not done else WHITE) + (255,))
    return im
FORGOT = {4, 9, 10}
def site_card(state, dust):
    im, d = canvas(270, 210)
    col = MINT if state == 'ok' else (70, 90, 110)
    d.rounded_rectangle([0, 0, 269, 209], radius=20, fill=(12, 20, 30, 255) if state == 'ok' else (14, 16, 20, 255), outline=col + (255,), width=4)
    d.rectangle([4, 4, 265, 34], fill=col + (60,))
    for k in range(3): d.rounded_rectangle([22, 60 + k * 36, 240 - 50 * (k == 2), 76 + k * 36], radius=6, fill=col + (110,))
    if state == 'ok':
        d.ellipse([200, 140, 250, 190], fill=MINT); d.line([(212, 165), (222, 176), (240, 152)], fill=INK, width=6)
    if dust:
        rr = np.random.default_rng(dust)
        for _ in range(60):
            x, y = rr.uniform(0, 270), rr.uniform(0, 210); r = rr.uniform(1, 3)
            d.ellipse([x - r, y - r, x + r, y + r], fill=(120, 130, 140, 160))
        d.line([(0, 0), (60, 40), (30, 90)], fill=(140, 150, 160, 120), width=2)
    return im
CARD_OK = site_card('ok', 0); CARD_WAIT = site_card('wait', 0)
CARD_OLD = [site_card('old', 11 + i) for i in range(3)]

def old_site(w, h, t):
    fl = 0.82 + 0.18 * (math.sin(t * 37) > 0.7) + 0.05 * math.sin(t * 3)
    im = vgrad(w, h, (int(30 * fl), int(52 * fl), int(70 * fl)), (int(14 * fl), int(24 * fl), int(34 * fl))); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w, 34], fill=(50, 80, 105))
    d.rectangle([14, 50, w - 14, 120], fill=(60, 95, 125))
    for k in range(6): d.rectangle([14, 140 + k * 26, w - 14 - 60 * (k % 3), 152 + k * 26], fill=(70, 100, 125))
    d.rectangle([14, 310, 150, 345], fill=(90, 140, 175))
    im.alpha_composite(scanlines(w, h, 4, 60))
    return im
def cobweb(S):
    im, d = canvas(S, S)
    for k in range(7):
        a = math.radians(k * 13); d.line([(0, 0), (math.cos(a) * S, math.sin(a) * S)], fill=(150, 170, 190, 90), width=2)
    for r in range(40, S, 46):
        pts = [(math.cos(math.radians(k * 13)) * r * (1 + 0.06 * (k % 2)), math.sin(math.radians(k * 13)) * r * (1 + 0.06 * (k % 2))) for k in range(7)]
        d.line(pts, fill=(150, 170, 190, 70), width=2)
    return im
COBWEB = cobweb(260)
def old_room(t, t0):
    img = ROOM91.copy()
    w, h = SCR91[2] - SCR91[0], SCR91[3] - SCR91[1]
    img.paste(old_site(w, h, t), (SCR91[0], SCR91[1]))
    img.alpha_composite(COBWEB, (SCR91[0] - 70, SCR91[1] - 70))
    img.alpha_composite(COBWEB.transpose(Image.FLIP_LEFT_RIGHT), (SCR91[2] - 150, 300))
    img.alpha_composite(Image.new('RGBA', (W, H), (0, 0, 0, 70)))
    z = 1.0 + 0.35 * ease_span(t, t0, t0 + 6)
    img = cam(img, z, 540, 720)
    put(img, pill("never updated", F('x', 46), 'dark', WHITE), 540, 1180, fin(t, t0 + 0.4), pop(t, t0 + 0.4))
    return img

def shot_news(t):
    t0 = SCN['F'][0]; tb = LS[9] - 0.2
    img = vgrad(W, H, (6, 12, 20), (2, 4, 8)); img.alpha_composite(glow(W, H, 540, 760, 640, (70, 140, 200), 0.24, 1.6))
    if t < tb:
        put(img, pill("The good news", F('x', 50), 'grad', INK), 540, 330, fin(t, t0 + 0.1), pop(t, t0 + 0.1))
        ck = clamp((t - wt(8, 'fix')) / 0.6)
        img.alpha_composite(glow(W, H, 540, 560, 300, (110, 200, 250), 0.35 * ck, 1.5))
        put(img, icon(ic_shield, 300, ck), 540, 560, fin(t, wt(8, 'fix') - 0.3), 0.8 + 0.2 * pop(t, wt(8, 'fix') - 0.3))
        th = wt(8, 'hosts') - 0.2
        for i in range(3):
            st = th + i * 0.15; p = clamp((t - st - 0.4) / (2.2 + 0.4 * i)); done = p >= 1
            put(img, rack(p, done), 540 + (i - 1) * 290, 990, fin(t, st, 0.3), 0.8)
        put(img, tsp("updating servers", F('sb', 36), MUTE), 540, 1240, fin(t, th + 0.4))
        return img
    tf = wt(9, 'forgotten') - 0.25
    if t >= tf:
        return old_room(t, tf)
    put(img, pill("The bad news", F('x', 50), 'white', INK), 540, 330, fin(t, tb), pop(t, tb))
    z = 1.0 + 0.5 * ease_span(t, wt(9, 'forgotten'), SCN['F'][1] + 0.3)
    lay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    oi = 0
    for i in range(12):
        r, c = divmod(i, 3)
        x, y = 540 + (c - 1) * 300, 560 + r * 240
        if i in FORGOT:
            fl = 0.55 + 0.45 * (math.sin(t * 13 + i * 2) > -0.6)
            put(lay, CARD_OLD[oi], x, y, fin(t, tb + 0.1 + i * 0.04, 0.2) * fl); oi += 1
            if t > wt(9, 'forgotten') - 0.1:
                put(lay, pill("never updated", F('sb', 26), 'dark', WHITE), x, y + 118, fin(t, wt(9, 'forgotten') - 0.1 + i * 0.05))
        else:
            ok = t > tb + 0.5 + i * 0.07
            put(lay, CARD_OK if ok else CARD_WAIT, x, y, fin(t, tb + 0.1 + i * 0.04, 0.2) * (1 - 0.55 * fin(t, wt(9, 'forgotten'), 0.5)))
    img.alpha_composite(cam(lay, z, 540, 560 + 3 * 240 * 0 + 240))
    return img

# ---------- shot G: ask your host
def shot_ask(t):
    t0 = SCN['G'][0]; tq = wt(10, 'ask') - 0.1
    q = len(QUESTION) * clamp((t - tq) / 1.2) if t > tq else 0
    img = chat_close(t, q, 0, dots=t > tq + 1.6, z=1.0 + 0.04 * (t - t0))
    put(img, pill("Ask your host", F('x', 50), 'grad', INK), 540, 330, fin(t, t0 + 0.1), pop(t, t0 + 0.1))
    return img

# ---------- shot H: the ending
def shot_end(t):
    t0 = SCN['H'][0]
    img = corridor(20 + 0.3 * (t - t0), 20)
    img.alpha_composite(Image.new('RGBA', (W, H), (0, 0, 0, int(120 + 60 * ease_span(t, t0, END_CARD)))))
    put(img, HOOD, 540, 1380 + 120 * ease_span(t, t0, END_CARD), 1 - 0.85 * ease_span(t, t0 + 0.5, END_CARD), 1.0 - 0.1 * ease_span(t, t0, END_CARD))
    tn = wt(11, 'never') - 0.2
    put(img, tsp("never see", F('x', 120), WHITE), 540, 840, fin(t, tn, 0.4) * (1 - fin(t, END_CARD - 0.2, 0.3)), 0.96 + 0.04 * fin(t, tn, 0.8))
    if t >= END_CARD:
        put(img, pill("Is your site updated?", F('x', 50), 'grad', INK), 540, 760, fin(t, END_CARD), pop(t, END_CARD, 0.4))
        put(img, tsp("Follow for more tech news", F('sb', 34), MUTE), 540, 860, fin(t, END_CARD + 0.3))
    return img

def bug(fr, t):
    if 0.8 < t < END_CARD:
        a = fin(t, 0.8, 0.6) * (1 - eio((t - (END_CARD - 0.4)) / 0.4))
        put(fr, tsp(BUGTXT, F('sb', 26), MUTE), 160, 228, a)

def scA(fr, t): fr.alpha_composite(shot_hook(t))
def scB(fr, t): fr.alpha_composite(shot_php(t))
def scC(fr, t): fr.alpha_composite(shot_stranger(t))
def scD(fr, t): fr.alpha_composite(shot_severity(t))
def scE(fr, t): fr.alpha_composite(shot_leak(t))
def scF(fr, t): fr.alpha_composite(shot_news(t))
def scG(fr, t): fr.alpha_composite(shot_ask(t))
def scH(fr, t): fr.alpha_composite(shot_end(t))
SCENEFN = {'A': scA, 'B': scB, 'C': scC, 'D': scD, 'E': scE, 'F': scF, 'G': scG, 'H': scH}