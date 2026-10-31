"""Review sheets in the app's hero conditions, ported from the independent review's sheet script (10-31).

The desktop series hero at 1440x900: the logo fits a 640x260 box above the meta line, over the banner under a
black wash (bg-linear-to-t from-ink-950 via-ink-950/65 to-ink-950/30), with the logo's two drop shadows
(0 2px 2px rgb(255 255 255/.28), 0 4px 12px rgb(0 0 0/.85)). Each tile: that hero view at 0.875 scale, the logo on
solid dark and light, and the review's automatic warnings (cut edges, haze, strays, dark body, thin strokes).

  sheets.py hero <name> <ids...>      logos as they would ship (cut + finish: haze cleanup, light version)
  sheets.py hero <name> --lib <ids>   the library files themselves
  sheets.py fix <ids...>              per logo: the source crop with a grid in cover fractions, the kept alpha over
                                      it (dropped parts dimmed), and the logo as it would ship
"""
import json
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import studio as S

ROOT = S.ROOT
FONTS = '/fonts'  # host /usr/share/fonts/opentype/noto, mounted read-only by sheet.sh
fB = ImageFont.truetype(f'{FONTS}/NotoSansCJK-Bold.ttc', 16, index=1)
fR = ImageFont.truetype(f'{FONTS}/NotoSansCJK-Regular.ttc', 12, index=1)
fW = ImageFont.truetype(f'{FONTS}/NotoSansCJK-Bold.ttc', 12, index=1)

HERO_W, HERO_H = 1440, 504
LOGO_X, LOGO_BOTTOM = 400, 440
BOX_W, BOX_H = 640, 260
TILE_SCALE = 560 / 640
PANEL_W, PANEL_H = 576, 244
PAD = 8


def titles():
    meta = json.loads((ROOT / 'sources' / 'anilist.json').read_text())
    out = {}
    for s in json.loads((ROOT / 'sources' / 'series.json').read_text()):
        aid = s['anilist_id']
        if aid and aid in meta:
            t = meta[aid]['title']
            aka = t.get('english') or t.get('romaji') or ''
            title = s['title'] or aka
            out[aid] = (title, aka if aka.lower() != title.lower() else '')
    return out


def shipped(aid, lib=False):
    if lib:
        return np.asarray(Image.open(ROOT / 'library' / 'logos' / f'{aid}.webp').convert('RGBA'))
    pick = json.loads((ROOT / 'picks.json').read_text())[aid]
    return S.finish(pick, np.asarray(Image.open(ROOT / 'work' / 'cut' / aid / 'logo.png').convert('RGBA')))


# ------------------------------------------------------------------------------------------------ stats

def _longest_run(b):
    if not b.any():
        return 0
    d = np.diff(np.concatenate(([0], b.astype(np.int8), [0])))
    return int((np.nonzero(d == -1)[0] - np.nonzero(d == 1)[0]).max())


def _lum(rgb):
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def stats(arr):
    a = arr[:, :, 3].astype(np.uint8)
    rgb = arr[:, :, :3].astype(np.float32)
    H, W = a.shape
    s = min(BOX_W / W, BOX_H / H)
    st = {'W': W, 'H': H, 'hero_px': [round(W * s), round(H * s)]}
    vis, opq = a >= 16, a >= 250
    nvis = int(vis.sum())
    st['edges'] = {}
    for side, lines in {'top': a[:2, :], 'bottom': a[-2:, :], 'left': a[:, :2].T, 'right': a[:, -2:].T}.items():
        st['edges'][side] = _longest_run((lines >= 200).any(axis=0)) * s
    st['edges64'] = {}
    for side, lines in {'top': a[:3, :], 'bottom': a[-3:, :], 'left': a[:, :3].T, 'right': a[:, -3:].T}.items():
        st['edges64'][side] = _longest_run((lines >= 64).any(axis=0)) * s
    semi = (a >= 16) & (a <= 239)
    st['semi_frac'] = float(semi.sum()) / max(1, nvis)
    st['area8_over_128'] = float((a >= 8).sum()) / max(1, int((a >= 128).sum()))
    r = max(2, int(round(3 / s)))
    near_hard = cv2.dilate((a >= 128).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)) > 0
    st['broad_semi_frac'] = float((semi & ~near_hard).sum()) / max(1, nvis)
    m = (a >= 96).astype(np.uint8)
    n, lab, cst, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    areas = cst[1:, cv2.CC_STAT_AREA]
    tot = int(areas.sum())
    st['specks'] = int((areas < max(12, 0.0002 * tot)).sum())
    rr = max(4, int(round(max(0.02 * max(W, H), 6 / s))))
    md = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rr + 1,) * 2))
    nc, clab, _, _ = cv2.connectedComponentsWithStats(md, connectivity=8)
    clusters = []
    for ci in range(1, nc):
        sel = (clab == ci) & (m > 0)
        area = int(sel.sum())
        if area:
            yy, xx = np.nonzero(sel)
            clusters.append({'area': area, 'share': area / max(1, tot),
                             'bbox': [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]})
    clusters.sort(key=lambda c: -c['area'])
    st['others'] = []
    if clusters:
        mb = clusters[0]['bbox']
        for c in clusters[1:]:
            b = c['bbox']
            gap = math.hypot(max(0, mb[0] - b[2], b[0] - mb[2]), max(0, mb[1] - b[3], b[1] - mb[3]))
            st['others'].append({'share': c['share'], 'size_hero': [(b[2] - b[0]) * s, (b[3] - b[1]) * s],
                                 'gap_hero': gap * s})
    trans = (a < 8).astype(np.uint8)
    ring = (a >= 16) & (a <= 229) & (cv2.dilate(trans, np.ones((5, 5), np.uint8)) > 0)
    opf = opq.astype(np.float32)
    cnt = cv2.blur(opf, (11, 11))
    csum = cv2.blur(rgb * opf[..., None], (11, 11))
    valid = ring & (cnt > 0.02)
    st['fringe_dL'] = None
    if valid.sum() > 50:
        loc = csum[valid] / cnt[valid][:, None]
        w8 = a[valid].astype(np.float32)
        w8 = w8 * (255 - w8)
        st['fringe_dL'] = float(((_lum(rgb[valid]) - _lum(loc)) * w8).sum() / w8.sum())
    hb = max(1, int(round(2 / s)))
    band = (a >= 200) & (cv2.erode((a >= 200).astype(np.uint8), np.ones((2 * hb + 1,) * 2, np.uint8)) == 0)
    st['band_L'] = float(np.median(_lum(rgb[band]))) if band.any() else 255
    Lo = _lum(rgb[a >= 200]) if (a >= 200).any() else np.array([0])
    st['body_dark'] = float((Lo < 60).mean())
    m128 = (a >= 128).astype(np.uint8)
    dt = cv2.distanceTransform(m128, cv2.DIST_L2, 5)
    ridge = (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8))) & (m128 > 0) & (dt > 0.5)
    if ridge.any():
        sw = 2 * dt[ridge] * s
        st['stroke_p50'] = float(np.percentile(sw, 50))
        st['thin_share'] = float((sw < 1.5).mean())
    return st


def warnings(st):
    out = []
    cut = [(sd, v) for sd, v in st['edges'].items() if v >= 3]
    if cut:
        out.append('EDGE-CUT ' + ','.join(f'{sd} {r:.0f}px' for sd, r in cut))
    soft = [(sd, v) for sd, v in st['edges64'].items() if v >= 10]
    if soft:
        out.append('EDGE-SOFT ' + ','.join(f'{sd} {r:.0f}px' for sd, r in soft))
    if st['semi_frac'] >= 0.45:
        out.append(f"SEMI {st['semi_frac'] * 100:.0f}%")
    if st['area8_over_128'] >= 1.30:
        out.append(f"SOFT-AREA x{st['area8_over_128']:.2f}")
    if st['broad_semi_frac'] >= 0.01:
        out.append(f"BROAD-HAZE {st['broad_semi_frac'] * 100:.0f}%")
    if st['fringe_dL'] is not None:
        if st['fringe_dL'] >= 35:
            out.append(f"LIGHT-FRINGE +{st['fringe_dL']:.0f}")
        elif st['fringe_dL'] <= -120:
            out.append(f"DARK-FRINGE {st['fringe_dL']:.0f}")
    stray = [o for o in st['others'] if o['share'] < 0.03]
    det = [o for o in st['others'] if o['share'] >= 0.03 and o['gap_hero'] >= 20]
    if stray:
        big = max(stray, key=lambda o: o['size_hero'][0] * o['size_hero'][1])
        out.append(f"STRAY x{len(stray)} ({big['size_hero'][0]:.0f}x{big['size_hero'][1]:.0f}px)")
    if det:
        out.append(f'DETACHED x{len(det)}')
    if st['specks'] >= 15:
        out.append(f"SPECKS {st['specks']}")
    if st['body_dark'] >= 0.5 or st['band_L'] <= 60:
        out.append(f"DARK body {st['body_dark'] * 100:.0f}% band L{st['band_L']:.0f}")
    if st.get('stroke_p50', 99) < 3.5 or st.get('thin_share', 0) >= 0.2:
        out.append(f"THIN p50 {st.get('stroke_p50', 0):.1f}px thin {st.get('thin_share', 0) * 100:.0f}%")
    if st['hero_px'][0] < 120:
        out.append(f"NARROW {st['hero_px'][0]}px")
    return out


# ------------------------------------------------------------------------------------------------ hero

def fit_logo(im, bw, bh):
    f = min(bw / im.width, bh / im.height)
    size = (max(1, round(im.width * f)), max(1, round(im.height * f)))
    return im.convert('RGBa').resize(size, Image.LANCZOS).convert('RGBA')


def wash_alpha(y):
    t = y / HERO_H
    return 0.30 + 0.35 * (t / 0.5) if t <= 0.5 else 0.65 + 0.35 * ((t - 0.5) / 0.5)


def hero_bg(aid):
    p = ROOT / 'gallery' / 'bg' / f'{aid}.jpg'
    if p.exists():
        b = Image.open(p).convert('RGB')
        s = max(HERO_W / b.width, HERO_H / b.height)
        b = b.resize((round(b.width * s), round(b.height * s)), Image.LANCZOS)
        x0, y0 = (b.width - HERO_W) // 2, (b.height - HERO_H) // 2
        b = b.crop((x0, y0, x0 + HERO_W, y0 + HERO_H))
    else:
        b = Image.new('RGB', (HERO_W, HERO_H), (0x3a, 0x3a, 0x44))
    mask = Image.new('L', (1, HERO_H))
    mask.putdata([round(255 * wash_alpha(y + 0.5)) for y in range(HERO_H)])
    return Image.composite(Image.new('RGB', b.size, (0, 0, 0)), b, mask.resize((HERO_W, HERO_H)))


def _shift(img, dx, dy):
    out = Image.new(img.mode, img.size, 0)
    out.paste(img, (round(dx), round(dy)))
    return out


def with_css_shadows(layer, scale):
    A = layer.getchannel('A')
    white = Image.new('RGBA', layer.size, (255, 255, 255, 0))
    white.putalpha(_shift(A.filter(ImageFilter.GaussianBlur(1.0 * scale)), 0, 2 * scale).point(lambda v: round(v * 0.28)))
    out1 = Image.alpha_composite(white, layer)
    black = Image.new('RGBA', layer.size, (0, 0, 0, 0))
    black.putalpha(_shift(out1.getchannel('A').filter(ImageFilter.GaussianBlur(6.0 * scale)), 0, 4 * scale)
                   .point(lambda v: round(v * 0.85)))
    return Image.alpha_composite(black, out1)


def hero_panel(aid, logo, scale=TILE_SCALE, pad=PAD):
    bw, bh = round(BOX_W * scale), round(BOX_H * scale)
    pw, ph = bw + 2 * pad, bh + 2 * pad
    lg = fit_logo(logo, bw, bh)
    rx0, ry1 = LOGO_X - pad / scale, LOGO_BOTTOM + pad / scale
    reg = hero_bg(aid).crop((round(rx0), round(ry1 - ph / scale), round(rx0 + pw / scale), round(ry1))).resize((pw, ph), Image.LANCZOS)
    layer = Image.new('RGBA', (pw, ph), (0, 0, 0, 0))
    layer.paste(lg, (pad, ph - pad - lg.height))
    return Image.alpha_composite(reg.convert('RGBA'), with_css_shadows(layer, scale))


def flat(logo, w, h, color, pad=6):
    lg = fit_logo(logo, w - 2 * pad, h - 2 * pad)
    bg = Image.new('RGBA', (w, h), color)
    bg.alpha_composite(lg, ((w - lg.width) // 2, (h - lg.height) // 2))
    return bg


def _wrap(d, text, font, maxw):
    if d.textlength(text, font=font) <= maxw:
        return text
    while text and d.textlength(text + '…', font=font) > maxw:
        text = text[:-1]
    return text + '…'


def tile(aid, arr, names):
    logo = Image.fromarray(arr)
    TW, LAB = 785, 58
    strip_w, strip_h = TW - PANEL_W - 8, (PANEL_H - 6) // 2
    t = Image.new('RGBA', (TW, LAB + PANEL_H), (38, 38, 46, 255))
    d = ImageDraw.Draw(t)
    title, aka = names.get(aid, (aid, ''))
    st = stats(arr)
    d.text((4, 2), _wrap(d, f'{aid}  {title}', fB, TW - 8), font=fB, fill=(255, 255, 255))
    line2 = f"{st['W']}x{st['H']} → hero {st['hero_px'][0]}x{st['hero_px'][1]}" + (f'  ·  aka: {aka}' if aka else '')
    d.text((4, 23), _wrap(d, line2, fR, TW - 8), font=fR, fill=(190, 190, 205))
    w = warnings(st)
    d.text((4, 40), _wrap(d, '  '.join(w) if w else '—', fW, TW - 8), font=fW, fill=(255, 176, 64) if w else (120, 200, 120))
    t.alpha_composite(hero_panel(aid, logo), (0, LAB))
    t.alpha_composite(flat(logo, strip_w, strip_h, (0x0b, 0x0b, 0x0f, 255)), (PANEL_W + 8, LAB))
    t.alpha_composite(flat(logo, strip_w, strip_h, (0xf2, 0xf2, 0xf2, 255)), (PANEL_W + 8, LAB + strip_h + 6))
    return t, w


def cmd_hero(args):
    lib = '--lib' in args
    args = [a for a in args if a != '--lib']
    name, ids = args[0], args[1:]
    names = titles()
    out_dir = ROOT / 'review' / 'hero'
    out_dir.mkdir(parents=True, exist_ok=True)
    G = 10
    for si in range(0, len(ids), 12):
        tiles = []
        for aid in ids[si:si + 12]:
            tl, w = tile(aid, shipped(aid, lib), names)
            tiles.append(tl)
            print(aid, '  '.join(w) or '-')
        th, rows = tiles[0].height, (len(tiles) + 1) // 2
        sh = Image.new('RGB', (2 * tiles[0].width + 3 * G, rows * th + (rows + 1) * G), (20, 20, 24))
        for i, tl in enumerate(tiles):
            r, c = divmod(i, 2)
            sh.paste(tl.convert('RGB'), (G + c * (tl.width + G), G + r * (th + G)))
        p = out_dir / f'{name}{"-" + str(si // 12 + 1) if len(ids) > 12 else ""}.png'
        sh.save(p)
        print(p, sh.size)


# ------------------------------------------------------------------------------------------------ fix view

def _grid(img, x0, y0, x1, y1, step):
    """A grid labelled in cover fractions over an image showing the cover region [x0,x1] x [y0,y1]."""
    w, h = img.size
    over = Image.new('RGBA', img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    v = math.ceil(x0 / step - 1e-9) * step
    while v < x1:
        px = (v - x0) / (x1 - x0) * w
        d.line([(px, 0), (px, h)], fill=(255, 255, 0, 120), width=1)
        d.text((px + 2, 1), f'{v:.3f}'.rstrip('0'), font=fR, fill=(255, 255, 0, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
        v += step
    v = math.ceil(y0 / step - 1e-9) * step
    while v < y1:
        py = (v - y0) / (y1 - y0) * h
        d.line([(0, py), (w, py)], fill=(0, 255, 255, 120), width=1)
        d.text((1, py + 1), f'{v:.3f}'.rstrip('0'), font=fR, fill=(0, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
        v += step
    return Image.alpha_composite(img.convert('RGBA'), over)


def fix_block(aid, width=1100):
    d = ROOT / 'work' / 'cut' / aid
    meta = json.loads((d / 'meta.json').read_text())
    W, H = Image.open(ROOT / meta['source']).size
    cx0, cy0, cx1, cy1 = meta['crop']
    crop = Image.open(d / 'crop.png').convert('RGB')
    alpha = Image.open(d / 'alpha.png').convert('L') if (d / 'alpha.png').exists() else None
    s = width / crop.width
    size = (width, round(crop.height * s))
    crop = crop.resize(size, Image.LANCZOS)
    fx0, fy0, fx1, fy1 = cx0 / W, cy0 / H, cx1 / W, cy1 / H
    span = max(fx1 - fx0, fy1 - fy0)
    step = 0.01 if span <= 0.35 else 0.02 if span <= 0.7 else 0.05
    panels = [_grid(crop, fx0, fy0, fx1, fy1, step)]
    if alpha is not None:
        a = np.asarray(alpha.resize(size, Image.LANCZOS)).astype(np.float32)[..., None] / 255
        c = np.asarray(crop).astype(np.float32)
        kept = c * a + np.array([255, 0, 255], np.float32) * 0.18 * (1 - a) + c * 0.12 * (1 - a)
        panels.append(_grid(Image.fromarray(kept.clip(0, 255).astype(np.uint8)), fx0, fy0, fx1, fy1, step))
    logo = Image.fromarray(shipped(aid))
    lw = min(width, round(logo.width * min(1, 300 / logo.height)))
    lg = fit_logo(logo, lw, 300)
    row = Image.new('RGBA', (width, lg.height + 12), (24, 24, 30, 255))
    half = Image.new('RGBA', (width // 2, lg.height + 12), (236, 236, 230, 255))
    row.alpha_composite(half, (width // 2, 0))
    lgs = fit_logo(logo, width // 2 - 12, lg.height)
    row.alpha_composite(lgs, (6, 6))
    row.alpha_composite(lgs, (width // 2 + 6, 6))
    panels.append(row)
    names = titles()
    head = 22
    out = Image.new('RGB', (width, head + sum(p.height + 6 for p in panels)), (40, 40, 46))
    ImageDraw.Draw(out).text((4, 2), f'{aid}  {names.get(aid, ("", ""))[0][:70]}  · {meta["source"].split("/")[-1]} {json.dumps(meta["pick"].get("src"))}',
                             font=fW, fill=(255, 255, 255))
    y = head
    for p in panels:
        out.paste(p.convert('RGB'), (0, y))
        y += p.height + 6
    return out


def cmd_fix(args):
    width = 1100
    if args and args[0].startswith('w='):
        width, args = int(args[0][2:]), args[1:]
    out_dir = ROOT / 'review' / 'fix'
    out_dir.mkdir(parents=True, exist_ok=True)
    for aid in args:
        p = out_dir / f'{aid}.png'
        img = fix_block(aid, width)
        img.save(p)
        print(p, img.size)
