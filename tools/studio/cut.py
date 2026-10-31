"""Step 2: cut the chosen logo out of its cover.

For every pick in picks.json:
  1. crop the union of the chosen boxes plus a margin, from the chosen source cover;
  2. upscale the crop with Real-ESRGAN (anime 6B) so the logo comes out about 1600 px wide;
  3. the soft alpha, kept SOFT, one of two ways:
     - "strokes" (default): Hi-SAM text strokes are the sure logo, a band around them (outlines, shadows,
       anything an include polygon names) is unknown, the rest is background; ViTMatte solves the band.
       This holds on busy art, where Lucida marks the whole crop.
     - "lucida": Lucida (BiRefNet fine-tuned for text and logos) on the crop at its 1024 training size,
       optionally refined by ViTMatte. Best on plain backgrounds and for emblems far from any lettering;
  5. keep what lies inside the chosen boxes (a part mostly inside is kept whole);
  6. erase / restore polygons from the pick;
  7. foreground colour estimation, so soft edges carry the logo's colour and not the cover's;
  8. trim and cap at 1600 px wide.

A pick (coordinates are fractions of the source cover):
  {"src": "A"|"M1"|"M2"|"S", "boxes": [proposal numbers], "box": [[x0,y0,x1,y1], ...],
   "lang": "en", "script": "Latn", "alpha": "strokes"|"lucida", "band": 0.035, "include": [polygons],
   "erase": [[[x,y], ...], ...], "restore": [...], "refine": true, "margin": 0.08, "keep": 0.85, "max_w": 1600,
   "seed_lum": [{"poly": [[x,y], ...], "min": 0.85}],   (extra sure strokes: pixels this bright, or "max": this dark,
                                                          "sat": 0.5 only colours at least that saturated)
   "erase_lum": [{"poly": [...], "max": 0.3}]}          (fade out pixels this dark, or "min": this light, in a polygon;
                                                          "grey": 0.2 limits it to colours less saturated than that,
                                                          "sat": 0.35 to colours more saturated than that,
                                                          "hue": [270, 350] to those hues)
  "solid" also takes {"poly": [...], "from": [x, y]}: a stroke the cover hides, rebuilt in the colour at `from`.
  "restore" widens what counts as inside without making it unknown; "include" makes a polygon unknown (ViTMatte
  decides it whole); "solid_fg" forces sure logo into the trimap; "solid" sets alpha 1 after matting.

Usage: cut.py [ids...]   (default: every pick without a current result)
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import studio as S  # noqa: E402
from keys import EXPORT_ONLY, pick_key  # noqa: E402,F401
from propose import candidates  # noqa: E402

ROOT = S.ROOT
OUT = ROOT / 'work' / 'cut'
TARGET_W = 1800


def pick_boxes(aid, pick):
    boxes = [list(b) for b in pick.get('box', [])]
    if pick.get('boxes'):
        prop = json.loads((ROOT / 'work' / 'prop' / aid / f"{pick['src']}.json").read_text())
        by_n = {r['n']: r['box'] for r in prop['regions']}
        boxes += [by_n[n] for n in pick['boxes']]
    if not boxes:
        raise ValueError(f'{aid}: no boxes')
    return boxes


def poly_mask(polys, shape, to_px):
    m = np.zeros(shape, np.uint8)
    for poly in polys or []:
        pts = np.array([to_px(x, y) for x, y in poly], np.int32)
        cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


CACHE = ROOT / 'work' / 'cache'


def cached(aid, name, key, make, kind):
    """The upscale and the model outputs depend only on the crop (and the trimap), so a re-cut that only moves a
    polygon reuses them: seconds instead of minutes."""
    d = CACHE / aid
    path = d / f'{name}-{hashlib.sha256(key).hexdigest()[:16]}.{"png" if kind != "npy" else "npy"}'
    if path.exists():
        if kind == 'npy':
            return np.load(path).astype(np.float32)
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if kind == 'rgb' else img.astype(np.float32) / 65535
    out = make()
    d.mkdir(parents=True, exist_ok=True)
    if kind == 'npy':  # return what a later run will read back, so a cached re-cut matches the first one exactly
        out = out.astype(np.float16)
        np.save(path, out)
        return out.astype(np.float32)
    if kind == 'rgb':
        cv2.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        return out
    out = (np.clip(out, 0, 1) * 65535).round().astype(np.uint16)
    cv2.imwrite(str(path), out)
    return out.astype(np.float32) / 65535


def official(aid, pick):
    """An official logo file we already hold (e.g. the Wikimedia Commons logo the server found): trim and cap."""
    rgba = S.trim(np.asarray(Image.open(ROOT / pick['official']).convert('RGBA')))
    max_w = pick.get('max_w', 1600)
    if rgba.shape[1] > max_w:
        rgba = S.rgba_resize(rgba, (max_w, max(1, round(rgba.shape[0] * max_w / rgba.shape[1]))))
    d = OUT / aid
    d.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba).save(d / 'logo.png')
    (d / 'meta.json').write_text(json.dumps({'pick': pick, 'key': pick_key(pick), 'source': pick['official'],
                                             'w': int(rgba.shape[1]), 'h': int(rgba.shape[0])}))
    return rgba.shape


def cut(aid, pick):
    if pick.get('official'):
        return official(aid, pick)
    src = dict(candidates(aid))[pick['src']]
    rgb = S.load_rgb(src)
    H, W = rgb.shape[:2]
    boxes = pick_boxes(aid, pick)
    ux0, uy0 = min(b[0] for b in boxes) * W, min(b[1] for b in boxes) * H
    ux1, uy1 = max(b[2] for b in boxes) * W, max(b[3] for b in boxes) * H
    m = max(10, pick.get('margin', 0.08) * max(ux1 - ux0, uy1 - uy0))
    cx0, cy0 = int(max(0, ux0 - m)), int(max(0, uy0 - m))
    cx1, cy1 = int(min(W, ux1 + m)), int(min(H, uy1 + m))
    crop = rgb[cy0:cy1, cx0:cx1]
    cw, ch = cx1 - cx0, cy1 - cy0
    # Work at about the output size: ~1800 px across and at most 1.6 MP, since ViTMatte's global attention grows
    # with the square of the pixel count (a 9 MP crop takes many minutes).
    factor = min(TARGET_W / cw, 1800 / ch, (1.6e6 / (cw * ch)) ** 0.5, 4.0)
    if factor < 1:  # a huge cover: work at about the output size, matting a 4800 px crop costs minutes for nothing
        up = cv2.resize(crop, (max(1, round(cw * factor)), max(1, round(ch * factor))), interpolation=cv2.INTER_AREA)
    else:
        if factor < 1.6:  # a crop this wide already has the detail; Real-ESRGAN would only cost minutes
            factor = 1.0
        up = cached(aid, 'up', f'{src}|{cx0},{cy0},{cx1},{cy1}|{factor:.4f}'.encode(),
                    lambda: S.upscale(crop, factor), 'rgb')
    up_key = hashlib.sha256(up.tobytes()).digest()
    uh, uw = up.shape[:2]
    fx, fy = uw / cw, uh / ch

    def to_px(x, y):  # cover fraction -> upscaled-crop pixel
        return (x * W - cx0) * fx, (y * H - cy0) * fy

    boxmask = np.zeros(up.shape[:2], np.uint8)
    pad = max(4, int(0.012 * max(uw, uh)))
    for b in boxes:
        x0, y0 = to_px(b[0], b[1])
        x1, y1 = to_px(b[2], b[3])
        cv2.rectangle(boxmask, (int(x0) - pad, int(y0) - pad), (int(x1) + pad, int(y1) + pad), 1, -1)
    alpha_shape = up.shape[:2]
    inc = pick.get('include')
    inc = [inc] if inc == 'boxes' else inc or []  # "boxes" makes every chosen box unknown, beside any polygons
    include = poly_mask([q for q in inc if q != 'boxes'], alpha_shape, to_px)
    if 'boxes' in inc:
        include |= boxmask > 0
    restore = poly_mask(pick.get('restore'), alpha_shape, to_px)
    inside = (boxmask > 0) | restore | include

    lum = up.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32) / 255
    mode = pick.get('alpha', 'strokes')
    if mode == 'lucida':
        alpha = cached(aid, 'lucida', up_key, lambda: S.lucida_alpha(up, (1024, 1024)), 'npy')
        raw = alpha.copy()
        if pick.get('refine'):
            fg = cv2.erode((alpha > 0.96).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            bg = cv2.erode((alpha < 0.04).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            trimap = np.full(alpha.shape, 128, np.uint8)
            trimap[bg] = 0
            trimap[fg] = 255
            alpha = cached(aid, 'vitmatte', up_key + trimap.tobytes(), lambda: S.vitmatte(up, trimap), 'alpha')
    else:
        # Hi-SAM strokes are the sure logo; a band around them (outlines, shadows, a frame named by an
        # include polygon) is left for ViTMatte; everything further away is background, however busy.
        logits = cached(aid, 'hisam', up_key, lambda: S.hisam_logits(up), 'npy')
        raw = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
        strokes = (logits > pick.get('stroke_logit', 0.0)) & inside
        if mode == 'hybrid':  # ornate numerals and emblems Hi-SAM misses: add Lucida's most certain logo pixels
            luc = cached(aid, 'lucida', up_key, lambda: S.lucida_alpha(up, (1024, 1024)), 'npy')
            zone = poly_mask(pick['seed_poly'], alpha_shape, to_px) if pick.get('seed_poly') else True
            strokes |= (luc > pick.get('lucida_sure', 0.97)) & inside & zone
        for seed in pick.get('seed_lum', []):  # strokes Hi-SAM missed, picked by brightness inside a polygon
            sel = (lum >= seed.get('min', 0.0)) & (lum <= seed.get('max', 1.0))
            if 'sat' in seed:  # and by colour: a red stroke over grey art
                sel &= (up.max(axis=2) - up.min(axis=2)) >= seed['sat'] * np.maximum(up.max(axis=2), 1)
            strokes |= sel & inside & poly_mask([seed['poly']], alpha_shape, to_px)
        band_px = max(6, int(round(pick.get('band', 0.035) * min(uh, uw))))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1,) * 2)
        band = (cv2.dilate(strokes.astype(np.uint8), ker) > 0) | include
        core = cv2.erode(strokes.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        if pick.get('solid_fg'):
            core |= poly_mask(pick.get('solid_fg'), alpha_shape, to_px)
        trimap = np.zeros(up.shape[:2], np.uint8)
        trimap[band & inside] = 128
        trimap[core] = 255
        (OUT / aid).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(OUT / aid / 'trimap.png'), trimap)
        alpha = cached(aid, 'vitmatte', up_key + trimap.tobytes(), lambda: S.vitmatte(up, trimap), 'alpha')
        alpha[trimap == 0] = 0

    # Keep the parts inside the chosen boxes; a component that is mostly inside is kept whole, so a letter
    # crossing the box edge is not clipped, while a badge or frame that touches the logo is cut at the edge.
    soft_inside = cv2.GaussianBlur(inside.astype(np.float32), (0, 0), max(1.0, pad / 3))
    solid = alpha > 0.12
    n, lab = cv2.connectedComponents(solid.astype(np.uint8), connectivity=8)
    mass_in = np.bincount(lab.ravel(), weights=(alpha * inside).ravel(), minlength=n)
    mass = np.bincount(lab.ravel(), weights=alpha.ravel(), minlength=n)
    whole_ids = np.nonzero(mass_in >= pick.get('keep', 0.85) * np.maximum(mass, 1e-6))[0]
    whole = np.isin(lab, whole_ids[whole_ids > 0])
    halo = max(2, int(round(uw / 500)))
    whole = cv2.dilate(whole.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * halo + 1,) * 2))
    alpha = alpha * np.maximum(soft_inside, cv2.GaussianBlur(whole.astype(np.float32), (0, 0), halo / 2))
    erase = poly_mask(pick.get('erase'), alpha.shape, to_px)
    alpha[erase] = 0
    for e in pick.get('erase_lum', []):  # background showing through a letter: fade out what is this dark (or light)
        zone = poly_mask([e['poly']], alpha.shape, to_px)
        keep = np.ones_like(alpha)
        if 'max' in e:
            keep = np.minimum(keep, np.clip((lum - e['max']) / 0.08, 0, 1))
        if 'min' in e:
            keep = np.minimum(keep, np.clip((e['min'] - lum) / 0.08, 0, 1))
        sat = (up.max(axis=2) - up.min(axis=2)) / np.maximum(up.max(axis=2), 1).astype(np.float32)
        if 'grey' in e:  # only greys go (a white haze), coloured lettering of the same lightness stays
            keep = np.maximum(keep, np.clip((sat - e['grey']) / 0.08, 0, 1))
        if 'sat' in e:  # only colours go (an orange glow), grey lettering and cream ribbons stay
            keep = np.maximum(keep, np.clip((e['sat'] - sat) / 0.08, 0, 1))
        if 'hue' in e:  # only these hues go, in degrees (pink hair behind a gold emblem)
            hue = cv2.cvtColor(up, cv2.COLOR_RGB2HSV_FULL)[..., 0].astype(np.float32) * 360 / 256
            h0, h1 = e['hue']
            keep = np.maximum(keep, (~((hue >= h0) & (hue <= h1))).astype(np.float32))
        alpha[zone] *= keep[zone]
    if pick.get('min_part'):  # drop specks, and what an erase left: parts under this share of the logo's mass
        n, lab = cv2.connectedComponents((alpha > 0.12).astype(np.uint8), connectivity=8)
        mass = np.bincount(lab.ravel(), weights=alpha.ravel(), minlength=n)
        small = np.nonzero(mass < pick['min_part'] * mass[1:].sum())[0]
        alpha[np.isin(lab, small[small > 0])] = 0
    solids = pick.get('solid') or []
    paint = up.copy() if any(isinstance(sp, dict) for sp in solids) else up
    for sp in solids:
        m = poly_mask([sp['poly'] if isinstance(sp, dict) else sp], alpha.shape, to_px)
        alpha[m] = 1
        if isinstance(sp, dict):  # a stroke the cover hides or breaks, rebuilt in the logo's colour sampled at `from`
            px_, py_ = to_px(*sp['from'])
            paint[m] = up[min(uh - 1, max(0, int(py_))), min(uw - 1, max(0, int(px_)))]

    fgc = S.estimate_foreground(paint, alpha)
    (OUT / aid).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / aid / 'alpha.png'), (alpha * 255).round().astype(np.uint8))  # untrimmed, for the fix view
    rgba = np.dstack([fgc, (alpha * 255).round().astype(np.uint8)])
    rgba = S.trim(rgba)
    max_w = pick.get('max_w', 1600)
    if rgba.shape[1] > max_w:
        rgba = S.rgba_resize(rgba, (max_w, max(1, round(rgba.shape[0] * max_w / rgba.shape[1]))))
    d = OUT / aid
    d.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba).save(d / 'logo.png')
    cv2.imwrite(str(d / 'raw-alpha.png'), (raw * 255).round().astype(np.uint8))
    cv2.imwrite(str(d / 'crop.png'), cv2.cvtColor(up, cv2.COLOR_RGB2BGR))
    (d / 'meta.json').write_text(json.dumps({'pick': pick, 'key': pick_key(pick), 'source': str(src.relative_to(ROOT)),
                                             'crop': [cx0, cy0, cx1, cy1], 'factor': round(factor, 3),
                                             'w': int(rgba.shape[1]), 'h': int(rgba.shape[0])}))
    return rgba.shape


def main(argv):
    picks = json.loads((ROOT / 'picks.json').read_text())
    ids = argv or list(picks)
    for aid in ids:
        pick = picks.get(aid)
        if not pick or pick.get('none'):
            continue
        meta = OUT / aid / 'meta.json'
        if not argv and meta.exists() and json.loads(meta.read_text()).get('key') == pick_key(pick):
            continue
        t = time.time()
        try:
            shape = cut(aid, pick)
            print(aid, pick.get('src', 'official'), f'{shape[1]}x{shape[0]}', f'{time.time() - t:.0f}s', flush=True)
        except Exception as e:  # one bad pick must not stop the batch
            print(aid, 'FAILED', repr(e), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
