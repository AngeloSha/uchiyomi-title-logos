"""Contact sheets for looking at many covers or logos in one image.

  sheets.py pick <first> <count> [ids...]   every candidate cover per series, Hi-SAM line boxes numbered, 10% grid
  sheets.py zoom <aid> <key>                one cover, large, with its numbered boxes
  sheets.py review <name> <ids...>          logos at hero size: darkened banner, solid dark, light
  sheets.py hero <name> [--lib] <ids...>    the app's hero conditions, as the independent review drew them (hero.py)
  sheets.py fix [w=1100] <ids...>           source crop with a cover-fraction grid, the kept alpha, the shipped logo
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import studio as S  # noqa: E402

ROOT = S.ROOT
META = json.loads((ROOT / 'sources' / 'anilist.json').read_text())
COLORS = [(255, 59, 48), (52, 199, 89), (0, 122, 255), (255, 204, 0), (175, 82, 222), (255, 149, 0), (0, 199, 190),
          (255, 45, 85)]


def font(size):
    return ImageFont.load_default(size=size)


def name(aid):
    t = META[aid]['title']
    return (t.get('english') or t.get('romaji') or '?')[:44]


def ids_all():
    return sorted(META, key=int)


def proposals(aid, key):
    p = ROOT / 'work' / 'prop' / aid / f'{key}.json'
    return json.loads(p.read_text()) if p.exists() else None


def draw_cover(aid, key, path, height, grid=True, label_size=11):
    rgb = S.load_rgb(path)
    h, w = rgb.shape[:2]
    width = max(1, round(w * height / h))
    img = Image.fromarray(cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)).convert('RGBA')
    over = Image.new('RGBA', img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    if grid:
        for i in range(1, 10):
            x, y = round(width * i / 10), round(height * i / 10)
            d.line([(x, 0), (x, height)], fill=(255, 255, 255, 70 if i != 5 else 120), width=1)
            d.line([(0, y), (width, y)], fill=(255, 255, 255, 70 if i != 5 else 120), width=1)
    prop = proposals(aid, key)
    f = font(label_size)
    for r in (prop or {}).get('regions', []):
        c = COLORS[(r['n'] - 1) % len(COLORS)]
        x0, y0, x1, y1 = r['box']
        box = [x0 * width, y0 * height, x1 * width, y1 * height]
        d.rectangle(box, outline=c + (255,), width=2)
        tag = str(r['n'])
        tw = d.textlength(tag, font=f)
        tx, ty = box[0], box[1] - label_size - 2 if box[1] > label_size + 2 else box[1]
        d.rectangle([tx, ty, tx + tw + 4, ty + label_size + 2], fill=c + (255,))
        d.text((tx + 2, ty), tag, font=f, fill=(0, 0, 0, 255))
    img = Image.alpha_composite(img, over).convert('RGB')
    cap = f'{key} · {w}×{h}' + ('' if prop else ' · no boxes')
    canvas = Image.new('RGB', (width, height + 14), (24, 24, 28))
    canvas.paste(img, (0, 14))
    ImageDraw.Draw(canvas).text((2, 1), cap, font=font(11), fill=(230, 230, 230))
    return canvas


def series_block(aid, height):
    from propose import candidates
    covers = [draw_cover(aid, key, path, height) for key, path in candidates(aid)]
    gap = 6
    width = sum(c.width for c in covers) + gap * (len(covers) - 1)
    head = 18
    block = Image.new('RGB', (max(width, 300), height + 14 + head), (40, 40, 46))
    ImageDraw.Draw(block).text((3, 2), f'{aid}  {name(aid)}  [{META[aid]["countryOfOrigin"]}]', font=font(13),
                               fill=(255, 255, 255))
    x = 0
    for c in covers:
        block.paste(c, (x, head))
        x += c.width + gap
    return block


def pack(blocks, max_width, out, bg=(16, 16, 18), gap=8):
    rows, row, w = [], [], 0
    for b in blocks:
        if row and w + gap + b.width > max_width:
            rows.append(row)
            row, w = [], 0
        row.append(b)
        w += b.width + (gap if len(row) > 1 else 0)
    if row:
        rows.append(row)
    W = max(sum(b.width for b in r) + gap * (len(r) - 1) for r in rows)
    H = sum(max(b.height for b in r) for r in rows) + gap * (len(rows) - 1)
    sheet = Image.new('RGB', (W, H), bg)
    y = 0
    for r in rows:
        x = 0
        for b in r:
            sheet.paste(b, (x, y))
            x += b.width + gap
        y += max(b.height for b in r) + gap
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return sheet.size


def cmd_pick(args):
    first, count = int(args[0]), int(args[1])
    ids = args[2:] or ids_all()[first:first + count]
    blocks = [series_block(aid, 230) for aid in ids]
    out = ROOT / 'review' / 'pick' / (f'pick-{first:03d}.png' if not args[2:] else f'pick-{"-".join(ids)[:60]}.png')
    print(out, pack(blocks, 1500, out))


def cmd_zoom(args):
    from propose import candidates
    aid, key = args[0], args[1]
    path = dict(candidates(aid))[key]
    height = int(args[2]) if len(args) > 2 else 800
    out = ROOT / 'review' / 'zoom' / f'{aid}-{key}.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    img = draw_cover(aid, key, path, height, label_size=16)
    img.save(out)
    print(out, img.size)

def cmd_region(args):
    """Part of a cover, large, with a labelled grid in cover fractions: region <height> aid:key:x0,y0,x1,y1 ..."""
    from propose import candidates
    height, blocks = int(args[0]), []
    for spec in args[1:]:
        aid, key, box = spec.split(':')
        x0, y0, x1, y1 = map(float, box.split(','))
        rgb = S.load_rgb(dict(candidates(aid))[key])
        H, W = rgb.shape[:2]
        crop = rgb[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)]
        w = round(crop.shape[1] * height / crop.shape[0])
        img = Image.fromarray(cv2.resize(crop, (w, height), interpolation=cv2.INTER_AREA if crop.shape[0] > height else cv2.INTER_CUBIC)).convert('RGBA')
        over = Image.new('RGBA', img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(over)
        step = 0.02 if max(x1 - x0, y1 - y0) <= 0.4 else 0.05
        f = font(11)
        v = math.ceil(x0 / step) * step
        while v < x1:
            px = (v - x0) / (x1 - x0) * w
            d.line([(px, 0), (px, height)], fill=(255, 255, 0, 110), width=1)
            d.text((px + 2, 2), f'{v:.2f}', font=f, fill=(255, 255, 0, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
            v += step
        v = math.ceil(y0 / step) * step
        while v < y1:
            py = (v - y0) / (y1 - y0) * height
            d.line([(0, py), (w, py)], fill=(0, 255, 255, 110), width=1)
            d.text((2, py + 2), f'{v:.2f}', font=f, fill=(0, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
            v += step
        img = Image.alpha_composite(img, over).convert('RGB')
        b = Image.new('RGB', (img.width, img.height + 16), (40, 40, 46))
        ImageDraw.Draw(b).text((3, 1), f'{aid} {key} {name(aid)[:40]}', font=font(12), fill=(255, 255, 255))
        b.paste(img, (0, 16))
        blocks.append(b)
    out = ROOT / 'review' / 'zoom' / f'region-{args[1].split(":")[0]}.png'
    print(out, pack(blocks, 1500, out))


def cmd_zooms(args):
    """Several covers large, one sheet: zooms <height> aid:key [aid:key ...]"""
    from propose import candidates
    height, blocks = int(args[0]), []
    for spec in args[1:]:
        aid, key = spec.split(':')
        img = draw_cover(aid, key, dict(candidates(aid))[key], height, label_size=14)
        b = Image.new('RGB', (img.width, img.height + 16), (40, 40, 46))
        ImageDraw.Draw(b).text((3, 1), f'{aid} {name(aid)[:30]}', font=font(12), fill=(255, 255, 255))
        b.paste(img, (0, 16))
        blocks.append(b)
    out = ROOT / 'review' / 'zoom' / f'zooms-{args[1].replace(":", "")}.png'
    print(out, pack(blocks, 1400, out))


def cmd_boxes(args):
    """The chosen cover of every pick still waiting for its boxes: boxes <first> <count> [height]"""
    from propose import candidates
    picks = json.loads((ROOT / 'picks.json').read_text())
    todo = sorted((a for a, v in picks.items() if v.get('todo') == 'box'), key=int)
    first, count = int(args[0]), int(args[1])
    height = int(args[2]) if len(args) > 2 else 400
    blocks = []
    for aid in todo[first:first + count]:
        key = picks[aid]['src']
        img = draw_cover(aid, key, dict(candidates(aid))[key], height, label_size=13)
        b = Image.new('RGB', (img.width, img.height + 16), (40, 40, 46))
        ImageDraw.Draw(b).text((3, 1), f'{aid} {name(aid)[:int(img.width / 6.5) - 8]}', font=font(12), fill=(255, 255, 255))
        b.paste(img, (0, 16))
        blocks.append(b)
    out = ROOT / 'review' / 'boxes' / f'boxes-{first:03d}.png'
    print(out, len(todo), pack(blocks, 1500, out))


# ------------------------------------------------------------------------------------------------- review


def banner_bg(aid, size):
    w, h = size
    for p in sorted((ROOT / 'sources' / 'banner').glob(f'{aid}.*')) + sorted((ROOT / 'sources' / 'anilist').glob(f'{aid}.*')):
        rgb = S.load_rgb(p)
        ih, iw = rgb.shape[:2]
        s = max(w / iw, h / ih)
        rgb = cv2.resize(rgb, (round(iw * s) + 1, round(ih * s) + 1), interpolation=cv2.INTER_AREA)
        y0 = (rgb.shape[0] - h) // 3 if 'banner' in str(p) else (rgb.shape[0] - h) // 4
        x0 = (rgb.shape[1] - w) // 2
        rgb = rgb[y0:y0 + h, x0:x0 + w].astype(np.float32)
        # The hero's treatment: darker, a little blurred, heavier on the left where the title sits.
        rgb = cv2.GaussianBlur(rgb, (0, 0), 2)
        ramp = np.linspace(0.30, 0.55, w)[None, :, None]
        return (rgb * ramp).clip(0, 255).astype(np.uint8)
    return np.full((h, w, 3), 30, np.uint8)


def place(rgba, bg, box_w, box_h, align='left', hero=False):
    """Fit the logo in box_w x box_h on bg. hero=True adds the app's two drop shadows
    (.hero-title-logo: 0 2px 2px #ffffff47, 0 4px 12px #000000d9)."""
    lh, lw = rgba.shape[:2]
    s = min(box_w / lw, box_h / lh)
    logo = S.rgba_resize(rgba, (max(1, round(lw * s)), max(1, round(lh * s))))
    H, W = bg.shape[:2]
    y = (H - logo.shape[0]) // 2
    x = (W - logo.shape[1]) // 2 if align == 'center' else max(8, (W - box_w) // 2)
    out = bg.astype(np.float32)
    if hero:
        a = np.zeros((H, W), np.float32)
        a[y:y + logo.shape[0], x:x + logo.shape[1]] = logo[..., 3] / 255
        for (dy, sigma, color, opacity) in ((2, 1.0, 255, 0.28), (4, 6.0, 0, 0.85)):
            sh = cv2.GaussianBlur(np.roll(a, dy, axis=0), (0, 0), sigma)[..., None] * opacity
            out = out * (1 - sh) + color * sh
    out = out.clip(0, 255).astype(np.uint8)
    out[y:y + logo.shape[0], x:x + logo.shape[1]] = S.composite(logo, out[y:y + logo.shape[0], x:x + logo.shape[1]])
    return out


def review_tile(aid, logo_path, tw=500, th=210):
    rgba = np.asarray(Image.open(logo_path).convert('RGBA'))
    hero = place(rgba, banner_bg(aid, (tw, th)), int(tw * 0.86), int(th * 0.72), hero=True)
    sw, sh = tw // 2, 104
    dark = place(rgba, np.full((sh, sw, 3), (14, 15, 19), np.uint8), sw - 20, sh - 20, 'center')
    light = place(rgba, np.full((sh, sw, 3), (238, 238, 232), np.uint8), sw - 20, sh - 20, 'center')
    tile = Image.new('RGB', (tw, th + sh + 16), (40, 40, 46))
    ImageDraw.Draw(tile).text((3, 1), f'{aid}  {name(aid)}', font=font(12), fill=(255, 255, 255))
    tile.paste(Image.fromarray(hero), (0, 16))
    tile.paste(Image.fromarray(dark), (0, th + 16))
    tile.paste(Image.fromarray(light), (sw, th + 16))
    return tile


def cmd_review(args):
    sheet_name, ids = args[0], args[1:]
    tiles = []
    for aid in ids:
        p = ROOT / 'work' / 'cut' / aid / 'logo.png'
        if p.exists():
            tiles.append(review_tile(aid, p))
    out = ROOT / 'review' / 'sheets' / f'{sheet_name}.png'
    print(out, pack(tiles, 1530, out))


def cmd_pending(args):
    """Review sheets (12 a sheet) for every cut without a verdict for its current version: pending [max_sheets]"""
    verdicts = json.loads((ROOT / 'review.json').read_text()) if (ROOT / 'review.json').exists() else {}
    picks = json.loads((ROOT / 'picks.json').read_text())
    ids = []
    for meta in sorted((ROOT / 'work' / 'cut').glob('*/meta.json'), key=lambda m: int(m.parent.name)):
        aid, key = meta.parent.name, json.loads(meta.read_text())['key']
        if picks.get(aid, {}).get('none') or verdicts.get(aid, {}).get('key') == key:
            continue
        ids.append(aid)
    limit = int(args[0]) if args else 99
    for n in range(min(limit, (len(ids) + 11) // 12)):
        chunk = ids[n * 12:(n + 1) * 12]
        cmd_review([f'pending-{n + 1:02d}'] + chunk)
        print(' '.join(chunk))


if __name__ == '__main__':
    if sys.argv[1] in ('hero', 'fix'):
        import hero
        {'hero': hero.cmd_hero, 'fix': hero.cmd_fix}[sys.argv[1]](sys.argv[2:])
    else:
        {'pick': cmd_pick, 'zoom': cmd_zoom, 'zooms': cmd_zooms, 'region': cmd_region, 'boxes': cmd_boxes, 'review': cmd_review, 'pending': cmd_pending}[sys.argv[1]](sys.argv[2:])
