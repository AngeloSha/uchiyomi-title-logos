"""Step 5: the owner's gallery: every exported logo on a hero-like background, with a Redo mark per logo.

Writes gallery/index.html, gallery/logos/<id>.webp (the library files themselves) and gallery/bg/<id>.jpg
(the series banner, or its cover when there is no banner, blurred a little like the hero).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import studio as S  # noqa: E402

ROOT = S.ROOT
OUT = ROOT / 'gallery'
PREVIEW_W = 960
LANG = {'en': 'English', 'ko': 'Korean', 'ja': 'Japanese', 'zh': 'Chinese'}


def background(aid):
    banner = sorted((ROOT / 'sources' / 'banner').glob(f'{aid}.*'))
    path = banner[0] if banner else sorted((ROOT / 'sources' / 'anilist').glob(f'{aid}.*'))[0]
    rgb = S.load_rgb(path)
    w, h = 960, 420
    ih, iw = rgb.shape[:2]
    s = max(w / iw, h / ih)
    rgb = cv2.resize(rgb, (round(iw * s) + 1, round(ih * s) + 1), interpolation=cv2.INTER_AREA)
    y0 = (rgb.shape[0] - h) // (3 if banner else 4)
    x0 = (rgb.shape[1] - w) // 2
    rgb = cv2.GaussianBlur(rgb[y0:y0 + h, x0:x0 + w], (0, 0), 1.6 if banner else 3)
    cv2.imwrite(str(OUT / 'bg' / f'{aid}.jpg'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])


def main():
    meta = json.loads((ROOT / 'sources' / 'anilist.json').read_text())
    series = {s['anilist_id']: s for s in json.loads((ROOT / 'sources' / 'series.json').read_text()) if s['anilist_id']}
    index = json.loads((ROOT / 'library' / 'index.json').read_text())
    picks = json.loads((ROOT / 'picks.json').read_text())
    for d in ('logos', 'bg'):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    logos, missing = [], []
    for aid in sorted(meta, key=lambda a: (series[a]['title'] or '').lower()):
        title = series[aid]['title'] or meta[aid]['title']['romaji']
        if aid in index:
            e = index[aid]
            img = Image.open(ROOT / 'library' / e['file']).convert('RGBA')
            if img.width > PREVIEW_W:  # the hero shows at most 640 css px; 960 keeps it sharp at 1.5x and the page light
                rgba = S.rgba_resize(np.asarray(img), (PREVIEW_W, max(1, round(img.height * PREVIEW_W / img.width))))
                img = Image.fromarray(rgba)
            img.save(OUT / 'logos' / f'{aid}.webp', 'WEBP', lossless=True, quality=100, method=6)
            background(aid)
            t = meta[aid]['title']
            official = t.get('english') or t.get('romaji') or ''
            logos.append({'id': aid, 'title': title, 'aka': official if official.lower() != title.lower() else '',
                          'w': e['w'], 'h': e['h'], 'lang': e['lang'],
                          'script': e['script'], 'source': e['source']})
        else:
            missing.append({'id': aid, 'title': title, 'why': (picks.get(aid) or {}).get('none') or 'not finished'})
    unlinked = sorted((s['title'] for s in json.loads((ROOT / 'sources' / 'series.json').read_text())
                       if not s['anilist_id']), key=str.lower)
    data = {'logos': logos, 'missing': missing, 'unlinked': unlinked, 'langs': LANG}
    page = (ROOT / 'src' / 'gallery.html').read_text()
    blob = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    page = page.replace('__DATA__', blob)
    (OUT / 'index.html').write_text(page)
    print('gallery:', len(logos), 'logos,', len(missing), 'without,', len(unlinked), 'unlinked')


if __name__ == '__main__':
    main()
