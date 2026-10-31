"""Step 1: numbered region proposals for every candidate cover.

Hi-SAM (the text-stroke model of services/title-extractor) runs once on the whole cover; its strokes are joined
into line boxes the same way the worker does, each becoming a numbered box. Lucida is not used here: on a
full-bleed cover it marks the characters as well as the lettering. Results: work/prop/<anilistId>/<source>.json
plus the stroke logits as an 8-bit PNG (128 = logit 0).

Usage: propose.py [--only anilist|mangadex|stored] [ids...]
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import studio as S  # noqa: E402

SRC = S.ROOT / 'sources'
OUT = S.ROOT / 'work' / 'prop'
WORK_H = 1000


def candidates(aid):
    """Every cover we hold for one AniList id, as (key, path)."""
    out = []
    for p in sorted((SRC / 'anilist').glob(f'{aid}.*')):
        out.append(('A', p))
    mdx = json.loads((SRC / 'mangadex.json').read_text()) if (SRC / 'mangadex.json').exists() else {}
    for n, c in enumerate((mdx.get(aid) or {}).get('covers', []), 1):
        p = SRC / 'mangadex' / c['file']
        if p.exists():
            out.append((f'M{n}', p))
    for p in sorted((SRC / 'stored').glob(f'{aid}.*')):
        out.append(('S', p))
    return out


def main(argv):
    only = None
    if '--only' in argv:
        only = argv[argv.index('--only') + 1]
        argv = [a for a in argv if a not in ('--only', only)]
    ids = argv or sorted({p.stem for p in (SRC / 'anilist').glob('*.*')}, key=int)
    for spec in ids:
        aid, _, want = spec.partition(':')  # "aid" or "aid:key" for one cover
        for key, path in candidates(aid):
            if want and key != want:
                continue
            if only and {'anilist': 'A', 'mangadex': 'M', 'stored': 'S'}[only] != key[0]:
                continue
            dest = OUT / aid / f'{key}.json'
            if dest.exists():
                continue
            t = time.time()
            rgb = S.load_rgb(path)
            h, w = rgb.shape[:2]
            if max(h, w) > 1600:  # Hi-SAM works at 1024 anyway; full-size grouping on a 26 MP cover takes minutes
                s = 1600 / max(h, w)
                rgb = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            logits = S.hisam_logits(rgb)
            strokes = logits > 0
            dest.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dest.with_suffix('.png')), (np.clip(logits * 16 + 128, 0, 255)).astype(np.uint8))
            sh, sw = strokes.shape
            lines = sorted(S.line_groups(strokes), key=lambda g: (round(g[0][1] / sh, 2), g[0][0]))
            regions = [{'n': i + 1, 'box': [round(b[0] / sw, 4), round(b[1] / sh, 4), round(b[2] / sw, 4), round(b[3] / sh, 4)],
                        'ink': ink} for i, (b, ink) in enumerate(lines[:40])]
            dest.write_text(json.dumps({'source': str(path.relative_to(S.ROOT)), 'w': w, 'h': h, 'regions': regions}))
            print(aid, key, f'{w}x{h}', f'{time.time() - t:.1f}s', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
