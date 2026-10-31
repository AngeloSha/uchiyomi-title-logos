"""Step 4: write the passed logos into the library: logos/<anilistId>.webp (lossless, alpha) and index.json.

Only picks whose verdict in review.json is "pass" are exported. Usage: export.py [--date YYYY-MM-DD]
"""
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import studio as S  # noqa: E402
from keys import look_key  # noqa: E402

ROOT = S.ROOT
LIB = ROOT / 'library'


def source_note(aid, key, meta):
    if key == 'A':
        return meta[aid]['coverImage']['extraLarge']
    if key.startswith('M'):
        mdx = json.loads((ROOT / 'sources' / 'mangadex.json').read_text())[aid]
        c = mdx['covers'][int(key[1:]) - 1]
        vol = f"vol. {c['volume']}" if c.get('volume') else 'cover'
        return f"MangaDex {vol} ({c['locale']}): https://mangadex.org/title/{mdx['manga']}"
    return 'cover held by an Uchiyomi server'


def main(argv):
    today = argv[argv.index('--date') + 1] if '--date' in argv else date.today().isoformat()
    picks = json.loads((ROOT / 'picks.json').read_text())
    verdicts = json.loads((ROOT / 'review.json').read_text())
    meta = json.loads((ROOT / 'sources' / 'anilist.json').read_text())
    old = json.loads((LIB / 'index.json').read_text()) if (LIB / 'index.json').exists() else {}
    index = {}
    (LIB / 'logos').mkdir(parents=True, exist_ok=True)
    for aid in sorted(picks, key=int):
        pick, verdict = picks[aid], verdicts.get(aid, {})
        if pick.get('none') or verdict.get('verdict') != 'pass':
            continue
        cut = json.loads((ROOT / 'work' / 'cut' / aid / 'meta.json').read_text())
        if cut['key'] != verdict.get('key') or look_key(pick) != verdict.get('look'):
            print(aid, 'skipped: the verdict is for an older cut or look', file=sys.stderr)
            continue
        img = Image.fromarray(S.finish(pick, np.asarray(Image.open(ROOT / 'work' / 'cut' / aid / 'logo.png').convert('RGBA'))))
        dest = LIB / 'logos' / f'{aid}.webp'
        img.save(dest, 'WEBP', lossless=True, quality=100, method=6)
        data = dest.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        prev = old.get(aid, {})
        index[aid] = {'file': f'logos/{aid}.webp', 'sha256': sha, 'w': img.width, 'h': img.height,
                      'lang': pick['lang'], 'script': pick['script'], 'source': pick.get('source', 'cover'),
                      'from': pick.get('from') or source_note(aid, pick['src'], meta),
                      'added': prev['added'] if prev.get('sha256') == sha else today}
    for stale in (LIB / 'logos').glob('*.webp'):
        if stale.stem not in index:
            stale.unlink()
    (LIB / 'index.json').write_text(json.dumps(index, indent=1, ensure_ascii=False, sort_keys=False) + '\n')
    print('exported', len(index))


if __name__ == '__main__':
    main(sys.argv[1:])
