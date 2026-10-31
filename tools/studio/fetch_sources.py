"""Fetch the approved cover sources for every series with a checked AniList id.

AniList: extraLarge cover + banner (one GraphQL call per 50 ids).
MangaDex: the manga whose own links.al equals the AniList id (exact match, never a fuzzy pick), then up to
two original covers: the first volume in the original language and the first English one (or the latest).
"""
import json, sys, time, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path('/studio')
SRC = ROOT / 'sources'
UA = {'User-Agent': 'uchiyomi-title-logos-studio/1 (personal library curation)'}
MDX_CAP = 300 * 1024 * 1024


def get(url, data=None, headers=None, timeout=60):
    req = urllib.request.Request(url, data=data, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.part'); tmp.write_bytes(data); tmp.replace(path)


def anilist(ids):
    q = '''query($ids:[Int],$page:Int){Page(page:$page,perPage:50){media(id_in:$ids,type:MANGA){id
      title{romaji english native} synonyms countryOfOrigin format coverImage{extraLarge color} bannerImage}}}'''
    out = {}
    for i in range(0, len(ids), 50):
        chunk = [int(x) for x in ids[i:i + 50]]
        body = json.dumps({'query': q, 'variables': {'ids': chunk, 'page': 1}}).encode()
        for attempt in range(5):
            try:
                res = json.loads(get('https://graphql.anilist.co', body, {'Content-Type': 'application/json'}))
                break
            except urllib.error.HTTPError as e:
                if e.code == 429: time.sleep(65); continue
                raise
        for m in res['data']['Page']['media']: out[str(m['id'])] = m
        time.sleep(2.5)
    return out


def mdx_find(meta):
    titles = [meta['title'].get('romaji'), meta['title'].get('english'), meta['title'].get('native')] + (meta.get('synonyms') or [])[:3]
    seen = set()
    for t in titles:
        if not t or t in seen: continue
        seen.add(t)
        qs = urllib.parse.urlencode([('title', t), ('limit', 20)] + [('contentRating[]', c) for c in ('safe', 'suggestive', 'erotica', 'pornographic')])
        try: res = json.loads(get('https://api.mangadex.org/manga?' + qs))
        except Exception as e: print('mdx search fail', meta['id'], e, file=sys.stderr); time.sleep(2); continue
        time.sleep(0.35)
        for m in res.get('data', []):
            if str((m['attributes'].get('links') or {}).get('al') or '') == str(meta['id']):
                return m
    return None


def main():
    series = json.load(open(SRC / 'series.json'))
    ids = sorted({s['anilist_id'] for s in series if s['anilist_id']}, key=int)
    meta_path = SRC / 'anilist.json'
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    missing = [i for i in ids if i not in meta]
    if missing: meta.update(anilist(missing)); save(meta_path, json.dumps(meta, ensure_ascii=False, indent=1).encode())
    print('anilist meta', len(meta), 'of', len(ids))
    for i in ids:
        m = meta.get(i)
        if not m: continue
        for kind, url in (('anilist', (m.get('coverImage') or {}).get('extraLarge')), ('banner', m.get('bannerImage'))):
            if not url: continue
            dest = SRC / kind / (i + Path(urllib.parse.urlparse(url).path).suffix)
            if dest.exists(): continue
            try: save(dest, get(url)); time.sleep(0.2)
            except Exception as e: print('fail', kind, i, e, file=sys.stderr)
    if '--no-mdx' in sys.argv: return
    mdx_path = SRC / 'mangadex.json'
    mdx = json.load(open(mdx_path)) if mdx_path.exists() else {}
    used = sum(p.stat().st_size for p in (SRC / 'mangadex').glob('*')) if (SRC / 'mangadex').exists() else 0
    for i in ids:
        if i in mdx or i not in meta: continue
        m = mdx_find(meta[i])
        entry = {'manga': None, 'covers': []}
        if m:
            orig = m['attributes'].get('originalLanguage')
            entry['manga'] = m['id']; entry['originalLanguage'] = orig
            qs = urllib.parse.urlencode([('manga[]', m['id']), ('limit', 100), ('order[volume]', 'asc')])
            try: covers = json.loads(get('https://api.mangadex.org/cover?' + qs)).get('data', [])
            except Exception as e: covers = []; print('mdx covers fail', i, e, file=sys.stderr)
            time.sleep(0.35)
            native = [c for c in covers if c['attributes'].get('locale') == orig]
            english = [c for c in covers if c['attributes'].get('locale') == 'en']
            picks = []
            if native: picks.append(native[0])
            if english: picks.append(english[0])
            elif len(native) > 1: picks.append(native[-1])
            if not picks and covers: picks.append(covers[0])
            for c in picks:
                fn = c['attributes']['fileName']
                dest = SRC / 'mangadex' / f"{i}_{c['id'][:8]}{Path(fn).suffix}"
                if used > MDX_CAP: print('mdx cap reached'); break
                if not dest.exists():
                    try:
                        data = get(f"https://uploads.mangadex.org/covers/{m['id']}/{fn}", timeout=120)
                        save(dest, data); used += len(data); time.sleep(0.3)
                    except Exception as e: print('mdx dl fail', i, e, file=sys.stderr); continue
                entry['covers'].append({'id': c['id'], 'file': dest.name, 'volume': c['attributes'].get('volume'), 'locale': c['attributes'].get('locale')})
        mdx[i] = entry
        save(mdx_path, json.dumps(mdx, indent=1).encode())
    print('mangadex linked', sum(1 for v in mdx.values() if v['manga']), 'covers', sum(len(v['covers']) for v in mdx.values()), 'bytes', used)


if __name__ == '__main__':
    main()
