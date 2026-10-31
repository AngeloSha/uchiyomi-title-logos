#!/usr/bin/env python3
"""Validate the logo library: index.json schema, every file's sha256 and size, real alpha, lossless WebP.

Usage: python tools/validate.py [library-root]     (exit status 1 on any problem)
Needs Pillow and numpy.
"""
import hashlib
import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

MAX_W = 1600
MAX_BYTES = 2_500_000
FIELDS = {'file', 'sha256', 'w', 'h', 'lang', 'script', 'source', 'from', 'added'}
LANG = re.compile(r'^[a-z]{2,3}(-[A-Za-z0-9]{2,8})*$')
SCRIPT = re.compile(r'^[A-Z][a-z]{3}(\+[A-Z][a-z]{3})*$')  # ISO 15924, "+" joins a mixed logo


def webp_is_lossless(data):
    """A lossless WebP with alpha is RIFF/WEBP with a VP8L chunk (alpha lives inside it); VP8/ALPH means lossy."""
    if data[:4] != b'RIFF' or data[8:12] != b'WEBP':
        return False
    pos, chunks = 12, set()
    while pos + 8 <= len(data):
        tag, size = data[pos:pos + 4], int.from_bytes(data[pos + 4:pos + 8], 'little')
        chunks.add(tag)
        pos += 8 + size + (size & 1)
    return b'VP8L' in chunks and b'VP8 ' not in chunks


def check(root):
    root = Path(root)
    errors = []
    try:
        index = json.loads((root / 'index.json').read_text(encoding='utf-8'))
    except Exception as e:  # noqa: BLE001
        return [f'index.json: {e}']
    if not isinstance(index, dict):
        return ['index.json: must be an object keyed by AniList id']
    listed = set()
    for aid, entry in index.items():
        where = f'index[{aid}]'
        if not re.fullmatch(r'[1-9][0-9]{0,9}', aid):
            errors.append(f'{where}: key must be an AniList id')
            continue
        if not isinstance(entry, dict) or set(entry) != FIELDS:
            errors.append(f'{where}: fields must be exactly {sorted(FIELDS)}')
            continue
        if entry['file'] != f'logos/{aid}.webp':
            errors.append(f'{where}: file must be logos/{aid}.webp')
            continue
        if entry['source'] not in ('cover', 'official'):
            errors.append(f'{where}: source must be cover or official')
        if not LANG.match(str(entry['lang'])):
            errors.append(f'{where}: lang must be a BCP 47 tag')
        if not SCRIPT.match(str(entry['script'])):
            errors.append(f'{where}: script must be ISO 15924 (e.g. Latn, Jpan, Hang)')
        if not isinstance(entry['from'], str) or not entry['from'].strip():
            errors.append(f'{where}: from must say where the logo came from')
        try:
            date.fromisoformat(entry['added'])
        except Exception:  # noqa: BLE001
            errors.append(f'{where}: added must be YYYY-MM-DD')
        path = root / entry['file']
        listed.add(path.name)
        if not path.is_file():
            errors.append(f'{where}: {entry["file"]} is missing')
            continue
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            errors.append(f'{where}: sha256 does not match the file')
        if len(data) > MAX_BYTES:
            errors.append(f'{where}: {len(data)} bytes is over {MAX_BYTES}')
        if not webp_is_lossless(data):
            errors.append(f'{where}: must be a lossless WebP')
        try:
            img = Image.open(path)
            img.load()
        except Exception as e:  # noqa: BLE001
            errors.append(f'{where}: unreadable image ({e})')
            continue
        if img.mode != 'RGBA':
            errors.append(f'{where}: must have an alpha channel (mode {img.mode})')
            continue
        if (img.width, img.height) != (entry['w'], entry['h']):
            errors.append(f'{where}: w/h {entry["w"]}x{entry["h"]} but the file is {img.width}x{img.height}')
        if img.width > MAX_W:
            errors.append(f'{where}: {img.width} px wide is over {MAX_W}')
        a = np.asarray(img)[..., 3]
        clear, opaque = (a == 0).mean(), (a == 255).mean()
        if clear < 0.05:
            errors.append(f'{where}: almost no transparent pixels; this is not a cut-out')
        if opaque < 0.02:
            errors.append(f'{where}: almost no opaque pixels; the logo is missing')
        edge = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
        if edge.max() > 200 and (edge > 200).mean() > 0.25:
            errors.append(f'{where}: opaque along the border; not trimmed or cut too tight')
    for path in sorted((root / 'logos').glob('*')):
        if path.name not in listed:
            errors.append(f'logos/{path.name}: not listed in index.json')
    return errors


if __name__ == '__main__':
    problems = check(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent)
    for p in problems:
        print('error:', p)
    if problems:
        sys.exit(1)
    print('ok')
