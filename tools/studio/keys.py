"""Which version of a pick a cut or a verdict belongs to (stdlib only, so verdict.py runs anywhere).

The cut key covers the fields that change the cut; the look key covers the export-only fields (haze cleanup, the
light version), which change what ships without a re-cut. A verdict holds both, so changing either needs a new look.
"""
import hashlib
import json

EXPORT_ONLY = {'clean', 'light'}  # fields that change the export, not the cut


def _hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def pick_key(pick):
    return _hash({k: v for k, v in pick.items() if k not in EXPORT_ONLY})


def look_key(pick):
    return _hash({k: v for k, v in pick.items() if k in EXPORT_ONLY})
