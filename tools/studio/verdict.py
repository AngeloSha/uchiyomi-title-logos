"""Record review verdicts: verdict.py pass <ids...> | verdict.py fail <id> "<what is wrong>"

A verdict is tied to the cut it was given for (meta.json key) and to the export-only fields (look key), so a re-cut
or a new light version needs a new look.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from keys import look_key  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
path = ROOT / 'review.json'
verdicts = json.loads(path.read_text()) if path.exists() else {}
picks = json.loads((ROOT / 'picks.json').read_text())
kind, ids = sys.argv[1], sys.argv[2:]
note = ''
if kind == 'fail':
    ids, note = ids[:1], ' '.join(ids[1:])
for aid in ids:
    meta = json.loads((ROOT / 'work' / 'cut' / aid / 'meta.json').read_text())
    verdicts[aid] = {'verdict': kind, 'key': meta['key'], 'look': look_key(picks[aid]), 'note': note}
path.write_text(json.dumps(verdicts, indent=1, ensure_ascii=False))
print({k: sum(1 for v in verdicts.values() if v['verdict'] == k) for k in ('pass', 'fail')})
