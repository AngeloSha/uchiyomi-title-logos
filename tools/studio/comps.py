"""The parts of a cut, in cover fractions, to place an erase or seed polygon exactly: comps.py <aid> [max_x0]

Lists every connected part of work/cut/<aid>/alpha.png (alpha > 0.12) with its box in fractions of the source
cover and its share of the logo, left to right up to max_x0.
"""
import json
import sys

import cv2
import numpy as np
from PIL import Image

aid = sys.argv[1]
x_max = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
d = f'/studio/work/cut/{aid}'
meta = json.load(open(f'{d}/meta.json'))
W, H = Image.open('/studio/' + meta['source']).size
cx0, cy0, cx1, cy1 = meta['crop']
a = cv2.imread(f'{d}/alpha.png', 0)
uh, uw = a.shape
fx, fy = uw / (cx1 - cx0), uh / (cy1 - cy0)
n, lab, st, _ = cv2.connectedComponentsWithStats((a > 30).astype(np.uint8), connectivity=8)
tot = st[1:, 4].sum()
for i in sorted(range(1, n), key=lambda i: st[i][0]):
    x, y, w, h, area = st[i]
    X0, Y0 = (x / fx + cx0) / W, (y / fy + cy0) / H
    X1, Y1 = ((x + w) / fx + cx0) / W, ((y + h) / fy + cy0) / H
    if X0 <= x_max:
        print(f'{X0:.3f},{Y0:.3f}-{X1:.3f},{Y1:.3f} area {area / tot * 100:.2f}%')
