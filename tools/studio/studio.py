"""Uchiyomi title-logo studio: the models and image helpers every step shares.

CPU only. Lucida (BiRefNet fine-tuned for text and logos) gives the soft alpha, Real-ESRGAN (anime 6B) gives the
upscale, ViTMatte-small refines edges from a trimap, and a blur-fusion foreground estimate removes background
colour from the edges. Nothing here downloads anything: weights are read from /studio/models.
"""
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(os.environ.get('STUDIO_ROOT', '/studio'))
MODELS = ROOT / 'models'
THREADS = int(os.environ.get('STUDIO_THREADS', '4'))
torch.set_num_threads(THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
cv2.setNumThreads(THREADS)
sys.path.insert(0, str(Path(__file__).resolve().parent))

# --------------------------------------------------------------------------------------------- image helpers

def load_rgb(path):
    img = Image.open(path)
    img.load()
    if img.mode in ('RGBA', 'LA', 'P'):
        img = img.convert('RGBA')
        bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return np.asarray(img.convert('RGB'))


def save_png(path, arr):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def rgba_resize(rgba, size):
    """Resize straight-alpha RGBA through premultiplied space so edges keep their colour."""
    a = rgba[..., 3:4].astype(np.float32) / 255
    pre = np.concatenate([rgba[..., :3].astype(np.float32) * a, a * 255], axis=2)
    pre = cv2.resize(pre, size, interpolation=cv2.INTER_AREA if size[0] < rgba.shape[1] else cv2.INTER_LANCZOS4)
    pre = np.clip(pre, 0, 255)
    alpha = pre[..., 3:4] / 255
    rgb = np.where(alpha > 1e-4, pre[..., :3] / np.maximum(alpha, 1e-4), 0)
    return np.concatenate([np.clip(rgb, 0, 255), pre[..., 3:4]], axis=2).round().astype(np.uint8)


def trim(rgba, thresh=3, pad=2):
    ys, xs = np.nonzero(rgba[..., 3] > thresh)
    if not len(xs):
        return rgba
    y0, y1 = max(0, ys.min() - pad), min(rgba.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(rgba.shape[1], xs.max() + 1 + pad)
    return rgba[y0:y1, x0:x1]


def composite(rgba, bg):
    """Straight-alpha RGBA over an RGB background of the same size."""
    a = rgba[..., 3:4].astype(np.float32) / 255
    return (rgba[..., :3].astype(np.float32) * a + bg.astype(np.float32) * (1 - a)).round().astype(np.uint8)

# ---------------------------------------------------------------------------------------------------- Lucida

_lucida = None


def lucida_model():
    global _lucida
    if _lucida is None:
        from safetensors.torch import load_file
        from lucida_model.birefnet import BiRefNet
        from lucida_model.BiRefNet_config import BiRefNetConfig
        model = BiRefNet(config=BiRefNetConfig(bb_pretrained=False))
        model.load_state_dict(load_file(str(MODELS / 'lucida' / 'model.safetensors'), device='cpu'), strict=True)
        _lucida = model.eval()
    return _lucida


_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def lucida_alpha(rgb, size=(1024, 1024)):
    """Lucida's final head as a soft alpha at the input's own size. `size` is the model input (w, h), multiples of 32."""
    h, w = rgb.shape[:2]
    x = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA if size[0] * size[1] < w * h else cv2.INTER_CUBIC)
    x = ((x.astype(np.float32) / 255 - _MEAN) / _STD).transpose(2, 0, 1)[None]
    with torch.inference_mode():
        out = lucida_model()(torch.from_numpy(np.ascontiguousarray(x)))[-1].sigmoid()[0, 0].numpy()
    return np.clip(cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR), 0, 1).astype(np.float32)


def model_size(w, h, long_side=1024, mult=32):
    s = long_side / max(w, h)
    return (max(mult, int(round(w * s / mult)) * mult), max(mult, int(round(h * s / mult)) * mult))

# ----------------------------------------------------------------------------------------------- Real-ESRGAN


class _RDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        return self.conv5(torch.cat((x, x1, x2, x3, x4), 1)) * 0.2 + x


class _RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.rdb1, self.rdb2, self.rdb3 = _RDB(nf, gc), _RDB(nf, gc), _RDB(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    """The x4 RRDBNet of ESRGAN/Real-ESRGAN (num_block=6 for the anime model)."""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32):
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[_RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


_esrgan = None


def esrgan_model():
    global _esrgan
    if _esrgan is None:
        from safetensors.torch import load_file
        model = RRDBNet()
        model.load_state_dict(load_file(str(MODELS / 'realesrgan-anime-6b' / 'model.safetensors'), device='cpu'), strict=True)
        _esrgan = model.eval()
    return _esrgan


def upscale4(rgb, tile=256, pad=12):
    """Real-ESRGAN x4 in overlapping tiles (bounded memory)."""
    h, w = rgb.shape[:2]
    x = torch.from_numpy(rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255)
    out = np.zeros((h * 4, w * 4, 3), np.float32)
    model = esrgan_model()
    with torch.inference_mode():
        for y0 in range(0, h, tile):
            for x0 in range(0, w, tile):
                y1, x1 = min(h, y0 + tile), min(w, x0 + tile)
                py0, px0, py1, px1 = max(0, y0 - pad), max(0, x0 - pad), min(h, y1 + pad), min(w, x1 + pad)
                o = model(x[:, :, py0:py1, px0:px1])[0].clamp(0, 1).numpy().transpose(1, 2, 0)
                oy, ox = (y0 - py0) * 4, (x0 - px0) * 4
                out[y0 * 4:y1 * 4, x0 * 4:x1 * 4] = o[oy:oy + (y1 - y0) * 4, ox:ox + (x1 - x0) * 4]
    return (out * 255).round().astype(np.uint8)


def upscale(rgb, factor):
    """Upscale by any factor in (1, 4]: Real-ESRGAN x4, then area-downscale to the factor."""
    if factor <= 1.01:
        return rgb
    big = upscale4(rgb)
    h, w = rgb.shape[:2]
    size = (int(round(w * factor)), int(round(h * factor)))
    return big if size == (w * 4, h * 4) else cv2.resize(big, size, interpolation=cv2.INTER_AREA)

# -------------------------------------------------------------------------------------------------- ViTMatte

_vitmatte = None


def vitmatte(rgb, trimap):
    """Alpha from an image and a 0/128/255 trimap (ViTMatte-small, composition-1k)."""
    global _vitmatte
    from transformers import VitMatteForImageMatting, VitMatteImageProcessor
    if _vitmatte is None:
        d = str(MODELS / 'vitmatte-small')
        _vitmatte = (VitMatteImageProcessor.from_pretrained(d, local_files_only=True),
                     VitMatteForImageMatting.from_pretrained(d, local_files_only=True).eval())
    proc, model = _vitmatte
    inputs = proc(images=Image.fromarray(rgb), trimaps=Image.fromarray(trimap), return_tensors='pt')
    with torch.inference_mode():
        alpha = model(**inputs).alphas[0, 0].numpy()
    h, w = rgb.shape[:2]
    return np.clip(alpha[:h, :w], 0, 1).astype(np.float32)

# ---------------------------------------------------------------------------------------- foreground colour


def _blur_fusion(image, fg, bg, alpha, r):
    a = alpha[..., None]
    ba = cv2.blur(alpha, (r, r))[..., None]
    bf = cv2.blur(fg * a, (r, r)) / (ba + 1e-5)
    bb = cv2.blur(bg * (1 - a), (r, r)) / ((1 - ba) + 1e-5)
    fg = bf + a * (image - a * bf - (1 - a) * bb)
    return np.clip(fg, 0, 1), bb


def estimate_foreground(rgb, alpha):
    """Approximate fast foreground colour estimation (Forte & Pitie 2021, as BiRefNet ships it): two blur-fusion
    passes, so a soft edge carries the logo's colour instead of the cover's background."""
    img = rgb.astype(np.float32) / 255
    r1 = max(15, int(round(90 * max(rgb.shape[:2]) / 1024)))
    fg, bg = _blur_fusion(img, img, img, alpha, r1)
    fg, _ = _blur_fusion(img, fg, bg, alpha, 6)
    return (fg * 255).round().astype(np.uint8)

# ---------------------------------------------------------------------------------------------------- Hi-SAM

_hisam = None


def hisam_logits(rgb):
    """Hi-SAM text-stroke logits for the whole image (the stroke model of services/title-extractor, v0.102.1)."""
    global _hisam
    if _hisam is None:
        from types import SimpleNamespace
        src = Path(os.environ.get('HI_SAM_DIR', '/opt/hi-sam'))
        sys.path.insert(0, str(src))
        cwd = os.getcwd()
        os.chdir(src)
        try:
            from hi_sam.modeling.build import model_registry
            from hi_sam.modeling.predictor import SamPredictor
            model = model_registry['vit_b'](SimpleNamespace(
                checkpoint=str(src / 'pretrained_checkpoint/sam_tss_b_textseg.pth'), model_type='vit_b',
                hier_det=False, attn_layers=1, prompt_len=12)).eval().to('cpu')
        finally:
            os.chdir(cwd)
        _hisam = SamPredictor(model)
    with torch.inference_mode():
        _hisam.set_image(np.ascontiguousarray(rgb))
        _, logits, _, _ = _hisam.predict(return_logits=True)
    return np.squeeze(logits).astype(np.float32)


def line_groups(mask):
    """Strokes joined into line boxes; the same grouping as worker.line_groups (ink bounds, not dilation)."""
    height, width = mask.shape
    joined = cv2.dilate(mask.astype(np.uint8), np.ones((max(3, height // 80), max(5, width // 18)), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined)
    groups = []
    for label in range(1, count):
        x, y, w, h, _ = map(int, stats[label])
        ink = mask[y:y + h, x:x + w] & (labels[y:y + h, x:x + w] == label)
        yy, xx = np.nonzero(ink)
        if len(xx) < max(24, width * height * .0002):
            continue
        groups.append(([x + int(xx.min()), y + int(yy.min()), x + int(xx.max()) + 1, y + int(yy.max()) + 1], len(xx)))
    return groups


def clean_matte(rgba, near=1.5, fade=3.0, floor=0.02):
    """Remove matting haze while keeping soft edges: alpha under 0.5 survives only within `near` px of the solid
    logo (alpha >= 0.5), fading to nothing over `fade` px more; then a small floor. Run at the output size."""
    a = rgba[..., 3].astype(np.float32) / 255
    solid = (a >= 0.5).astype(np.uint8)
    if not solid.any():
        return rgba
    d = cv2.distanceTransform(1 - solid, cv2.DIST_L2, 3)
    keep = np.clip(1 - (d - near) / fade, 0, 1)
    a = np.where(solid > 0, a, a * keep)
    a = np.clip((a - floor) / (1 - floor), 0, 1)
    return trim(np.dstack([rgba[..., :3], (a * 255).round().astype(np.uint8)]), thresh=0, pad=1)

# ------------------------------------------------------------------------------------------- light versions

_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


def to_oklab(rgb):
    c = rgb / 255
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return np.cbrt(lin @ _M1.T) @ _M2.T


def from_oklab(lab):
    lin = ((lab @ np.linalg.inv(_M2).T) ** 3) @ np.linalg.inv(_M1).T
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055) * 255


def lighten(rgba, below=0.5, soft=0.15, top=0.97, chroma=0.45, region=None, rims=None, gain=1.0):
    """A light version of a dark logo for dark heroes, as clearlogo sites keep one beside the original.

    Pixels darker than `below` (OKLab lightness) get their lightness mirrored: black turns near-white (`top`), navy
    a pale blue keeping `chroma` of its colour. Lighter colours, accents and outlines stay as they are, and so do the
    letterforms. `region`: polygons in fractions of the logo, the only place it applies (a dark first line over a
    coloured second one). `rims`: also mirror greys lighter than this, so a white rim round black letters turns dark
    and still reads as a rim. `gain` below 1 squeezes the mirrored range towards `top`, so a mid-grey fill (not
    only a black one) comes out near-white."""
    lab = to_oklab(rgba[..., :3].astype(np.float64))
    L = lab[..., 0]
    w = np.clip((below - L) / soft, 0, 1)
    w = w * w * (3 - 2 * w)
    if rims is not None:
        grey = np.clip(1 - np.hypot(lab[..., 1], lab[..., 2]) / 0.06, 0, 1)
        r = np.clip((L - rims) / soft, 0, 1)
        w = np.maximum(w, r * r * (3 - 2 * r) * grey)
    if region:
        h, wd = L.shape
        m = np.zeros((h, wd), np.uint8)
        for poly in region:
            cv2.fillPoly(m, [np.array([(x * wd, y * h) for x, y in poly], np.int32)], 1)
        w = w * cv2.GaussianBlur(m.astype(np.float32), (0, 0), 2)
    out = lab.copy()
    target = np.where(L < 0.5, top - gain * L, 1.12 - L)  # mirrored: a dark pixel becomes top - gain * L
    out[..., 0] = np.clip(L + w * (target - L), 0, 1)
    out[..., 1:] = lab[..., 1:] * (1 - w * (1 - chroma))[..., None]
    return np.dstack([from_oklab(out).round().astype(np.uint8), rgba[..., 3]])


def finish(pick, rgba):
    """What ships, from a cut: matting haze off (soft edges kept), then the light version when the pick asks."""
    if pick.get('source', 'cover') == 'cover' and pick.get('clean', True):
        rgba = clean_matte(rgba)
    if pick.get('light'):
        rgba = lighten(rgba, **({} if pick['light'] is True else pick['light']))
    return rgba
