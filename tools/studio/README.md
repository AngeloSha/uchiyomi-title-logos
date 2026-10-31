# The logo studio

The scripts these logos were made with. CPU only; they run in a container (see `Dockerfile`; build it as
`uchiyomi-logo-studio`, or set `STUDIO_IMAGE`) and read model weights from `models/`. Apart from
`fetch_sources.py`, which fetches covers, nothing here downloads anything.

Layout of a studio folder (the scripts see it as `/studio`):

```
models/lucida/model.safetensors            huggingface.co/egeorcun/lucida
models/vitmatte-small/                      huggingface.co/hustvl/vitmatte-small-composition-1k (config + weights)
models/realesrgan-anime-6b/model.safetensors   RealESRGAN_x4plus_anime_6B as safetensors
models/hi-sam/sam_tss_b_textseg.pth, sam_vit_b_01ec64.pth   Hi-SAM text stroke model + SAM ViT-B
src/                                        these scripts, plus src/lucida_model/ = birefnet.py and
                                            BiRefNet_config.py from the Lucida repository (MIT)
sources/  work/  review/  picks.json  review.json  library/
```

`./run.sh cut.py <ids>` runs a step with 4 threads (`THREADS=2 ./run.sh …` for fewer); `./queue.sh <log> <ids>`
runs a cut after any cut still running, one CPU job at a time; `./sheet.sh …` renders sheets (the `hero` and `fix`
sheets need Noto Sans CJK on the host, `fonts-noto-cjk`, which `sheet.sh` mounts at `/fonts`).

| Step | Script | What it does |
| --- | --- | --- |
| 0 | `series.sql`, `fetch_sources.py` | A server's series with their checked AniList ids; AniList `extraLarge` covers and banners; MangaDex original covers where the MangaDex entry links that same AniList id. |
| 1 | `propose.py` | Hi-SAM text strokes on each cover, joined into numbered line boxes. |
| 2 | `sheets.py pick` / `boxes` / `zooms` | Contact sheets: every candidate cover, the chosen cover with its numbered boxes and a 10% grid. A person picks the source and the boxes into `picks.json`. |
| 3 | `cut.py` | Crop with a margin, upscale small crops with Real-ESRGAN (anime 6B), soft alpha, keep what lies in the chosen boxes, erase/restore polygons, foreground colour estimation, trim, cap at 1600 px. |
| 4 | `sheets.py hero` / `fix`, `verdict.py` | Logos in the app's hero conditions (640x260 box, black wash, its drop shadows) with automatic warnings for cut edges, haze, strays and dark lettering, plus the source crop with a grid and the kept alpha over it; a person passes or fails each. |
| 5 | `export.py` | Passed logos, with the haze cleanup and any light version, to `logos/<id>.webp` (lossless) and `index.json`. |

## Soft alpha, three ways

- **strokes** (default): Hi-SAM's text strokes are the sure logo; a band around them (outlines, shadows, and any
  frame or emblem named by an `include` polygon) is left unknown; everything further away is background.
  ViTMatte-small solves the unknown band from that trimap. This holds on busy cover art.
- **hybrid**: as `strokes`, plus Lucida's most certain logo pixels as extra sure-logo seeds. For ornate numerals,
  emblems and coloured strokes that Hi-SAM leaves out.
- **lucida**: Lucida (BiRefNet fine-tuned for text and logos, MIT) on the crop at its 1024 px training size,
  optionally refined by ViTMatte. Best on plain backgrounds and for emblems far from the lettering. On a busy
  background it marks the whole crop, so it is not the default.

The alpha is never thresholded. Foreground colour is estimated with two blur-fusion passes
(Forte & Pitié, "Approximate Fast Foreground Colour Estimation", 2021), so a soft edge carries the logo's own colour.

## Models

| Model | Licence | Where |
| --- | --- | --- |
| Hi-SAM text stroke segmentation (`sam_tss_b_textseg.pth`, SAM ViT-B) | Apache-2.0 | github.com/ymy-k/Hi-SAM |
| Lucida | MIT | huggingface.co/egeorcun/lucida |
| ViTMatte-small (composition-1k) | Apache-2.0 | huggingface.co/hustvl/vitmatte-small-composition-1k |
| Real-ESRGAN x4plus anime 6B | BSD-3-Clause | github.com/xinntao/Real-ESRGAN |

`studio.py` holds a small RRDBNet (the Real-ESRGAN network) so no extra package is needed for it.

## A pick

```json
"30013": {"src": "M1", "boxes": [2, 3], "box": [[0.02, 0.06, 0.97, 0.28]], "lang": "ja", "script": "Latn",
          "alpha": "lucida", "include": "boxes", "erase": [[[0.8, 0.9], [0.95, 0.9], [0.95, 0.95], [0.8, 0.95]]]}
```

Coordinates are fractions of the source cover. `boxes` are proposal numbers, `box` adds boxes by hand, `include`
marks parts that are logo but not lettering, `erase` removes credits or art that touch the logo.
`{"none": "why"}` records a series that gets no entry.

More fields, for the strokes a model leaves out and the art it keeps:

- `seed_poly` (hybrid): where Lucida's seeds count. `seed_lum`: `[{"poly": [...], "min": 0.85}]` adds the pixels
  this bright (or `"max"`: this dark) inside a polygon as sure logo, for a stroke Hi-SAM missed.
- `restore`: counts a polygon as inside the chosen boxes; `solid_fg`: sure logo in the trimap.
- `erase_lum`: `[{"poly": [...], "max": 0.3}]` fades out what is this dark (or `"min"`: this light) inside a
  polygon, for background showing through a letter.
- `solid`: alpha 1 after matting; `{"poly": [...], "from": [x, y]}` rebuilds a stroke the cover hides, in the
  colour at `from`.
- `min_part`: drops specks lighter than this share of the logo.
- `light` (export only): the light version of a dark logo; `{"region": [...]}` in fractions of the logo limits it
  to one line, `below`, `chroma` and `rims` tune it (see `studio.lighten`). `clean: false` skips the haze cleanup.

A verdict is tied to both the cut and the export fields (`keys.py`), so changing either needs a new look. The
upscale and the model outputs are cached per crop in `work/cache/`, so moving a polygon re-cuts in seconds.
