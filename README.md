# Uchiyomi title logos

Transparent title logos for manga, manhwa and manhua, keyed by [AniList](https://anilist.co) id. Each one was cut
out of an official cover from a hand-picked region (a few are official logo files), then checked by eye on dark,
light and banner backgrounds: the whole logo (subtitle line, emblem, frame), nothing extra (no credits, volume
numbers or publisher marks), clean soft edges.

A logo whose lettering is dark (black, navy, dark red) would vanish on the dark page behind it, so it ships as a
light version, as clearlogo sites keep one: the dark fill is recoloured to near-white, while coloured accents,
outlines and the letterforms stay as published.

[Uchiyomi](https://github.com/AngeloSha/uchiyomi) shows these in the hero of a series page on a computer. A phone
shows the cover instead, so no logo is downloaded there.

## How Uchiyomi uses it

1. A series has a checked AniList link on the server.
2. The server reads `index.json` from this repository (cached, refreshed daily) and looks up that id.
3. It downloads `logos/<id>.webp`, checks its `sha256`, and shows it as the title logo.

A logo an admin uploaded for a series is never replaced. A series with no entry keeps its styled text title.

## Files

- `logos/<anilistId>.webp`: lossless WebP with alpha, trimmed to the logo, at most 1600 px wide.
- `index.json`: one entry per AniList id:

```json
{
  "30013": {
    "file": "logos/30013.webp",
    "sha256": "…",
    "w": 1600, "h": 581,
    "lang": "ja",
    "script": "Latn",
    "source": "cover",
    "from": "MangaDex vol. 1 (ja): https://mangadex.org/title/…",
    "added": "2026-10-31"
  }
}
```

- `lang`: the edition the logo belongs to (BCP 47: `ja`, `ko`, `zh`, `en` …).
- `script`: the main script of the lettering (ISO 15924: `Latn`, `Jpan`, `Hang`, `Hans` …). A server can add a
  text caption under a logo its readers may not read.
- `source`: `cover` (cut from a cover) or `official` (an official logo file).
- `from`: where it came from. `added`: the date it was added or last changed.

## Notice

The logos are trademarks and artwork of their publishers and creators. They are included only to identify each
series, the way fan databases such as TMDB and fanart.tv do, and they are not covered by this repository's MIT
license, which applies to the tools and `index.json` only. If you hold the rights to a logo and want it removed,
open an issue and it will be taken down.

## Tools

- `tools/validate.py`: checks the index and every file (schema, sha256, lossless WebP, real alpha, size). CI runs it
  on every push and pull request.
- `tools/studio/`: the pipeline these logos were made with (CPU only): numbered region proposals, upscaling, soft
  alpha matting, foreground colour estimation and contact sheets for review. See its README.

Want to add or fix a logo? See [CONTRIBUTING.md](CONTRIBUTING.md).
