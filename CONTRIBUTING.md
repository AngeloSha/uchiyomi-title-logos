# Adding or fixing a logo

One logo per AniList id. A pull request needs:

1. **The right series.** Use the AniList id of the manga entry (not the anime). The logo must be that series' own
   title logo, from an official cover or an official logo file. No fan-made logos, no AI-generated lettering.
2. **The whole logo, nothing else.** Keep the subtitle line, emblem, frame or "+" that belong to the logo. Leave out
   author credits, volume numbers, publisher and label marks, taglines and any art behind the lettering.
3. **Clean edges.** Soft alpha (no hard staircase edges), no halo of the cover's background colour, no holes.
   Check it on a dark, a light and a busy background.
4. **Readable on a dark page.** Uchiyomi shows logos over a darkened banner. If the lettering is dark, send the
   light version: the dark fill recoloured to near-white, coloured accents, outlines and letterforms unchanged.
5. **The file.** `logos/<anilistId>.webp`, lossless WebP with alpha, trimmed to the logo, at most 1600 px wide.
6. **The index entry** in `index.json` (see README.md), with the `sha256` of the file.
7. **`python tools/validate.py` passes.**

In the pull request, show the logo on a dark and a light background and say which cover it came from.

When a series has logos in several languages, prefer the one matching the title most readers use, and set `lang`
and `script` to match the logo.

## Using the studio

`tools/studio/` holds the scripts these logos were made with. They run on a CPU in a container, read the cover
images you give them and write a candidate cut-out for each pick. Every result still needs a look by eye before
it goes in.
