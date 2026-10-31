#!/bin/sh
# Render a contact sheet (1 thread, no network): ./sheet.sh pick 0 12 | zoom <aid> <key> | review <name> <ids...> | hero <name> <ids...> | fix <ids...>
cd "$(dirname "$0")"
exec docker run --rm --user "$(id -u):$(id -g)" --cpus 1 --network none -e STUDIO_THREADS=1 -e PYTHONPATH=/studio/src \
  -v "$PWD":/studio -v /usr/share/fonts/opentype/noto:/fonts:ro -w /studio --entrypoint python "${STUDIO_IMAGE:-uchiyomi-logo-studio}" -I src/sheets.py "$@" 2>&1 | grep -v -i warn
