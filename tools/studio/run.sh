#!/bin/sh
# Run one studio step in the CPU container: ./run.sh <script.py> [args...]  (4 threads, no network)
cd "$(dirname "$0")"
exec docker run --rm --label studio-step="$1" --user "$(id -u):$(id -g)" --cpus "${THREADS:-4}" --memory 14g --network none \
  -e STUDIO_THREADS="${THREADS:-4}" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONPATH=/studio/src \
  -v "$PWD":/studio -v "$PWD/models/hi-sam":/models:ro -w /studio --entrypoint python "${STUDIO_IMAGE:-uchiyomi-logo-studio}" -I src/"$@"
