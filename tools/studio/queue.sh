#!/bin/sh
# Run a cut batch after any cut still running (one CPU job at a time): ./queue.sh <log-name> <ids...>
cd "$(dirname "$0")"
log="logs/$1.log"; shift
while [ -n "$(docker ps -q --filter label=studio-step=cut.py)" ]; do sleep 5; done
THREADS=4 ./run.sh cut.py "$@" > "$log" 2>&1
grep -v -i -e warn -e 'return fn' -e 'Freeze image' -e 'keys matched' "$log"
