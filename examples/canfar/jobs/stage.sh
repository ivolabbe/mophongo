#!/bin/bash
# Decompress one config's inputs into $RUN/data.
#
# Nothing is uploaded: every file listed in <CFG>_stage.tsv already lives on
# /arc/projects/minerva, and /arc is mounted in the container, so this is a
# local gunzip. Uncompressed inputs are not listed at all - arcify.py points the
# config straight at them.
#
# Bands share the F444W mosaic and the segmap, so concurrent stage jobs race on
# the same destination. Each writes to its own temp name and drops it if another
# job finished the file first.
set -euo pipefail
: "${RUN:?RUN not set}" "${CFG:?CFG not set}"
D=$RUN/data
mkdir -p $D
manifest=$RUN/${CFG}_stage.tsv
[[ -s $manifest ]] || { echo "no manifest: $manifest" >&2; exit 1; }

while IFS=$'\t' read -r src dst; do
    [[ -n "$src" ]] || continue
    if [[ -s "$D/$dst" ]]; then echo "have     $dst"; continue; fi
    tmp="$D/.$dst.$$.part"
    if [[ "$src" == *.gz ]]; then
        echo "gunzip   $dst"
        gunzip -c "$src" > "$tmp"
    else
        echo "copy     $dst"
        cp "$src" "$tmp"
    fi
    if [[ -s "$D/$dst" ]]; then
        echo "         (another job won the race, discarding)"
        rm -f "$tmp"
    else
        mv "$tmp" "$D/$dst"
    fi
done < "$manifest"

echo "=== staged for $CFG:"
cut -f2 "$manifest" | while read -r f; do [[ -n "$f" ]] && ls -lh "$D/$f"; done
df -h $D | tail -1
echo STAGE_DONE
