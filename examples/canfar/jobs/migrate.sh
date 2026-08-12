#!/bin/bash
# Move a run tree from one root to another, e.g. out of a quota-limited home
# into project space. /arc/home and /arc/projects are the same filesystem, so
# this is a rename: instant, and nothing is copied or re-staged.
#
# SRC and RUN are the old and new roots.
set -euo pipefail
: "${RUN:?RUN not set}" "${SRC:?SRC not set}"
[[ -d "$SRC" ]] || { echo "no source tree at $SRC"; exit 0; }
mkdir -p "$RUN"

for item in data out PSF; do
    if [[ -e "$SRC/$item" && ! -e "$RUN/$item" ]]; then
        echo "moving $item"
        mv "$SRC/$item" "$RUN/$item"
    elif [[ -e "$SRC/$item" ]]; then
        echo "$item already at destination, leaving source in place"
    fi
done

echo "=== new tree:"; du -sh $RUN/* 2>/dev/null | sort -rh | head -8
echo "=== old tree:"; du -sh $SRC 2>/dev/null
echo MIGRATE_DONE
