#!/bin/bash
#SBATCH --job-name=moph-push
#SBATCH --partition=datamover
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=3g
#SBATCH --time=24:00:00
#
# Push a directory from /fred to CANFAR arc, in parallel, skipping what is
# already there.
#
# A SLURM job on the datamover partition, so it survives the laptop being shut,
# an ssh drop and the end of the session - the transfer is the queue's problem,
# not a terminal's. Ordinary compute nodes cannot do this (no route to CANFAR)
# and the login node should not hold a multi-hour transfer open.
#
# `vsync` would be the obvious tool and is not used: measured end to end, a
# single stream to arc runs at 1.25 MB/s while six parallel streams reach
# 14 MB/s aggregate. The bottleneck is per-connection latency to Canada, not
# bandwidth, so the win comes from concurrency. This walks the tree, diffs it
# against the destination and fans the missing files out over $JOBS streams.
#
# Environment:
#   SRCDIR    local directory to push            (required)
#   DEST      arc: URI to push into              (required)
#   JOBS      parallel streams, default 6
#   COMPRESS  1 to gzip before sending           (default 0)
#   DRYRUN    1 to list what would be sent and stop
#
# COMPRESS pays only when the data are mostly zeros. Residuals, stamps and
# segmentation-like products compress many-fold; ePSF grids are dense float
# arrays and measured 1.1x, where the CPU and the temp copy cost more than the
# 10% saved. It is off by default for that reason - turn it on per transfer.
set -uo pipefail
: "${SRCDIR:?SRCDIR not set}" "${DEST:?DEST not set}" "${VOS:?VOS not set}"
JOBS=${JOBS:-6}
COMPRESS=${COMPRESS:-0}
DRYRUN=${DRYRUN:-0}

export PATH="$VOS/bin:$PATH"
unset PYTHONPATH
command -v vcp > /dev/null || { echo "no vcp under $VOS" >&2; exit 1; }
[[ -s $HOME/.ssl/cadcproxy.pem ]] || {
    echo "no CADC certificate at $HOME/.ssl/cadcproxy.pem" >&2
    echo "run 'python submit.py cert' from the laptop; it expires after 10 days" >&2
    exit 1
}
# A certificate that dies mid-transfer fails every remaining file with a
# permission error rather than stopping cleanly, so say how long is left.
echo "certificate: $(openssl x509 -in "$HOME/.ssl/cadcproxy.pem" -noout -enddate)"
echo "source:      $SRCDIR"
echo "destination: $DEST"
echo "streams:     $JOBS   compress: $COMPRESS"

SRCDIR=${SRCDIR%/}
DEST=${DEST%/}
TMP=$(mktemp -d "${SRCDIR%/*}/.push_arc.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

# --- what is already there --------------------------------------------------
cd "$SRCDIR"
find . -type f -printf "%P\n" | sort > "$TMP/local.txt"
echo "    $(wc -l < "$TMP/local.txt") files locally"

# One vls per directory of the local tree. `vls -R` is not used: it returns
# nothing here (silently, which cost a dry run to notice) and an empty listing
# is indistinguishable from an empty destination - so every file would be
# re-sent. Listing only the directories we are about to write into is exact,
# and the round trips are per directory rather than per file.
echo "=== listing destination"
sed 's|/[^/]*$||' "$TMP/local.txt" | sed 's|^[^/]*$|.|' | sort -u > "$TMP/dirs.txt"
: > "$TMP/remote.txt"
while read -r d; do
    if [[ $d == . ]]; then
        vls "$DEST/" 2>/dev/null | sed 's|/$||' | awk 'NF' >> "$TMP/remote.txt"
    else
        vls "$DEST/$d/" 2>/dev/null | sed 's|/$||' | awk -v p="$d" 'NF {print p "/" $0}' \
            >> "$TMP/remote.txt"
    fi
done < "$TMP/dirs.txt"
sort -u -o "$TMP/remote.txt" "$TMP/remote.txt"
echo "    $(wc -l < "$TMP/remote.txt") entries already on arc"

comm -23 "$TMP/local.txt" "$TMP/remote.txt" > "$TMP/missing.txt"
n=$(wc -l < "$TMP/missing.txt")
bytes=$(while read -r f; do stat -c %s "$f"; done < "$TMP/missing.txt" | awk '{t+=$1} END {print t+0}')
printf "=== to send: %d file(s), %.1f GB\n" "$n" "$(echo "$bytes" | awk '{print $1/1073741824}')"
[[ $n -eq 0 ]] && { echo "nothing to do"; echo PUSH_DONE; exit 0; }
if [[ $DRYRUN == 1 ]]; then
    head -20 "$TMP/missing.txt"; echo "(dry run)"; exit 0
fi

# --- destination directories ------------------------------------------------
# vcp does not create intermediate directories, and one vmkdir per file would
# be thousands of round trips, so make each distinct directory once.
sed 's|/[^/]*$||' "$TMP/missing.txt" | grep -v '^[^/]*$' | sort -u > "$TMP/newdirs.txt"
if [[ -s $TMP/newdirs.txt ]]; then
    echo "=== creating $(wc -l < "$TMP/newdirs.txt") remote directory/ies"
    while read -r d; do vmkdir -p "$DEST/$d" > /dev/null 2>&1; done < "$TMP/newdirs.txt"
fi

# --- send -------------------------------------------------------------------
send_one() {
    local rel="$1" src="$SRCDIR/$1" dst="$DEST/$1" tmp=""
    if [[ $COMPRESS == 1 && $rel != *.gz ]]; then
        tmp="$TMP/$(echo "$rel" | tr / _).gz"
        # -1: at 14 MB/s over the link even the fastest setting outruns it, so
        # spend no more CPU than necessary
        pigz -1 -c "$src" > "$tmp" 2>/dev/null || gzip -1 -c "$src" > "$tmp"
        src="$tmp"; dst="$dst.gz"
    fi
    for attempt in 1 2 3; do
        if vcp "$src" "$dst" > /dev/null 2>&1; then
            [[ -n $tmp ]] && rm -f "$tmp"
            echo "sent   $rel"
            return 0
        fi
        sleep $((attempt * 10))
    done
    [[ -n $tmp ]] && rm -f "$tmp"
    echo "FAILED $rel" >&2
    return 1
}
export -f send_one
export SRCDIR DEST TMP COMPRESS

start=$SECONDS
xargs -d '\n' -a "$TMP/missing.txt" -P "$JOBS" -I {} bash -c 'send_one "$@"' _ {} \
    | awk -v n="$n" '{c++; if (c % 25 == 0) printf "  %d/%d\n", c, n} END {printf "  %d/%d\n", c, n}'
elapsed=$((SECONDS - start))
printf "=== %d file(s), %.1f GB in %d min (%.1f MB/s)\n" \
    "$n" "$(echo "$bytes" | awk '{print $1/1073741824}')" "$((elapsed / 60))" \
    "$(echo "$bytes $elapsed" | awk '{print $1/1048576/($2>0?$2:1)}')"

# Project space is shared with the collaboration; a file only its owner can
# read is no use there. vcp leaves 0600, so open group/other read afterwards.
echo "=== opening read permission"
vchmod o+r "$DEST" > /dev/null 2>&1 || true
echo PUSH_DONE
