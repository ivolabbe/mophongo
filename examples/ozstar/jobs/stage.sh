#!/bin/bash
#SBATCH --job-name=moph-stage
#SBATCH --partition=datamover
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=3g
# 3 GB, not 4: a datamover node advertises 4000 MB and "--mem=4g" is 4096, which
# is refused outright as an impossible node configuration.
#SBATCH --time=24:00:00
#
# Copy one field's inputs from CANFAR arc into $DATA, decompressing the
# gzipped ones on the way.
#
# This runs on the *datamover* partition. Ordinary compute nodes have no
# internet, so they cannot reach arc at all; datamover nodes can, and unlike the
# login node they are meant to hold a transfer open for hours. They have only
# 4 GB of RAM, which is why the decompression is streamed rather than buffered,
# and no /apps module tree, which is why the transfer tools live in venv-vos,
# built from /usr/bin/python3 - the module python does not exist on these nodes.
#
# The partition has to be requested on the sbatch command line: a site plugin
# reassigns the partition of a job that only names it in a #SBATCH directive,
# and the job then lands on a compute node with no route to CANFAR.
#
# $CFGS is a space-separated list of config names whose manifests to process.
# Bands of a field share the F444W mosaic, the weight and the segmap, so all of
# a field's bands belong in one job: the destination list is deduplicated and
# each file transfers once. Already-present files are skipped, so a job that
# ran out of time can simply be resubmitted.
set -euo pipefail
: "${BASE:?BASE not set}" "${RUN:?RUN not set}" "${CFGDIR:?CFGDIR not set}" "${VENV:?VENV not set}" "${SRC:?SRC not set}" "${CFGS:?CFGS not set}"
JOBS=${JOBS:-6}

export PATH="$VOS/bin:$PATH"
unset PYTHONPATH
command -v vcp > /dev/null || { echo "no vcp; run 'submit.py setup' first" >&2; exit 1; }

[[ -s $HOME/.ssl/cadcproxy.pem ]] || {
    echo "no CADC certificate at $HOME/.ssl/cadcproxy.pem" >&2
    echo "run 'python submit.py cert' from the laptop (it expires after 10 days)" >&2
    exit 1
}

D=$DATA
mkdir -p "$D"

# A cancelled or timed-out job leaves its ".<name>.<pid>" temporaries behind,
# and they are several GB each. Sweep only the old ones: a concurrent staging
# job's temporaries look the same and are still being written.
find "$D" -maxdepth 1 -name '.*.[0-9]*' -mmin +360 -delete 2>/dev/null || true

manifests=()
for cfg in $CFGS; do
    m=$CFGDIR/${cfg}_stage.tsv
    [[ -s $m ]] || { echo "no manifest: $m" >&2; exit 1; }
    manifests+=("$m")
done

# One file: fetch to a private temp name, decompress if gzipped, move into
# place. Concurrent jobs of different fields can want the same file, so the
# move is a no-clobber move and a loser just drops its copy.
fetch_one() {
    local src="$1" dst="$2" tmp
    if [[ -s $D/$dst ]]; then echo "have     $dst"; return 0; fi
    tmp="$D/.$dst.$$"
    for attempt in 1 2 3; do
        if [[ $src == *.gz ]]; then
            echo "fetch    $dst (gz, attempt $attempt)"
            vcp "$src" "$tmp.gz" && gunzip -c "$tmp.gz" > "$tmp" && rm -f "$tmp.gz" && break
        else
            echo "fetch    $dst (attempt $attempt)"
            vcp "$src" "$tmp" && break
        fi
        rm -f "$tmp" "$tmp.gz"
        [[ $attempt == 3 ]] && { echo "FAILED   $dst" >&2; return 1; }
        sleep 20
    done
    if [[ -s $D/$dst ]]; then
        echo "         ($dst arrived meanwhile, discarding)"
        rm -f "$tmp"
    else
        mv -n "$tmp" "$D/$dst"
        rm -f "$tmp"
    fi
}
export -f fetch_one
export D

# One CANFAR stream is much slower than the link, hence several at once.
# -d '\n' so xargs splits on lines only and leaves quoting alone.
sort -u "${manifests[@]}" | awk -F'\t' '!seen[$2]++' | tr '\t' '\n' |
    xargs -d '\n' -P "$JOBS" -n 2 bash -c 'fetch_one "$0" "$1"'

echo "=== staged:"
cut -f2 "${manifests[@]}" | sort -u | while read -r f; do
    [[ -n $f ]] && ls -lh "$D/$f"
done
du -sh "$D"
echo STAGE_DONE
