#!/usr/bin/env bash
# Copy a directory tree between CANFAR ARC storage and local disk with vsync.
# Works in either direction; skips files that are already present and identical.
# No archive is created - files transfer individually.
#
# Usage:
#   ./canfar-sync.sh /arc/projects/minerva/uds/catalogs/mophongo_tests ./mophongo_tests
#   ./canfar-sync.sh ./results /arc/projects/minerva/uds/catalogs/mophongo_tests
#
# Either side may be given as /arc/... or arc:... - both are normalised.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/canfar-common.sh"

[[ $# -eq 2 ]] || die "usage: $(basename "$0") <source> <destination>"

src="$1"
dst="$2"

command -v vsync >/dev/null || die "vsync not found. Install: pip install vos"

# Normalise ARC paths to VOSpace URIs; leave local paths alone.
[[ "$src" == /arc/* || "$src" == arc:* ]] && src="$(to_vos_uri "$src")"
[[ "$dst" == /arc/* || "$dst" == arc:* ]] && dst="$(to_vos_uri "$dst")"

"$(dirname "${BASH_SOURCE[0]}")/canfar-cert.sh"

# vsync wants trailing slashes on directories.
[[ "$src" == */ ]] || src="${src}/"
[[ "$dst" == */ ]] || dst="${dst}/"

[[ "$dst" == arc:* ]] || mkdir -p "$dst"

echo "vsync $src -> $dst"
vsync "$src" "$dst"
