#!/usr/bin/env bash
# Get / refresh the CADC proxy certificate used by the vos tools (vcp, vsync, vls).
# The certificate lands in ~/.ssl/cadcproxy.pem and is valid for 10 days.
#
# Usage:
#   ./canfar-cert.sh          # refresh only if missing or older than 9 days
#   ./canfar-cert.sh --force  # always fetch a new one

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/canfar-common.sh"

cert="$HOME/.ssl/cadcproxy.pem"
force="${1:-}"

require_user
command -v cadc-get-cert >/dev/null || die "cadc-get-cert not found. Install: pip install vos"

if [[ "$force" != "--force" && -f "$cert" ]]; then
    # -mmin +12960 == modified more than 9 days ago (cert is valid for 10)
    if [[ -z "$(find "$cert" -mmin +12960 2>/dev/null)" ]]; then
        echo "certificate still fresh: $cert"
        exit 0
    fi
fi

echo "fetching CADC certificate for $CANFAR_USER (prompts for your CADC password)"
cadc-get-cert -u "$CANFAR_USER"
