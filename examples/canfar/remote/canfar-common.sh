#!/usr/bin/env bash
# Shared config loading for the canfar-*.sh helpers. Sourced, not executed.

set -euo pipefail

_canfar_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults; overridden by canfar.conf, then by the environment.
CANFAR_HOST_DEFAULT="ws-uv.canfar.net"
CANFAR_PORT_DEFAULT="64022"
CANFAR_REMOTE_DEFAULT="/projects/minerva"
CANFAR_MOUNT_DEFAULT="$HOME/canfar"
# The vos tools (vcp, vsync, vls, cadc-get-cert) live in their own venv, the same
# one MINERVA/data/stage/stage_canfar.py uses. They are not on the default PATH.
CANFAR_VENV_DEFAULT="$HOME/.venvs/canfar"

if [[ -f "$_canfar_dir/canfar.conf" ]]; then
    # shellcheck disable=SC1091
    source "$_canfar_dir/canfar.conf"
fi

CANFAR_HOST="${CANFAR_HOST:-$CANFAR_HOST_DEFAULT}"
CANFAR_PORT="${CANFAR_PORT:-$CANFAR_PORT_DEFAULT}"
CANFAR_REMOTE="${CANFAR_REMOTE:-$CANFAR_REMOTE_DEFAULT}"
CANFAR_MOUNT="${CANFAR_MOUNT:-$CANFAR_MOUNT_DEFAULT}"
CANFAR_VENV="${CANFAR_VENV:-$CANFAR_VENV_DEFAULT}"

[[ -d "$CANFAR_VENV/bin" ]] && PATH="$CANFAR_VENV/bin:$PATH"

die() { echo "error: $*" >&2; exit 1; }

require_user() {
    [[ -n "${CANFAR_USER:-}" && "$CANFAR_USER" != "your_cadc_username" ]] ||
        die "CANFAR_USER not set. Copy canfar.conf.example to canfar.conf and edit it, or export CANFAR_USER."
}

# /arc/projects/minerva/uds  ->  arc:projects/minerva/uds  (VOSpace URI form)
to_vos_uri() {
    local p="$1"
    case "$p" in
        arc:*) echo "$p" ;;
        /arc/*) echo "arc:${p#/arc/}" ;;
        *) die "path must be absolute under /arc (got: $p)" ;;
    esac
}
