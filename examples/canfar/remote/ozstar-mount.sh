#!/usr/bin/env bash
# Mount the OzStar project directory locally with SSHFS.
#
# The counterpart of canfar-mount.sh beside it. Simpler, because OzStar is a
# plain ssh host: no chrooted gateway, no non-standard port, and the remote
# path is absolute. It lives here rather than under ../../ozstar because the
# two mounts are one piece of machine setup and are read together.
#
# Usage:
#   ./ozstar-mount.sh                       # mount $OZSTAR_REMOTE at $OZSTAR_MOUNT
#   ./ozstar-mount.sh /fred/oz030/shared    # mount that path
#   ./ozstar-mount.sh /fred/oz030/x ~/x     # explicit mount point
#   ./ozstar-mount.sh -u                    # unmount
#
# Idempotent: an existing mount is left alone and reported, so it is safe to
# run from a LaunchAgent on a timer.
set -euo pipefail

OZSTAR_HOST="${OZSTAR_HOST:-nt.swin.edu.au}"
OZSTAR_USER="${OZSTAR_USER:-ilabbe}"
OZSTAR_REMOTE="${OZSTAR_REMOTE:-/fred/oz030/ilabbe}"
OZSTAR_MOUNT="${OZSTAR_MOUNT:-$HOME/ozstar}"

die() { echo "$*" >&2; exit 1; }

if [[ "${1:-}" == "-u" || "${1:-}" == "--unmount" ]]; then
    mount_point="${2:-$OZSTAR_MOUNT}"
    umount "$mount_point" 2>/dev/null || diskutil unmount force "$mount_point"
    echo "unmounted $mount_point"
    exit 0
fi

remote="${1:-$OZSTAR_REMOTE}"
mount_point="${2:-$OZSTAR_MOUNT}"

command -v sshfs >/dev/null || die "sshfs not found. Install FUSE-T (kext-free, unlike macFUSE): brew tap macos-fuse-t/homebrew-cask && brew install --cask fuse-t && brew install --cask fuse-t-sshfs"

# Presence in `mount` is the whole test, deliberately. Reading the mount point
# to prove it is alive looks tempting, but a FUSE-T mount made from a terminal
# is not always readable by a launchd job, so the read fails for a healthy
# mount and the timer then tears it down and rebuilds it every five minutes,
# breaking whatever was reading it. A genuinely hung mount is rare and is
# cleared by hand with -u.
if mount | grep -q " on ${mount_point} "; then
    echo "already mounted: $mount_point"
    exit 0
fi

mkdir -p "$mount_point"
[[ -z "$(ls -A "$mount_point" 2>/dev/null)" ]] || die "mount point not empty: $mount_point"

# CountMax=3 (45s), not 10 (150s): see the same note in canfar-mount.sh.
opts="reconnect,ServerAliveInterval=15,ServerAliveCountMax=3,auto_cache,noappledouble"
if [[ "$(uname -s)" == "Darwin" ]]; then
    opts="${opts},defer_permissions,volname=$(basename "$mount_point")"
fi

echo "mounting ${OZSTAR_USER}@${OZSTAR_HOST}:${remote} -> ${mount_point}"
sshfs -o "$opts" "${OZSTAR_USER}@${OZSTAR_HOST}:${remote}" "$mount_point"

echo "mounted. contents:"
ls "$mount_point" | head
