#!/usr/bin/env bash
# Mount a CANFAR ARC directory locally with SSHFS.
#
# Usage:
#   ./canfar-mount.sh                              # mount $CANFAR_REMOTE at $CANFAR_MOUNT
#   ./canfar-mount.sh /projects/minerva/uds        # mount that path at $CANFAR_MOUNT
#   ./canfar-mount.sh /projects/minerva/uds ~/uds  # explicit mount point
#
# The gateway is chrooted at /arc, so remote paths omit the /arc prefix; a
# leading /arc is stripped for convenience.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/canfar-common.sh"

remote="${1:-$CANFAR_REMOTE}"
mount_point="${2:-$CANFAR_MOUNT}"

# The SFTP gateway is chrooted at /arc: /arc/projects/x is served as /projects/x.
remote="${remote#/arc}"

require_user
command -v sshfs >/dev/null || die "sshfs not found. Install FUSE-T (kext-free, unlike macFUSE): brew tap macos-fuse-t/homebrew-cask && brew install --cask fuse-t && brew trust --cask macos-fuse-t/cask/fuse-t-sshfs && brew install --cask fuse-t-sshfs"

if mount | grep -q " on ${mount_point} "; then
    echo "already mounted: $mount_point"
    exit 0
fi

mkdir -p "$mount_point"
[[ -z "$(ls -A "$mount_point")" ]] || die "mount point not empty: $mount_point"

# CountMax=3 (45s), not 10 (150s): FUSE-T serves the mount to macOS as NFS, and
# the NFS client gives up on an unresponsive server after
# vfs.generic.nfs.client.initialdowndelay. sshfs has to notice the dead link and
# reconnect inside that window or macOS declares the volume down.
opts="reconnect,ServerAliveInterval=15,ServerAliveCountMax=3,auto_cache,noappledouble"
if [[ "$(uname -s)" == "Darwin" ]]; then
    opts="${opts},defer_permissions,volname=$(basename "$mount_point")"
fi

echo "mounting ${CANFAR_USER}@${CANFAR_HOST}:${remote} -> ${mount_point}"
sshfs -p "$CANFAR_PORT" -o "$opts" \
    "${CANFAR_USER}@${CANFAR_HOST}:${remote}" "$mount_point"

echo "mounted. contents:"
ls -la "$mount_point" | head -20
