#!/usr/bin/env bash
# Unmount a CANFAR SSHFS mount.
#
# Usage:
#   ./canfar-umount.sh            # unmount $CANFAR_MOUNT
#   ./canfar-umount.sh ~/uds      # unmount a specific mount point

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/canfar-common.sh"

mount_point="${1:-$CANFAR_MOUNT}"

if ! mount | grep -q " on ${mount_point} "; then
    echo "not mounted: $mount_point"
    exit 0
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
    umount "$mount_point" 2>/dev/null || diskutil unmount force "$mount_point"
else
    fusermount -u "$mount_point"
fi

echo "unmounted: $mount_point"
