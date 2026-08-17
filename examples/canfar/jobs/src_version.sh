#!/bin/bash
# Report the commit of this run's checkout, for `submit.check_src_current`.
#
# Asks git in the checkout rather than reading a stamped file: the repository
# cannot disagree with itself, while a stamp has to be rewritten by every path
# that moves the source and is silently wrong if one forgets. The OzStar
# toolkit asks the same question over ssh; this is the container equivalent,
# sized to schedule immediately.
set -euo pipefail
: "${RUN:?RUN not set}"
RUNNUM=${RUNNUM:-1}
SRC=$RUN/run$RUNNUM/config/mophongo
if [ -d "$SRC/.git" ]; then
    echo "SRC_VERSION $(git -C "$SRC" log --oneline -1)"
else
    echo "no checkout at $SRC"
fi
