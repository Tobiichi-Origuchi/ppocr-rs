#!/bin/sh
set -eu

bundle_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd -P)
if [ -z "${PPOCR_LOCALEDIR:-}" ]; then
    PPOCR_LOCALEDIR=$bundle_dir/share/locale
    export PPOCR_LOCALEDIR
fi
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH="$bundle_dir/lib:$LD_LIBRARY_PATH"
else
    export LD_LIBRARY_PATH="$bundle_dir/lib"
fi

exec "$bundle_dir/bin/ppocr-gui" "$@"
