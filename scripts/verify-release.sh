#!/usr/bin/env bash
set -Eeuo pipefail

fail() {
    printf 'verify-release: %s\n' "$*" >&2
    exit 1
}

if (($# != 1)); then
    printf 'Usage: %s ARCHIVE.tar.gz\n' "$0" >&2
    exit 2
fi

for command in find readelf sha256sum tar; do
    command -v "$command" >/dev/null || fail "$command is required"
done

archive_dir=$(CDPATH='' cd -- "$(dirname -- "$1")" && pwd -P)
archive=$archive_dir/$(basename -- "$1")
[[ -f "$archive" ]] || fail "archive does not exist: $archive"

checksum_file=$archive.sha256
if [[ -f "$checksum_file" ]]; then
    (
        cd "$archive_dir"
        sha256sum --check "$(basename -- "$checksum_file")"
    )
fi

while IFS= read -r entry; do
    case "$entry" in
        /*|../*|*/../*|*/..)
            fail "unsafe archive path: $entry"
            ;;
    esac
done < <(tar -tzf "$archive")

mapfile -t top_levels < <(
    tar -tzf "$archive" |
        sed 's#^\./##; s#/.*##' |
        LC_ALL=C sort -u
)
((${#top_levels[@]} == 1)) || fail "archive must contain exactly one top-level directory"
[[ -n "${top_levels[0]}" ]] || fail "archive has an empty top-level directory"

temporary_dir=$(mktemp -d "${TMPDIR:-/tmp}/ppocr-verify.XXXXXXXX")
cleanup() {
    rm -rf -- "$temporary_dir"
}
trap cleanup EXIT
tar -xzf "$archive" -C "$temporary_dir"
bundle=$temporary_dir/${top_levels[0]}

for file in \
    ppocr-gui \
    bin/ppocr-gui \
    bin/ppocr-worker \
    bin/ppocr-screencopy \
    share/locale/zh_CN/LC_MESSAGES/ppocr-rs.mo \
    PACKAGE-METADATA \
    SHA256SUMS \
    README.txt \
    ONNX-Runtime-LICENSE.txt; do
    [[ -e "$bundle/$file" ]] || fail "bundle is missing $file"
done
for executable in ppocr-gui bin/ppocr-gui bin/ppocr-worker bin/ppocr-screencopy; do
    [[ -x "$bundle/$executable" ]] || fail "$executable is not executable"
done

while IFS= read -r -d '' link; do
    target=$(readlink -- "$link")
    case "$target" in
        /*|../*|*/../*|*/..)
            fail "unsafe symbolic link: ${link#"$bundle/"} -> $target"
            ;;
    esac
    [[ -e "$link" ]] || fail "broken symbolic link: ${link#"$bundle/"}"
done < <(find "$bundle" -type l -print0)

(
    cd "$bundle"
    sha256sum --check SHA256SUMS
)
grep -qx 'backend=cpu' "$bundle/PACKAGE-METADATA" || fail "bundle is not marked as CPU-only"
grep -qx 'platform=linux' "$bundle/PACKAGE-METADATA" || fail "bundle is not marked as Linux"
if find "$bundle/lib" -maxdepth 1 \
    \( -iname '*cuda*' -o -iname '*tensorrt*' -o -iname '*rocm*' \) |
    grep -q .; then
    fail "CPU bundle contains a GPU provider library"
fi

shopt -s nullglob
ort_libraries=("$bundle"/lib/libonnxruntime.so*)
shopt -u nullglob

ort_linkage=$(sed -n 's/^ort_linkage=//p' "$bundle/PACKAGE-METADATA")
case "$ort_linkage" in
    static)
        if readelf -d "$bundle/bin/ppocr-worker" |
            grep -q 'Shared library: \[libonnxruntime\.so'; then
            fail "bundle metadata claims static ORT but the worker needs a shared ORT library"
        fi
        ;;
    dynamic)
        ((${#ort_libraries[@]} > 0)) || fail "dynamic ORT bundle does not contain libonnxruntime"
        readelf -d "$bundle/bin/ppocr-worker" |
            grep -q 'Shared library: \[libonnxruntime\.so' ||
            fail "bundle metadata claims dynamic ORT but the worker has no shared ORT dependency"
        ;;
    *)
        fail "invalid ort_linkage value in PACKAGE-METADATA: $ort_linkage"
        ;;
esac

sh -n "$bundle/ppocr-gui"
if command -v msgunfmt >/dev/null; then
    msgunfmt "$bundle/share/locale/zh_CN/LC_MESSAGES/ppocr-rs.mo" >/dev/null
fi
if command -v ldd >/dev/null; then
    loader_output=$(LD_LIBRARY_PATH="$bundle/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        ldd "$bundle/bin/ppocr-worker" 2>&1) || fail "ldd rejected the packaged worker"
    if grep -q 'not found' <<<"$loader_output"; then
        printf '%s\n' "$loader_output" >&2
        fail "packaged worker has unresolved shared libraries"
    fi
fi

worker_output=$temporary_dir/worker-output.bin
PPOCR_LOCALEDIR="$bundle/share/locale" \
LD_LIBRARY_PATH="$bundle/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$bundle/bin/ppocr-worker" </dev/null >"$worker_output"
[[ -s "$worker_output" ]] || fail "packaged worker did not emit its protocol greeting"

printf 'Verified %s\n' "$archive"
