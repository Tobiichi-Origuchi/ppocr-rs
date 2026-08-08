#!/usr/bin/env bash
set -euo pipefail

readonly GETTEXT_VERSION=1.0
readonly GETTEXT_SHA256=71132a3fb71e68245b8f2ac4e9e97137d3e5c02f415636eb508ae607bc01add7

if [[ $# -ne 1 || -z "$1" ]]; then
    printf 'usage: %s PREFIX\n' "$0" >&2
    exit 2
fi

if command -v xgettext >/dev/null && xgettext --help | grep -F 'Rust' >/dev/null; then
    printf 'Using %s\n' "$(xgettext --version | sed -n '1p')"
    exit 0
fi

readonly install_prefix=$1
build_root=$(mktemp -d)
readonly build_root
trap 'rm -rf -- "$build_root"' EXIT

readonly archive="$build_root/gettext-$GETTEXT_VERSION.tar.xz"
readonly configure_log="$build_root/configure.log"
readonly build_log="$build_root/build.log"
printf 'Building GNU gettext %s with Rust extraction support\n' "$GETTEXT_VERSION"
curl \
    --fail \
    --location \
    --retry 3 \
    --silent \
    --show-error \
    --output "$archive" \
    "https://ftp.gnu.org/gnu/gettext/gettext-$GETTEXT_VERSION.tar.xz"
printf '%s  %s\n' "$GETTEXT_SHA256" "$archive" | sha256sum --check

tar --extract --xz --file "$archive" --directory "$build_root"
cd "$build_root/gettext-$GETTEXT_VERSION"
if ! ./configure \
    --prefix="$install_prefix" \
    --disable-acl \
    --disable-csharp \
    --disable-java \
    --disable-shared \
    --enable-static \
    --without-emacs \
    --without-git \
    >"$configure_log" 2>&1; then
    printf 'gettext configure failed; the final log lines follow:\n' >&2
    tail --lines 200 "$configure_log" >&2
    exit 1
fi
if ! make --jobs="$(nproc)" >"$build_log" 2>&1; then
    printf 'gettext build failed; the final log lines follow:\n' >&2
    tail --lines 200 "$build_log" >&2
    exit 1
fi

install --directory "$install_prefix/bin"
install --mode 0755 gettext-tools/src/xgettext "$install_prefix/bin/xgettext"
"$install_prefix/bin/xgettext" --help | grep -F 'Rust' >/dev/null
"$install_prefix/bin/xgettext" --version | sed -n '1p'
