#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    printf '%s\n' \
        "Usage: $0 [OPTIONS]" \
        "" \
        "Build a reproducible ppocr-rs tar.gz release bundle." \
        "" \
        "Options:" \
        "  --backend NAME       Inference backend (currently: cpu)" \
        "  --target TRIPLE      Rust target triple (default: rustc host)" \
        "  --version VERSION    Release version (default: workspace version)" \
        "  --output-dir PATH    Artifact directory (default: dist)" \
        "  --skip-build         Package an existing release build" \
        "  -h, --help           Show this help"
}

fail() {
    printf 'package-release: %s\n' "$*" >&2
    exit 1
}

script_dir=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(CDPATH='' cd -- "$script_dir/.." && pwd -P)
backend=cpu
target=
version=
output_dir=$repo_root/dist
skip_build=false

while (($# > 0)); do
    case "$1" in
        --backend)
            (($# >= 2)) || fail "--backend requires a value"
            backend=$2
            shift 2
            ;;
        --target)
            (($# >= 2)) || fail "--target requires a value"
            target=$2
            shift 2
            ;;
        --version)
            (($# >= 2)) || fail "--version requires a value"
            version=$2
            shift 2
            ;;
        --output-dir)
            (($# >= 2)) || fail "--output-dir requires a value"
            output_dir=$2
            shift 2
            ;;
        --skip-build)
            skip_build=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "unknown option: $1"
            ;;
    esac
done

command -v cargo >/dev/null || fail "cargo is required"
command -v rustc >/dev/null || fail "rustc is required"
command -v msgfmt >/dev/null || fail "GNU msgfmt is required"
command -v sha256sum >/dev/null || fail "sha256sum is required"
command -v tar >/dev/null || fail "GNU tar is required"
command -v gzip >/dev/null || fail "gzip is required"
command -v readelf >/dev/null || fail "readelf is required for Linux packaging"

manifest_version() {
    awk '
        /^\[package\]$/ { in_package = 1; next }
        in_package && /^\[/ { exit }
        in_package && /^version = "/ {
            value = $0
            sub(/^version = "/, "", value)
            sub(/"$/, "", value)
            print value
            exit
        }
    ' "$1"
}

workspace_version=$(manifest_version "$repo_root/apps/ppocr-gui/Cargo.toml")
[[ -n "$workspace_version" ]] || fail "could not read the workspace version"
for manifest in \
    "$repo_root/apps/ppocr-worker/Cargo.toml" \
    "$repo_root/crates/ppocr-app-protocol/Cargo.toml" \
    "$repo_root/crates/ppocr-i18n/Cargo.toml" \
    "$repo_root/crates/ppocr-pipeline/Cargo.toml" \
    "$repo_root/crates/ppocr-screenshot/Cargo.toml"; do
    crate_version=$(manifest_version "$manifest")
    [[ "$crate_version" == "$workspace_version" ]] ||
        fail "workspace version mismatch in ${manifest#"$repo_root/"}: $crate_version != $workspace_version"
done

if [[ -z "$version" ]]; then
    version=$workspace_version
fi
[[ "$version" == "$workspace_version" ]] ||
    fail "requested version $version does not match workspace version $workspace_version"
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z][0-9A-Za-z.-]*)?$ ]] ||
    fail "invalid release version: $version"

if [[ -z "$target" ]]; then
    target=$(rustc -vV | sed -n 's/^host: //p')
fi
[[ -n "$target" ]] || fail "could not determine the Rust host target"

# Keep platform selection separate from provider selection. New operating
# systems and ORT providers can be added as independent cases without changing
# the archive layout or release workflow contract.
case "$target" in
    x86_64-unknown-linux-gnu)
        platform=linux
        architecture=x86_64
        ;;
    aarch64-unknown-linux-gnu)
        platform=linux
        architecture=aarch64
        ;;
    *)
        fail "release packaging is not implemented for target $target"
        ;;
esac

case "$backend" in
    cpu)
        cargo_features=()
        ;;
    *)
        fail "release packaging is not implemented for backend $backend"
        ;;
esac

case "${CARGO_TARGET_DIR:-}" in
    "") cargo_target_dir=$repo_root/target ;;
    /*) cargo_target_dir=$CARGO_TARGET_DIR ;;
    *) cargo_target_dir=$repo_root/$CARGO_TARGET_DIR ;;
esac
build_dir=$cargo_target_dir/$target/release

if [[ "$skip_build" == false ]]; then
    build_command=(
        cargo build --locked --release --target "$target"
        --package ppocr-gui --bin ppocr-gui
        --package ppocr-worker --bin ppocr-worker
        --package ppocr-screenshot --bin ppocr-screencopy
    )
    if ((${#cargo_features[@]} > 0)); then
        build_command+=(--features "$(IFS=,; printf '%s' "${cargo_features[*]}")")
    fi

    # A release must not silently inherit a developer's custom ORT build or a
    # GPU provider from the environment. The CPU package uses ort's pinned,
    # downloaded runtime, which may be linked statically or copied as a shared
    # library depending on the target distribution.
    (
        cd "$repo_root"
        env \
            -u ORT_LIB_PATH \
            -u ORT_LIB_LOCATION \
            -u ORT_LIB_PROFILE \
            -u ORT_PREFER_DYNAMIC_LINK \
            -u ORT_SKIP_DOWNLOAD \
            -u ORT_OFFLINE \
            -u ORT_CUDA_VERSION \
            "${build_command[@]}"
    )
fi

for binary in ppocr-gui ppocr-worker ppocr-screencopy; do
    [[ -x "$build_dir/$binary" ]] || fail "missing release binary: $build_dir/$binary"
done

shopt -s nullglob
ort_libraries=("$build_dir"/libonnxruntime.so*)
provider_libraries=("$build_dir"/libonnxruntime_providers_shared.so*)
shopt -u nullglob
if readelf -d "$build_dir/ppocr-worker" | grep -q 'Shared library: \[libonnxruntime\.so'; then
    ort_linkage=dynamic
    ((${#ort_libraries[@]} > 0)) ||
        fail "the dynamically linked build did not copy libonnxruntime into $build_dir"
else
    ort_linkage=static
fi

mkdir -p "$output_dir"
output_dir=$(CDPATH='' cd -- "$output_dir" && pwd -P)
stage_dir=$(mktemp -d "${TMPDIR:-/tmp}/ppocr-release.XXXXXXXX")
cleanup() {
    rm -rf -- "$stage_dir"
}
trap cleanup EXIT

archive_root=ppocr-rs-$version-$target-$backend
bundle=$stage_dir/$archive_root
install -d -m 0755 "$bundle/bin" "$bundle/lib" "$bundle/share/locale/zh_CN/LC_MESSAGES"
install -m 0755 \
    "$build_dir/ppocr-gui" \
    "$build_dir/ppocr-worker" \
    "$build_dir/ppocr-screencopy" \
    "$bundle/bin/"
if ((${#ort_libraries[@]} > 0)); then
    cp -a -- "${ort_libraries[@]}" "$bundle/lib/"
fi
if ((${#provider_libraries[@]} > 0)); then
    cp -a -- "${provider_libraries[@]}" "$bundle/lib/"
fi

install -m 0755 "$repo_root/packaging/ppocr-gui-launcher.sh" "$bundle/ppocr-gui"
install -m 0644 "$repo_root/packaging/README.txt" "$bundle/README.txt"
install -m 0644 \
    "$repo_root/packaging/ONNX-Runtime-LICENSE.txt" \
    "$bundle/ONNX-Runtime-LICENSE.txt"
msgfmt --check --check-format \
    --output-file "$bundle/share/locale/zh_CN/LC_MESSAGES/ppocr-rs.mo" \
    "$repo_root/po/zh_CN.po"

source_commit=$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || printf 'unknown')
source_date_epoch=${SOURCE_DATE_EPOCH:-$(git -C "$repo_root" log -1 --format=%ct 2>/dev/null || printf '0')}
[[ "$source_date_epoch" =~ ^[0-9]+$ ]] || fail "SOURCE_DATE_EPOCH must be an integer"
source_dirty=unknown
if git -C "$repo_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if [[ -n "$(git -C "$repo_root" status --porcelain --untracked-files=normal)" ]]; then
        source_dirty=true
    else
        source_dirty=false
    fi
fi
rustc_version=$(rustc --version)
{
    printf 'name=ppocr-rs\n'
    printf 'version=%s\n' "$version"
    printf 'target=%s\n' "$target"
    printf 'platform=%s\n' "$platform"
    printf 'architecture=%s\n' "$architecture"
    printf 'backend=%s\n' "$backend"
    printf 'ort_linkage=%s\n' "$ort_linkage"
    printf 'source_commit=%s\n' "$source_commit"
    printf 'source_dirty=%s\n' "$source_dirty"
    printf 'source_date_epoch=%s\n' "$source_date_epoch"
    printf 'rustc=%s\n' "$rustc_version"
} >"$bundle/PACKAGE-METADATA"

checksums=$stage_dir/SHA256SUMS
(
    cd "$bundle"
    LC_ALL=C find . -type f ! -name SHA256SUMS -print0 |
        LC_ALL=C sort -z |
        xargs -0 sha256sum
) >"$checksums"
install -m 0644 "$checksums" "$bundle/SHA256SUMS"
find "$bundle" -exec touch -h -d "@$source_date_epoch" {} +

archive=$output_dir/$archive_root.tar.gz
temporary_archive=$archive.tmp
rm -f -- "$temporary_archive"
tar \
    --sort=name \
    --format=gnu \
    --mtime="@$source_date_epoch" \
    --owner=0 \
    --group=0 \
    --numeric-owner \
    -C "$stage_dir" \
    -cf - "$archive_root" |
    gzip -n -9 >"$temporary_archive"
mv -f -- "$temporary_archive" "$archive"
(
    cd "$output_dir"
    sha256sum "$(basename -- "$archive")" >"$(basename -- "$archive").sha256"
)

printf 'Created %s\n' "$archive"
printf 'Created %s.sha256\n' "$archive"
