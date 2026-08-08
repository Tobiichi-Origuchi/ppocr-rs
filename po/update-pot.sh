#!/usr/bin/env sh
set -eu

project_root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_root"

xgettext \
    --language=Rust \
    --from-code=UTF-8 \
    --keyword='tr!' \
    --keyword='ntr!:1,2' \
    --add-comments=TRANSLATORS \
    --files-from=po/POTFILES.in \
    --directory=. \
    --package-name=ppocr-rs \
    --package-version=0.1.0 \
    --msgid-bugs-address=ppocr-rs \
    --output=po/ppocr-rs.pot

while IFS= read -r language; do
    case "$language" in
        ''|'#'*) continue ;;
    esac
    msgmerge --update --backup=none "po/$language.po" po/ppocr-rs.pot
done < po/LINGUAS

if msgattrib --untranslated --no-obsolete po/zh_CN.po | grep -q .; then
    printf '%s\n' 'po/zh_CN.po contains untranslated messages' >&2
    exit 1
fi
if msgattrib --only-fuzzy --no-obsolete po/zh_CN.po | grep -q .; then
    printf '%s\n' 'po/zh_CN.po contains fuzzy messages that require review' >&2
    exit 1
fi

msgfmt --check --check-format --output-file=/dev/null po/zh_CN.po
