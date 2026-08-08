use std::{io, path::PathBuf};

use gettextrs::{LocaleCategory, bind_textdomain_codeset, bindtextdomain, setlocale, textdomain};
use thiserror::Error;

pub const DOMAIN: &str = "ppocr-rs";

#[doc(hidden)]
pub use formatx as __formatx;

#[derive(Debug, Error)]
pub enum InitError {
    #[error("the process locale is not available")]
    LocaleUnavailable,
    #[error("failed to initialize gettext: {0}")]
    Gettext(#[from] io::Error),
}

pub fn init() -> Result<(), InitError> {
    // SAFETY: every ppocr binary calls this once, on its main thread, before
    // GTK, worker threads, or any other locale-sensitive subsystem starts.
    if unsafe { setlocale(LocaleCategory::LcAll, "") }.is_none() {
        return Err(InitError::LocaleUnavailable);
    }
    bindtextdomain(DOMAIN, locale_dir())?;
    bind_textdomain_codeset(DOMAIN, "UTF-8")?;
    textdomain(DOMAIN)?;
    Ok(())
}

pub fn gettext(message: &str) -> String {
    gettextrs::gettext(message)
}

pub fn ngettext(singular: &str, plural: &str, count: u64) -> String {
    gettextrs::ngettext(singular, plural, count.min(u64::from(u32::MAX)) as u32)
}

pub fn locale_dir() -> PathBuf {
    std::env::var_os("PPOCR_LOCALEDIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("PPOCR_DEFAULT_LOCALEDIR")))
}

#[macro_export]
macro_rules! tr {
    ($message:literal $(,)?) => {
        $crate::gettext($message)
    };
    ($message:literal, $($arguments:tt)+) => {{
        let template = $crate::gettext($message);
        $crate::__formatx::formatx!(template, $($arguments)+)
            .expect("translated message contains invalid format placeholders")
    }};
}

#[macro_export]
macro_rules! ntr {
    ($singular:literal, $plural:literal, $count:expr $(,)?) => {
        $crate::ngettext($singular, $plural, ($count) as u64)
    };
    ($singular:literal, $plural:literal, $count:expr, $($arguments:tt)+) => {{
        let template = $crate::ngettext($singular, $plural, ($count) as u64);
        $crate::__formatx::formatx!(template, $($arguments)+)
            .expect("translated message contains invalid format placeholders")
    }};
}

#[cfg(test)]
mod tests {
    use std::{
        path::{Path, PathBuf},
        process::Command,
    };

    fn project_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn run(command: &mut Command) {
        let status = command.status().expect("failed to start GNU gettext tool");
        assert!(
            status.success(),
            "GNU gettext validation failed: {command:?}"
        );
    }

    #[test]
    fn catalogs_are_current_complete_and_well_formed() {
        let root = project_root();
        let generated =
            std::env::temp_dir().join(format!("ppocr-rs-i18n-{}.pot", std::process::id()));
        let compiled =
            std::env::temp_dir().join(format!("ppocr-rs-i18n-{}.mo", std::process::id()));
        run(Command::new("xgettext")
            .current_dir(&root)
            .args([
                "--language=Rust",
                "--from-code=UTF-8",
                "--keyword=tr!",
                "--keyword=ntr!:1,2",
                "--add-comments=TRANSLATORS",
                "--files-from=po/POTFILES.in",
                "--directory=.",
                "--package-name=ppocr-rs",
                "--package-version=0.1.0",
                "--msgid-bugs-address=ppocr-rs",
                "--output",
            ])
            .arg(&generated));

        let pot = root.join("po/ppocr-rs.pot");
        let chinese = root.join("po/zh_CN.po");
        run(Command::new("msgcmp")
            .arg("--use-untranslated")
            .arg(&pot)
            .arg(&generated));
        run(Command::new("msgcmp")
            .arg("--use-untranslated")
            .arg(&generated)
            .arg(&pot));
        run(Command::new("msgcmp").arg(&chinese).arg(&generated));
        run(Command::new("msgfmt")
            .args(["--check", "--check-format", "--output-file"])
            .arg(&compiled)
            .arg(&chinese));
        let untranslated = Command::new("msgattrib")
            .args(["--untranslated", "--no-obsolete"])
            .arg(&chinese)
            .output()
            .expect("failed to start msgattrib");
        assert!(untranslated.status.success());
        assert!(
            untranslated.stdout.is_empty(),
            "the Simplified Chinese catalog contains untranslated messages"
        );
        let fuzzy = Command::new("msgattrib")
            .args(["--only-fuzzy", "--no-obsolete"])
            .arg(&chinese)
            .output()
            .expect("failed to start msgattrib");
        assert!(fuzzy.status.success());
        assert!(
            fuzzy.stdout.is_empty(),
            "the Simplified Chinese catalog contains fuzzy messages"
        );
        std::fs::remove_file(generated).expect("failed to remove temporary POT file");
        std::fs::remove_file(compiled).expect("failed to remove temporary MO file");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn catalog_follows_the_process_locale() {
        const CHILD: &str = "PPOCR_I18N_LOCALE_TEST_CHILD";
        if let Ok(expect_translation) = std::env::var(CHILD) {
            super::init().expect("failed to initialize gettext in the test child");
            let translated = super::gettext("Settings");
            if expect_translation == "yes" {
                assert_ne!(translated, "Settings");
            } else {
                assert_eq!(translated, "Settings");
            }
            return;
        }

        let executable = std::env::current_exe().expect("test executable path is unavailable");
        for (locale, language, expect_translation) in
            [("en_US.UTF-8", "en", "no"), ("zh_CN.UTF-8", "zh_CN", "yes")]
        {
            let status = Command::new(&executable)
                .args(["--exact", "tests::catalog_follows_the_process_locale"])
                .env("LC_ALL", locale)
                .env("LANGUAGE", language)
                .env(CHILD, expect_translation)
                .status()
                .expect("failed to start locale test child");
            assert!(status.success(), "locale test child failed for {locale}");
        }
    }
}
