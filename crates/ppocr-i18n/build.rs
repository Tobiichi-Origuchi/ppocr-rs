use std::{env, fs, path::PathBuf, process::Command};

const DOMAIN: &str = "ppocr-rs";

fn main() {
    println!("cargo:rerun-if-env-changed=LOCALEDIR");
    println!("cargo:rerun-if-changed=../../po/LINGUAS");
    println!("cargo:rerun-if-changed=../../po/zh_CN.po");

    if let Some(locale_dir) = env::var_os("LOCALEDIR") {
        println!(
            "cargo:rustc-env=PPOCR_DEFAULT_LOCALEDIR={}",
            PathBuf::from(locale_dir).display()
        );
        return;
    }

    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let po_file = manifest_dir.join("../../po/zh_CN.po");
    let locale_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("locale");
    let messages_dir = locale_dir.join("zh_CN/LC_MESSAGES");
    fs::create_dir_all(&messages_dir).expect("failed to create the gettext output directory");
    let mo_file = messages_dir.join(format!("{DOMAIN}.mo"));
    let status = Command::new("msgfmt")
        .args(["--check", "--check-format", "--output-file"])
        .arg(&mo_file)
        .arg(&po_file)
        .status()
        .expect("failed to execute msgfmt; install GNU gettext tools");
    assert!(status.success(), "msgfmt rejected {}", po_file.display());
    println!(
        "cargo:rustc-env=PPOCR_DEFAULT_LOCALEDIR={}",
        locale_dir.display()
    );
}
