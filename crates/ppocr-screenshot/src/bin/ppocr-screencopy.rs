#[cfg(target_os = "linux")]
use std::io::{Cursor, Write};

#[cfg(target_os = "linux")]
use image::{
    ExtendedColorType, ImageEncoder,
    codecs::png::{CompressionType, FilterType, PngEncoder},
};
#[cfg(target_os = "linux")]
use libwayshot::WayshotConnection;
#[cfg(target_os = "linux")]
use ppocr_i18n::tr;

#[cfg(target_os = "linux")]
fn main() {
    ppocr_i18n::init().expect("failed to initialize translations");
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

#[cfg(not(target_os = "linux"))]
fn main() {
    ppocr_i18n::init().expect("failed to initialize translations");
    eprintln!(
        "{}",
        ppocr_i18n::tr!("Wayland physical screencopy is only available on Linux")
    );
    std::process::exit(1);
}

#[cfg(target_os = "linux")]
fn run() -> Result<(), String> {
    let requested_output = parse_output()?;
    let connection = WayshotConnection::new().map_err(|error| {
        tr!(
            "Failed to connect to Wayland screencopy: {error}",
            error = error
        )
    })?;
    let outputs = connection.get_all_outputs();
    let output = match requested_output.as_deref() {
        Some(name) => outputs
            .iter()
            .find(|output| output.name == name)
            .ok_or_else(|| tr!("Wayland has no output named {name}", name = name))?,
        None if outputs.len() == 1 => &outputs[0],
        None if outputs.is_empty() => return Err(tr!("Wayland has no capturable output")),
        None => {
            return Err(tr!(
                "Wayland has multiple outputs but no target was specified"
            ));
        }
    };
    let image = connection
        .screenshot_single_output(output, false)
        .map_err(|error| {
            tr!(
                "Wayland physical-output capture failed: {error}",
                error = error
            )
        })?;
    let image = image.to_rgba8();
    let mut encoded = Cursor::new(Vec::new());
    PngEncoder::new_with_quality(&mut encoded, CompressionType::Fast, FilterType::Sub)
        .write_image(
            image.as_raw(),
            image.width(),
            image.height(),
            ExtendedColorType::Rgba8,
        )
        .map_err(|error| {
            tr!(
                "Failed to encode the Wayland screenshot: {error}",
                error = error
            )
        })?;
    std::io::stdout()
        .lock()
        .write_all(encoded.get_ref())
        .map_err(|error| {
            tr!(
                "Failed to write the Wayland screenshot: {error}",
                error = error
            )
        })
}

#[cfg(target_os = "linux")]
fn parse_output() -> Result<Option<String>, String> {
    let mut args = std::env::args().skip(1);
    let Some(flag) = args.next() else {
        return Ok(None);
    };
    if flag != "--output" {
        return Err(tr!("Unknown argument: {flag}", flag = flag));
    }
    let name = args
        .next()
        .ok_or_else(|| tr!("--output requires an output name"))?;
    if args.next().is_some() {
        return Err(tr!("Too many arguments"));
    }
    Ok(Some(name))
}
