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
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("Wayland physical screencopy is only available on Linux");
    std::process::exit(1);
}

#[cfg(target_os = "linux")]
fn run() -> Result<(), String> {
    let requested_output = parse_output()?;
    let connection = WayshotConnection::new()
        .map_err(|error| format!("无法连接 Wayland screencopy：{error}"))?;
    let outputs = connection.get_all_outputs();
    let output = match requested_output.as_deref() {
        Some(name) => outputs
            .iter()
            .find(|output| output.name == name)
            .ok_or_else(|| format!("Wayland 没有名为 {name} 的输出"))?,
        None if outputs.len() == 1 => &outputs[0],
        None if outputs.is_empty() => return Err("Wayland 没有可捕获的输出".to_owned()),
        None => return Err("存在多个 Wayland 输出，但没有指定目标输出".to_owned()),
    };
    let image = connection
        .screenshot_single_output(output, false)
        .map_err(|error| format!("Wayland 物理屏幕捕获失败：{error}"))?;
    let image = image.to_rgba8();
    let mut encoded = Cursor::new(Vec::new());
    PngEncoder::new_with_quality(&mut encoded, CompressionType::Fast, FilterType::Sub)
        .write_image(
            image.as_raw(),
            image.width(),
            image.height(),
            ExtendedColorType::Rgba8,
        )
        .map_err(|error| format!("无法编码 Wayland 屏幕截图：{error}"))?;
    std::io::stdout()
        .lock()
        .write_all(encoded.get_ref())
        .map_err(|error| format!("无法输出 Wayland 屏幕截图：{error}"))
}

#[cfg(target_os = "linux")]
fn parse_output() -> Result<Option<String>, String> {
    let mut args = std::env::args().skip(1);
    let Some(flag) = args.next() else {
        return Ok(None);
    };
    if flag != "--output" {
        return Err(format!("未知参数：{flag}"));
    }
    let name = args
        .next()
        .ok_or_else(|| "--output 缺少输出名称".to_owned())?;
    if args.next().is_some() {
        return Err("参数过多".to_owned());
    }
    Ok(Some(name))
}
