use std::{
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

const MAX_SCREENSHOT_BYTES: u64 = 256 * 1024 * 1024;

pub async fn capture_png(output_connector: Option<String>) -> Result<Vec<u8>, String> {
    let (sender, receiver) = async_channel::bounded(1);
    std::thread::Builder::new()
        .name("ppocr-screencopy-client".to_owned())
        .spawn(move || {
            let result = capture_png_blocking(output_connector.as_deref());
            let _ = sender.send_blocking(result);
        })
        .map_err(|error| format!("无法启动物理屏幕捕获线程：{error}"))?;
    receiver
        .recv()
        .await
        .map_err(|error| format!("物理屏幕捕获线程意外结束：{error}"))?
}

fn capture_png_blocking(output_connector: Option<&str>) -> Result<Vec<u8>, String> {
    let helper = helper_path()?;
    let mut command = Command::new(&helper);
    if let Some(connector) = output_connector {
        command.arg("--output").arg(connector);
    }
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("无法启动 {}：{error}", helper.display()))?;
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| "物理屏幕捕获辅助进程没有标准输出".to_owned())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "物理屏幕捕获辅助进程没有错误输出".to_owned())?;
    let stderr_reader = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        let _ = stderr.take(64 * 1024).read_to_end(&mut bytes);
        bytes
    });
    let mut bytes = Vec::new();
    stdout
        .by_ref()
        .take(MAX_SCREENSHOT_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("读取物理屏幕截图失败：{error}"))?;
    let status = child
        .wait()
        .map_err(|error| format!("等待物理屏幕捕获辅助进程失败：{error}"))?;
    let stderr = stderr_reader.join().unwrap_or_default();
    if !status.success() {
        let detail = String::from_utf8_lossy(&stderr).trim().to_owned();
        return Err(if detail.is_empty() {
            format!("物理屏幕捕获辅助进程退出：{status}")
        } else {
            detail
        });
    }
    if bytes.len() as u64 > MAX_SCREENSHOT_BYTES {
        return Err("物理屏幕截图超过 256 MiB 安全限制".to_owned());
    }
    if bytes.is_empty() {
        return Err("物理屏幕捕获辅助进程返回了空图像".to_owned());
    }
    Ok(bytes)
}

fn helper_path() -> Result<PathBuf, String> {
    let executable = std::env::current_exe().map_err(|error| error.to_string())?;
    Ok(helper_path_for(&executable))
}

fn helper_path_for(executable: &Path) -> PathBuf {
    executable
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!("ppocr-screencopy{}", std::env::consts::EXE_SUFFIX))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helper_is_resolved_next_to_calling_executable() {
        let executable = Path::new("/opt/application/bin/consumer");
        assert_eq!(
            helper_path_for(executable).file_name().unwrap(),
            format!("ppocr-screencopy{}", std::env::consts::EXE_SUFFIX).as_str()
        );
    }
}
