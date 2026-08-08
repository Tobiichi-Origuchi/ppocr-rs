use ppocr_i18n::tr;
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
        .map_err(|error| {
            tr!(
                "Failed to start the physical-output capture thread: {error}",
                error = error
            )
        })?;
    receiver.recv().await.map_err(|error| {
        tr!(
            "The physical-output capture thread ended unexpectedly: {error}",
            error = error
        )
    })?
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
        .map_err(|error| {
            tr!(
                "Failed to start {helper}: {error}",
                helper = helper.display(),
                error = error
            )
        })?;
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| tr!("The physical-output capture helper has no standard output"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| tr!("The physical-output capture helper has no error output"))?;
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
        .map_err(|error| {
            tr!(
                "Failed to read the physical-output capture: {error}",
                error = error
            )
        })?;
    let status = child.wait().map_err(|error| {
        tr!(
            "Failed to wait for the physical-output capture helper: {error}",
            error = error
        )
    })?;
    let stderr = stderr_reader.join().unwrap_or_default();
    if !status.success() {
        let detail = String::from_utf8_lossy(&stderr).trim().to_owned();
        return Err(if detail.is_empty() {
            tr!(
                "The physical-output capture helper exited: {status}",
                status = status
            )
        } else {
            detail
        });
    }
    if bytes.len() as u64 > MAX_SCREENSHOT_BYTES {
        return Err(tr!(
            "The physical-output capture exceeds the 256 MiB safety limit"
        ));
    }
    if bytes.is_empty() {
        return Err(tr!(
            "The physical-output capture helper returned an empty image"
        ));
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
