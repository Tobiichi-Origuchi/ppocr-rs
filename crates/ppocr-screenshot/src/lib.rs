#[cfg(target_os = "linux")]
mod layer_shell;
#[cfg(target_os = "linux")]
mod pipewire_frame;
#[cfg(target_os = "linux")]
mod screencopy_frame;
mod selector;

use gtk::prelude::*;
use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CaptureKind {
    Region,
    Screen,
    Window,
}

pub struct CapturedImage {
    pub bytes: Vec<u8>,
    pub source_name: String,
}

pub enum CaptureOutcome {
    Captured(CapturedImage),
    Cancelled,
}

#[derive(Debug, Error)]
pub enum CaptureError {
    #[cfg(not(target_os = "linux"))]
    #[error("screenshot capture is not supported on this platform yet")]
    UnsupportedPlatform,
    #[error("failed to capture the physical output: {0}")]
    Screencopy(String),
    #[error("screenshot portal failed: {0}")]
    Portal(String),
    #[error("failed to prepare screenshot selection: {0}")]
    Selection(String),
    #[error("failed to capture window frame: {0}")]
    PipeWire(String),
}

#[cfg(target_os = "linux")]
pub async fn capture(
    window: &gtk::Window,
    kind: CaptureKind,
    settle_delay: std::time::Duration,
) -> Result<CaptureOutcome, CaptureError> {
    let monitor = window
        .surface()
        .and_then(|surface| gtk::prelude::WidgetExt::display(window).monitor_at_surface(&surface));
    let hide_main_window = kind != CaptureKind::Window;
    if hide_main_window {
        window.set_visible(false);
        gtk::glib::timeout_future(settle_delay).await;
    }
    let result = capture_hidden(window, kind, monitor.as_ref()).await;
    if hide_main_window {
        window.present();
    }
    result
}

#[cfg(target_os = "linux")]
async fn capture_hidden(
    window: &gtk::Window,
    kind: CaptureKind,
    monitor: Option<&gtk::gdk::Monitor>,
) -> Result<CaptureOutcome, CaptureError> {
    match kind {
        CaptureKind::Region => {
            let bytes = capture_output(monitor).await?;
            match selector::select_region(window, monitor, bytes)
                .await
                .map_err(CaptureError::Selection)?
            {
                Some(bytes) => Ok(CaptureOutcome::Captured(CapturedImage {
                    bytes,
                    source_name: "区域截图.png".to_owned(),
                })),
                None => Ok(CaptureOutcome::Cancelled),
            }
        }
        CaptureKind::Screen => Ok(CaptureOutcome::Captured(CapturedImage {
            bytes: capture_output(monitor).await?,
            source_name: "屏幕截图.png".to_owned(),
        })),
        CaptureKind::Window => capture_window().await,
    }
}

#[cfg(target_os = "linux")]
async fn capture_output(monitor: Option<&gtk::gdk::Monitor>) -> Result<Vec<u8>, CaptureError> {
    let connector = monitor
        .and_then(gtk::gdk::prelude::MonitorExt::connector)
        .map(|value| value.to_string());
    screencopy_frame::capture_png(connector)
        .await
        .map_err(CaptureError::Screencopy)
}

#[cfg(target_os = "linux")]
async fn capture_window() -> Result<CaptureOutcome, CaptureError> {
    use ashpd::desktop::{
        PersistMode,
        screencast::{CursorMode, Screencast, SelectSourcesOptions, SourceType},
    };

    let portal = Screencast::new()
        .await
        .map_err(|error| CaptureError::Portal(error.to_string()))?;
    let sources = portal
        .available_source_types()
        .await
        .map_err(|error| CaptureError::Portal(error.to_string()))?;
    if !sources.contains(SourceType::Window) {
        return Err(CaptureError::Portal(
            "当前 ScreenCast portal 不提供窗口捕获".to_owned(),
        ));
    }
    let session = portal
        .create_session(Default::default())
        .await
        .map_err(|error| CaptureError::Portal(error.to_string()))?;
    let selection = portal
        .select_sources(
            &session,
            SelectSourcesOptions::default()
                .set_cursor_mode(CursorMode::Hidden)
                .set_sources(ashpd::enumflags2::BitFlags::from_flag(SourceType::Window))
                .set_multiple(false)
                .set_persist_mode(PersistMode::DoNot),
        )
        .await;
    let selection = match selection {
        Ok(request) => request,
        Err(error) if portal_cancelled(&error) => return Ok(CaptureOutcome::Cancelled),
        Err(error) => return Err(CaptureError::Portal(error.to_string())),
    };
    if let Err(error) = selection.response() {
        if portal_cancelled(&error) {
            return Ok(CaptureOutcome::Cancelled);
        }
        return Err(CaptureError::Portal(error.to_string()));
    }
    let started = match portal.start(&session, None, Default::default()).await {
        Ok(request) => request,
        Err(error) if portal_cancelled(&error) => return Ok(CaptureOutcome::Cancelled),
        Err(error) => return Err(CaptureError::Portal(error.to_string())),
    };
    let response = match started.response() {
        Ok(response) => response,
        Err(error) if portal_cancelled(&error) => return Ok(CaptureOutcome::Cancelled),
        Err(error) => return Err(CaptureError::Portal(error.to_string())),
    };
    let Some(stream) = response.streams().first() else {
        return Err(CaptureError::Portal(
            "选择完成但 portal 没有返回视频流".to_owned(),
        ));
    };
    let fd = portal
        .open_pipe_wire_remote(&session, Default::default())
        .await
        .map_err(|error| CaptureError::Portal(error.to_string()))?;
    let result = pipewire_frame::capture_png(stream.pipe_wire_node_id(), fd)
        .await
        .map(|bytes| {
            CaptureOutcome::Captured(CapturedImage {
                bytes,
                source_name: "窗口截图.png".to_owned(),
            })
        })
        .map_err(CaptureError::PipeWire);
    let _ = session.close().await;
    result
}

#[cfg(target_os = "linux")]
fn portal_cancelled(error: &ashpd::Error) -> bool {
    use ashpd::{PortalError, desktop::ResponseError};

    matches!(
        error,
        ashpd::Error::Portal(PortalError::Cancelled(_))
            | ashpd::Error::Response(ResponseError::Cancelled)
    )
}

#[cfg(not(target_os = "linux"))]
pub async fn capture(
    _: &gtk::Window,
    _: CaptureKind,
    _: std::time::Duration,
) -> Result<CaptureOutcome, CaptureError> {
    Err(CaptureError::UnsupportedPlatform)
}
