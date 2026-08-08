#[cfg(target_os = "linux")]
mod layer_shell;
#[cfg(target_os = "linux")]
mod pipewire_frame;
#[cfg(target_os = "linux")]
mod screencopy_frame;
mod selector;

use gtk::prelude::*;
use ppocr_i18n::tr;

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

#[derive(Debug)]
pub enum CaptureError {
    #[cfg(not(target_os = "linux"))]
    UnsupportedPlatform,
    Screencopy(String),
    Portal(String),
    Selection(String),
    PipeWire(String),
}

impl std::fmt::Display for CaptureError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            #[cfg(not(target_os = "linux"))]
            Self::UnsupportedPlatform => {
                tr!("Screenshot capture is not supported on this platform yet")
            }
            Self::Screencopy(error) => tr!(
                "Failed to capture the physical output: {error}",
                error = error
            ),
            Self::Portal(error) => tr!("Screenshot portal failed: {error}", error = error),
            Self::Selection(error) => tr!(
                "Failed to prepare screenshot selection: {error}",
                error = error
            ),
            Self::PipeWire(error) => {
                tr!("Failed to capture window frame: {error}", error = error)
            }
        };
        formatter.write_str(&message)
    }
}

impl std::error::Error for CaptureError {}

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
                    source_name: tr!("Region capture.png"),
                })),
                None => Ok(CaptureOutcome::Cancelled),
            }
        }
        CaptureKind::Screen => Ok(CaptureOutcome::Captured(CapturedImage {
            bytes: capture_output(monitor).await?,
            source_name: tr!("Screen capture.png"),
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
        return Err(CaptureError::Portal(tr!(
            "The current ScreenCast portal does not provide window capture"
        )));
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
        return Err(CaptureError::Portal(tr!(
            "The portal returned no video stream after selection"
        )));
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
                source_name: tr!("Window capture.png"),
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
