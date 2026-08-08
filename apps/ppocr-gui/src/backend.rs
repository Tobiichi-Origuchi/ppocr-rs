use std::{
    cell::{Cell, RefCell},
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::{ChildStdin, ChildStdout, Command, Stdio},
    rc::Rc,
    sync::mpsc::{self, SyncSender},
    thread,
};

use ppocr_app_protocol::{
    Configure, EncodedImageFormat, Frame, Initialize, Message, PipelineConfig,
    ProcessedImagePolicy, Progress, RecognitionResult, Recognize, RequestFailed, Shutdown,
    WorkerHello, read_frame, write_frame,
};
use ppocr_i18n::tr;

pub struct ImageJob {
    pub source_name: String,
    pub bytes: Vec<u8>,
    pub format: EncodedImageFormat,
    pub config: PipelineConfig,
    pub keep_worker_warm: bool,
    pub idle_timeout_minutes: u32,
}

pub enum BackendEvent {
    Status(String),
    Progress(Progress),
    Capabilities(WorkerHello),
    Result {
        source_name: String,
        result: Box<RecognitionResult>,
        processed_png: Vec<u8>,
    },
    Error(String),
}

pub struct Backend {
    callback: Box<dyn Fn(BackendEvent)>,
    session: RefCell<Option<Session>>,
    pending: RefCell<Option<ImageJob>>,
    next_id: Cell<u64>,
    next_session: Cell<u64>,
}

struct Session {
    token: u64,
    client: WorkerClient,
    stage: SessionStage,
    config: PipelineConfig,
    configuration_id: u64,
    job: Option<ActiveJob>,
    keep_worker_warm: bool,
    idle_timeout_minutes: u32,
    accept_result: bool,
}

struct ActiveJob {
    request_id: u64,
    job_id: u64,
    source_name: String,
    bytes: Vec<u8>,
    format: EncodedImageFormat,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SessionStage {
    Starting,
    Initializing,
    Configuring,
    Recognizing,
    Ready,
    Stopping,
}

impl Backend {
    pub fn new(callback: impl Fn(BackendEvent) + 'static) -> Rc<Self> {
        Rc::new(Self {
            callback: Box::new(callback),
            session: RefCell::new(None),
            pending: RefCell::new(None),
            next_id: Cell::new(1),
            next_session: Cell::new(1),
        })
    }

    pub fn submit(self: &Rc<Self>, job: ImageJob) {
        let mut session_ref = self.session.borrow_mut();
        if let Some(session) = session_ref.as_mut()
            && session.stage == SessionStage::Ready
            && session.config == job.config
            && session.job.is_none()
        {
            session.keep_worker_warm = job.keep_worker_warm;
            session.idle_timeout_minutes = job.idle_timeout_minutes;
            session.accept_result = true;
            session.job = Some(self.active_job(job));
            let frame = recognize_frame(session);
            session.stage = SessionStage::Recognizing;
            drop(session_ref);
            self.send(frame);
            return;
        }

        self.pending.replace(Some(job));
        if let Some(session) = session_ref.as_mut() {
            session.accept_result = false;
            session.stage = SessionStage::Stopping;
            let request_id = self.id();
            let _ = session
                .client
                .send(Frame::new(Message::Shutdown(Shutdown { request_id })));
            (self.callback)(BackendEvent::Status(tr!(
                "Restarting the recognition process to apply the configuration"
            )));
            return;
        }
        drop(session_ref);
        self.spawn_pending();
    }

    pub fn shutdown(&self) {
        self.pending.take();
        if let Some(session) = self.session.borrow_mut().as_mut() {
            session.accept_result = false;
            session.stage = SessionStage::Stopping;
            let _ = session.client.send(Frame::new(Message::Shutdown(Shutdown {
                request_id: self.id(),
            })));
        }
    }

    pub fn invalidate(&self) {
        self.pending.take();
        if let Some(session) = self.session.borrow_mut().as_mut() {
            session.accept_result = false;
        }
    }

    fn spawn_pending(self: &Rc<Self>) {
        let Some(job) = self.pending.take() else {
            return;
        };
        let keep_worker_warm = job.keep_worker_warm;
        let idle_timeout_minutes = job.idle_timeout_minutes;
        let token = self.next_session.get();
        self.next_session.set(token.wrapping_add(1).max(1));
        let (client, events) = match WorkerClient::spawn() {
            Ok(value) => value,
            Err(error) => {
                (self.callback)(BackendEvent::Error(error.to_string()));
                return;
            }
        };
        let configuration_id = self.id();
        let session = Session {
            token,
            client,
            stage: SessionStage::Starting,
            config: job.config.clone(),
            configuration_id,
            job: Some(self.active_job(job)),
            keep_worker_warm,
            idle_timeout_minutes,
            accept_result: true,
        };
        self.session.replace(Some(session));
        (self.callback)(BackendEvent::Status(tr!(
            "Starting the recognition process"
        )));
        let weak = Rc::downgrade(self);
        gtk::glib::MainContext::default().spawn_local(async move {
            while let Ok(event) = events.recv().await {
                let Some(backend) = weak.upgrade() else {
                    break;
                };
                backend.handle_client_event(token, event);
            }
        });
    }

    fn handle_client_event(self: &Rc<Self>, token: u64, event: ClientEvent) {
        match event {
            ClientEvent::Frame(frame) => self.handle_frame(token, *frame),
            ClientEvent::Exited(status) => {
                let is_current = self
                    .session
                    .borrow()
                    .as_ref()
                    .is_some_and(|session| session.token == token);
                if !is_current {
                    return;
                }
                let was_stopping = self
                    .session
                    .borrow()
                    .as_ref()
                    .is_some_and(|session| session.stage == SessionStage::Stopping);
                self.session.take();
                if self.pending.borrow().is_some() {
                    self.spawn_pending();
                } else if !was_stopping && !status.success {
                    (self.callback)(BackendEvent::Error(tr!(
                        "The recognition process exited unexpectedly: {description}",
                        description = status.description
                    )));
                }
            }
            ClientEvent::IoFailed(message) => {
                if self
                    .session
                    .borrow()
                    .as_ref()
                    .is_some_and(|session| session.token == token && session.accept_result)
                {
                    (self.callback)(BackendEvent::Error(message));
                }
            }
        }
    }

    fn handle_frame(self: &Rc<Self>, token: u64, frame: Frame) {
        let mut session_ref = self.session.borrow_mut();
        let Some(session) = session_ref.as_mut() else {
            return;
        };
        if session.token != token {
            return;
        }
        match frame.message {
            Message::WorkerHello(hello) => {
                (self.callback)(BackendEvent::Capabilities(hello));
                let request_id = self.id();
                let idle_timeout_ms = session
                    .keep_worker_warm
                    .then(|| u64::from(session.idle_timeout_minutes) * 60_000);
                session.stage = SessionStage::Initializing;
                let frame = Frame::new(Message::Initialize(Initialize {
                    request_id,
                    protocol_minor: ppocr_app_protocol::PROTOCOL_MINOR,
                    idle_timeout_ms,
                }));
                drop(session_ref);
                self.send(frame);
            }
            Message::Ready(_) => {
                session.stage = SessionStage::Configuring;
                let frame = Frame::new(Message::Configure(Configure {
                    request_id: self.id(),
                    configuration_id: session.configuration_id,
                    config: session.config.clone(),
                }));
                (self.callback)(BackendEvent::Status(tr!(
                    "Loading models and initializing the pipeline"
                )));
                drop(session_ref);
                self.send(frame);
            }
            Message::Configured(_) => {
                let frame = recognize_frame(session);
                session.stage = SessionStage::Recognizing;
                (self.callback)(BackendEvent::Status(tr!("Recognizing the image")));
                drop(session_ref);
                self.send(frame);
            }
            Message::Progress(progress) => {
                (self.callback)(BackendEvent::Progress(progress));
            }
            Message::RecognitionResult(result) => {
                let source_name = session
                    .job
                    .take()
                    .map(|job| job.source_name)
                    .unwrap_or_else(|| tr!("Image"));
                let accept_result = session.accept_result;
                let keep_worker_warm = session.keep_worker_warm;
                session.stage = SessionStage::Ready;
                if accept_result {
                    (self.callback)(BackendEvent::Result {
                        source_name,
                        result: Box::new(result),
                        processed_png: frame.attachment,
                    });
                }
                if !keep_worker_warm {
                    session.stage = SessionStage::Stopping;
                    let shutdown = Frame::new(Message::Shutdown(Shutdown {
                        request_id: self.id(),
                    }));
                    drop(session_ref);
                    self.send(shutdown);
                }
            }
            Message::RequestFailed(error) => {
                let message = format_request_error(&error);
                if session.accept_result {
                    (self.callback)(BackendEvent::Error(message));
                }
                if matches!(
                    error.code,
                    ppocr_app_protocol::ErrorCode::ConfigurationRequiresRestart
                        | ppocr_app_protocol::ErrorCode::PipelineConfiguration
                        | ppocr_app_protocol::ErrorCode::ModelDownload
                        | ppocr_app_protocol::ErrorCode::UnsupportedProvider
                ) {
                    session.stage = SessionStage::Stopping;
                    let shutdown = Frame::new(Message::Shutdown(Shutdown {
                        request_id: self.id(),
                    }));
                    drop(session_ref);
                    self.send(shutdown);
                } else {
                    session.stage = SessionStage::Ready;
                    session.job.take();
                }
            }
            Message::ShutdownComplete(_) => {}
            _ => {}
        }
    }

    fn active_job(&self, job: ImageJob) -> ActiveJob {
        ActiveJob {
            request_id: self.id(),
            job_id: self.id(),
            source_name: job.source_name,
            bytes: job.bytes,
            format: job.format,
        }
    }

    fn id(&self) -> u64 {
        let id = self.next_id.get();
        self.next_id.set(id.wrapping_add(1).max(1));
        id
    }

    fn send(&self, frame: Frame) {
        if let Some(session) = self.session.borrow().as_ref()
            && let Err(error) = session.client.send(frame)
        {
            (self.callback)(BackendEvent::Error(error.to_string()));
        }
    }
}

fn recognize_frame(session: &mut Session) -> Frame {
    let job = session
        .job
        .as_mut()
        .expect("recognition requires an active job");
    Frame::with_attachment(
        Message::Recognize(Recognize {
            request_id: job.request_id,
            job_id: job.job_id,
            configuration_id: session.configuration_id,
            source_name: Some(job.source_name.clone()),
            image: ppocr_app_protocol::ImageDescriptor::Encoded { format: job.format },
            processed_image: ProcessedImagePolicy::Always,
        }),
        std::mem::take(&mut job.bytes),
    )
}

fn format_request_error(error: &RequestFailed) -> String {
    tr!(
        "Recognition backend error ({code}): {message}",
        code = format!("{:?}", error.code),
        message = error.message
    )
}

enum ClientEvent {
    Frame(Box<Frame>),
    IoFailed(String),
    Exited(ProcessStatus),
}

struct ProcessStatus {
    success: bool,
    description: String,
}

struct WorkerClient {
    commands: SyncSender<Frame>,
}

impl WorkerClient {
    fn spawn() -> Result<(Self, async_channel::Receiver<ClientEvent>), ClientError> {
        let path = worker_path()?;
        let mut child = Command::new(&path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|source| ClientError::Spawn {
                path: path.clone(),
                source,
            })?;
        let stdin = child
            .stdin
            .take()
            .ok_or(ClientError::MissingPipe("stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or(ClientError::MissingPipe("stdout"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or(ClientError::MissingPipe("stderr"))?;
        let (commands, command_rx) = mpsc::sync_channel(2);
        let (events_tx, events_rx) = async_channel::bounded(32);

        spawn_writer(stdin, command_rx, events_tx.clone())?;
        spawn_reader(stdout, events_tx.clone())?;
        thread::Builder::new()
            .name("ppocr-worker-stderr".to_owned())
            .stack_size(256 * 1024)
            .spawn(move || {
                for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                    eprintln!("ppocr-worker: {line}");
                }
            })?;
        thread::Builder::new()
            .name("ppocr-worker-supervisor".to_owned())
            .stack_size(256 * 1024)
            .spawn(move || {
                let status = child.wait();
                let event = match status {
                    Ok(status) => ClientEvent::Exited(ProcessStatus {
                        success: status.success(),
                        description: status.to_string(),
                    }),
                    Err(error) => ClientEvent::Exited(ProcessStatus {
                        success: false,
                        description: error.to_string(),
                    }),
                };
                let _ = events_tx.send_blocking(event);
            })?;
        Ok((Self { commands }, events_rx))
    }

    fn send(&self, frame: Frame) -> Result<(), ClientError> {
        self.commands
            .send(frame)
            .map_err(|_| ClientError::CommandChannelClosed)
    }
}

fn spawn_writer(
    mut stdin: ChildStdin,
    commands: mpsc::Receiver<Frame>,
    events: async_channel::Sender<ClientEvent>,
) -> Result<(), ClientError> {
    thread::Builder::new()
        .name("ppocr-worker-writer".to_owned())
        .stack_size(512 * 1024)
        .spawn(move || {
            for frame in commands {
                if let Err(error) = write_frame(&mut stdin, &frame) {
                    let _ = events.send_blocking(ClientEvent::IoFailed(error.to_string()));
                    break;
                }
            }
        })?;
    Ok(())
}

fn spawn_reader(
    mut stdout: ChildStdout,
    events: async_channel::Sender<ClientEvent>,
) -> Result<(), ClientError> {
    thread::Builder::new()
        .name("ppocr-worker-reader".to_owned())
        .stack_size(512 * 1024)
        .spawn(move || {
            loop {
                match read_frame(&mut stdout) {
                    Ok(Some(frame)) => {
                        if events
                            .send_blocking(ClientEvent::Frame(Box::new(frame)))
                            .is_err()
                        {
                            break;
                        }
                    }
                    Ok(None) => break,
                    Err(error) => {
                        let _ = events.send_blocking(ClientEvent::IoFailed(error.to_string()));
                        break;
                    }
                }
            }
        })?;
    Ok(())
}

fn worker_path() -> Result<PathBuf, ClientError> {
    let executable = std::env::current_exe().map_err(ClientError::CurrentExecutable)?;
    Ok(worker_path_for(&executable))
}

fn worker_path_for(executable: &Path) -> PathBuf {
    let filename = format!("ppocr-worker{}", std::env::consts::EXE_SUFFIX);
    executable
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(filename)
}

#[derive(Debug)]
enum ClientError {
    CurrentExecutable(std::io::Error),
    Spawn {
        path: PathBuf,
        source: std::io::Error,
    },
    MissingPipe(&'static str),
    Thread(std::io::Error),
    CommandChannelClosed,
}

impl std::fmt::Display for ClientError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::CurrentExecutable(error) => tr!(
                "Failed to resolve the current executable: {error}",
                error = error
            ),
            Self::Spawn { path, source } => tr!(
                "Failed to start OCR worker {path}: {error}",
                path = path.display(),
                error = source
            ),
            Self::MissingPipe(pipe) => {
                tr!("The OCR worker did not expose its {pipe} pipe", pipe = pipe)
            }
            Self::Thread(error) => tr!(
                "Failed to create an OCR worker I/O thread: {error}",
                error = error
            ),
            Self::CommandChannelClosed => tr!("The OCR worker command channel closed"),
        };
        formatter.write_str(&message)
    }
}

impl std::error::Error for ClientError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::CurrentExecutable(error) | Self::Thread(error) => Some(error),
            Self::Spawn { source, .. } => Some(source),
            Self::MissingPipe(_) | Self::CommandChannelClosed => None,
        }
    }
}

impl From<std::io::Error> for ClientError {
    fn from(error: std::io::Error) -> Self {
        Self::Thread(error)
    }
}

pub fn detect_image_format(bytes: &[u8]) -> Option<EncodedImageFormat> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some(EncodedImageFormat::Png)
    } else if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        Some(EncodedImageFormat::Jpeg)
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some(EncodedImageFormat::Gif)
    } else if bytes.starts_with(b"BM") {
        Some(EncodedImageFormat::Bmp)
    } else if bytes.starts_with(b"II*\0") || bytes.starts_with(b"MM\0*") {
        Some(EncodedImageFormat::Tiff)
    } else if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        Some(EncodedImageFormat::WebP)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_is_resolved_next_to_the_gui_without_path_lookup() {
        let executable = if cfg!(windows) {
            Path::new("C:/Program Files/PaddleOCR/ppocr-gui.exe")
        } else {
            Path::new("/opt/ppocr/bin/ppocr-gui")
        };
        assert_eq!(
            worker_path_for(executable).file_name().unwrap(),
            format!("ppocr-worker{}", std::env::consts::EXE_SUFFIX).as_str()
        );
    }

    #[test]
    fn content_magic_wins_over_a_misleading_filename() {
        assert_eq!(
            detect_image_format(&[0xff, 0xd8, 0xff, 0xe0]),
            Some(EncodedImageFormat::Jpeg)
        );
        assert_eq!(
            detect_image_format(b"\x89PNG\r\n\x1a\nrest"),
            Some(EncodedImageFormat::Png)
        );
        assert_eq!(detect_image_format(b"not an image"), None);
    }
}
