use std::{
    collections::HashSet,
    io,
    sync::{
        Arc, Mutex,
        mpsc::{self, Receiver, SyncSender},
    },
    thread,
    time::Duration,
};

use ppocr_app_protocol::{
    Cancel, CodecError, Configure, Configured, ErrorCode, Frame, Initialize, Limits, Message, Ping,
    PipelineConfig, Pong, Progress, ProgressPhase, ProviderCapability, ProviderKind, Ready,
    RecognitionResult, Recognize, RequestFailed, ShutdownComplete, WorkerHello, read_frame,
    write_frame,
};
use thiserror::Error;

#[derive(Debug, Error)]
#[error("{message}")]
pub struct EngineError {
    pub code: ErrorCode,
    pub message: String,
    pub retryable: bool,
}

impl EngineError {
    pub fn new(code: ErrorCode, message: impl Into<String>, retryable: bool) -> Self {
        Self {
            code,
            message: message.into(),
            retryable,
        }
    }
}

pub struct EngineOutput {
    pub elapsed_ms: u64,
    pub metadata: ppocr_app_protocol::OcrMetadata,
    pub processed_png: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct EngineProgress {
    pub phase: ProgressPhase,
    pub model_index: u32,
    pub model_count: u32,
    pub model_id: String,
    pub bytes_completed: Option<u64>,
    pub total_bytes: Option<u64>,
}

pub type EngineProgressCallback = Arc<dyn Fn(EngineProgress) + Send + Sync>;

pub trait Engine {
    fn capabilities(&self) -> Vec<ProviderCapability>;
    fn configure(
        &mut self,
        config: &PipelineConfig,
        progress: EngineProgressCallback,
    ) -> Result<ProviderKind, EngineError>;
    fn recognize(
        &mut self,
        request: &Recognize,
        attachment: &[u8],
    ) -> Result<EngineOutput, EngineError>;
}

enum Incoming {
    Frame(Box<Frame>),
    Eof,
    Failed(CodecError),
}

pub fn run_stdio(engine: impl Engine + 'static) -> Result<(), ServerError> {
    let (incoming_tx, incoming_rx) = mpsc::sync_channel(2);
    let cancelled_jobs = Arc::new(Mutex::new(HashSet::new()));
    let cancelled_for_reader = cancelled_jobs.clone();
    let _reader = thread::Builder::new()
        .name("ppocr-ipc-reader".to_owned())
        .stack_size(512 * 1024)
        .spawn(move || read_stdin(incoming_tx, cancelled_for_reader))?;

    let stdout = Arc::new(Mutex::new(io::stdout()));
    let hello = WorkerHello {
        worker_version: env!("CARGO_PKG_VERSION").to_owned(),
        target_os: std::env::consts::OS.to_owned(),
        target_arch: std::env::consts::ARCH.to_owned(),
        protocol_minor: ppocr_app_protocol::PROTOCOL_MINOR,
        limits: Limits::DEFAULT.protocol_limits(),
        providers: engine.capabilities(),
    };
    write_stdout_frame(&stdout, &Frame::new(Message::WorkerHello(hello)))?;

    let stdout_for_events = stdout.clone();
    let event_sink: Arc<dyn Fn(Frame) + Send + Sync> = Arc::new(move |frame| {
        if let Err(error) = write_stdout_frame(&stdout_for_events, &frame) {
            eprintln!("ppocr-worker failed to report progress: {error}");
        }
    });
    let mut machine = WorkerMachine::with_cancelled_jobs(engine, cancelled_jobs, event_sink);
    loop {
        let incoming = receive(
            &incoming_rx,
            machine.idle_timeout(),
            machine.is_configured(),
        );
        let incoming = match incoming {
            Some(incoming) => incoming,
            None => break,
        };
        match incoming {
            Incoming::Frame(frame) => {
                let outcome = machine.handle(*frame);
                for response in outcome.responses {
                    write_stdout_frame(&stdout, &response)?;
                }
                if outcome.exit {
                    break;
                }
            }
            Incoming::Eof => break,
            Incoming::Failed(error) => return Err(error.into()),
        }
    }
    drop(incoming_rx);
    Ok(())
}

fn write_stdout_frame(stdout: &Arc<Mutex<io::Stdout>>, frame: &Frame) -> Result<(), CodecError> {
    let mut stdout = stdout
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    write_frame(&mut *stdout, frame)
}

fn read_stdin(sender: SyncSender<Incoming>, cancelled_jobs: Arc<Mutex<HashSet<u64>>>) {
    let mut stdin = io::stdin().lock();
    loop {
        let incoming = match read_frame(&mut stdin) {
            Ok(Some(frame)) => {
                if let Message::Cancel(request) = &frame.message
                    && let Ok(mut cancelled) = cancelled_jobs.lock()
                {
                    cancelled.insert(request.job_id);
                }
                Incoming::Frame(Box::new(frame))
            }
            Ok(None) => Incoming::Eof,
            Err(error) => Incoming::Failed(error),
        };
        let done = matches!(incoming, Incoming::Eof | Incoming::Failed(_));
        if sender.send(incoming).is_err() || done {
            break;
        }
    }
}

fn receive(
    receiver: &Receiver<Incoming>,
    timeout: Option<Duration>,
    configured: bool,
) -> Option<Incoming> {
    if configured && let Some(timeout) = timeout {
        receiver.recv_timeout(timeout).ok()
    } else {
        receiver.recv().ok()
    }
}

struct Outcome {
    responses: Vec<Frame>,
    exit: bool,
}

impl Outcome {
    fn one(response: Frame) -> Self {
        Self {
            responses: vec![response],
            exit: false,
        }
    }

    fn exit(response: Frame) -> Self {
        Self {
            responses: vec![response],
            exit: true,
        }
    }
}

struct WorkerMachine<E> {
    engine: E,
    initialized: bool,
    idle_timeout: Option<Duration>,
    configured: Option<(u64, PipelineConfig, ProviderKind)>,
    cancelled_jobs: Arc<Mutex<HashSet<u64>>>,
    event_sink: Arc<dyn Fn(Frame) + Send + Sync>,
}

impl<E: Engine> WorkerMachine<E> {
    #[cfg(test)]
    fn new(engine: E) -> Self {
        Self::with_cancelled_jobs(
            engine,
            Arc::new(Mutex::new(HashSet::new())),
            Arc::new(|_| {}),
        )
    }

    fn with_cancelled_jobs(
        engine: E,
        cancelled_jobs: Arc<Mutex<HashSet<u64>>>,
        event_sink: Arc<dyn Fn(Frame) + Send + Sync>,
    ) -> Self {
        Self {
            engine,
            initialized: false,
            idle_timeout: None,
            configured: None,
            cancelled_jobs,
            event_sink,
        }
    }

    fn idle_timeout(&self) -> Option<Duration> {
        self.idle_timeout
    }

    fn is_configured(&self) -> bool {
        self.configured.is_some()
    }

    fn handle(&mut self, frame: Frame) -> Outcome {
        match frame.message {
            Message::Initialize(request) => self.initialize(request, frame.attachment),
            Message::Configure(request) => self.configure(request, frame.attachment),
            Message::Recognize(request) => self.recognize(request, frame.attachment),
            Message::Cancel(request) => self.cancel(request, frame.attachment),
            Message::Ping(Ping { request_id }) if frame.attachment.is_empty() => {
                Outcome::one(Frame::new(Message::Pong(Pong { request_id })))
            }
            Message::Unload(request) => {
                Outcome::exit(Frame::new(Message::ShutdownComplete(ShutdownComplete {
                    request_id: request.request_id,
                })))
            }
            Message::Shutdown(request) => {
                Outcome::exit(Frame::new(Message::ShutdownComplete(ShutdownComplete {
                    request_id: request.request_id,
                })))
            }
            message => Outcome::one(failure(
                request_id(&message),
                None,
                ErrorCode::InvalidRequest,
                "message is not valid in the client-to-worker direction",
                false,
            )),
        }
    }

    fn initialize(&mut self, request: Initialize, attachment: Vec<u8>) -> Outcome {
        if !attachment.is_empty() {
            return Outcome::one(failure(
                Some(request.request_id),
                None,
                ErrorCode::InvalidRequest,
                "initialize must not contain an attachment",
                false,
            ));
        }
        if request.protocol_minor > ppocr_app_protocol::PROTOCOL_MINOR {
            return Outcome::one(failure(
                Some(request.request_id),
                None,
                ErrorCode::Protocol,
                "client protocol minor is newer than the worker",
                false,
            ));
        }
        self.initialized = true;
        self.idle_timeout = request.idle_timeout_ms.map(Duration::from_millis);
        Outcome::one(Frame::new(Message::Ready(Ready {
            request_id: request.request_id,
        })))
    }

    fn configure(&mut self, request: Configure, attachment: Vec<u8>) -> Outcome {
        if !self.initialized {
            return Outcome::one(failure(
                Some(request.request_id),
                None,
                ErrorCode::InvalidState,
                "initialize must complete before configure",
                false,
            ));
        }
        if !attachment.is_empty() {
            return Outcome::one(failure(
                Some(request.request_id),
                None,
                ErrorCode::InvalidRequest,
                "configure must not contain an attachment",
                false,
            ));
        }
        if let Some((configuration_id, config, provider)) = &self.configured {
            if *configuration_id == request.configuration_id && *config == request.config {
                return Outcome::one(Frame::new(Message::Configured(Configured {
                    request_id: request.request_id,
                    configuration_id: *configuration_id,
                    provider: *provider,
                })));
            }
            return Outcome::one(failure(
                Some(request.request_id),
                None,
                ErrorCode::ConfigurationRequiresRestart,
                "pipeline configuration is immutable for one worker process",
                true,
            ));
        }

        (self.event_sink)(Frame::new(Message::Progress(Progress {
            request_id: request.request_id,
            job_id: None,
            phase: ProgressPhase::PreparingModels,
            message: "正在准备所选模型".to_owned(),
            model_index: None,
            model_count: None,
            model_id: None,
            bytes_completed: None,
            total_bytes: None,
        })));
        let mut responses = Vec::new();
        let event_sink = self.event_sink.clone();
        let request_id = request.request_id;
        let engine_progress: EngineProgressCallback = Arc::new(move |progress| {
            event_sink(Frame::new(Message::Progress(protocol_progress(
                request_id, progress,
            ))));
        });
        match self.engine.configure(&request.config, engine_progress) {
            Ok(provider) => {
                self.configured = Some((request.configuration_id, request.config, provider));
                responses.push(Frame::new(Message::Configured(Configured {
                    request_id: request.request_id,
                    configuration_id: request.configuration_id,
                    provider,
                })));
            }
            Err(error) => responses.push(engine_failure(Some(request.request_id), None, error)),
        }
        Outcome {
            responses,
            exit: false,
        }
    }

    fn recognize(&mut self, request: Recognize, attachment: Vec<u8>) -> Outcome {
        let Some((configuration_id, _, _)) = self.configured.as_ref() else {
            return Outcome::one(failure(
                Some(request.request_id),
                Some(request.job_id),
                ErrorCode::InvalidState,
                "configure must complete before recognize",
                false,
            ));
        };
        if request.configuration_id != *configuration_id {
            return Outcome::one(failure(
                Some(request.request_id),
                Some(request.job_id),
                ErrorCode::InvalidRequest,
                "recognize references a stale configuration",
                false,
            ));
        }
        if attachment.is_empty() {
            return Outcome::one(failure(
                Some(request.request_id),
                Some(request.job_id),
                ErrorCode::InvalidRequest,
                "recognize requires an image attachment",
                false,
            ));
        }
        if self.take_cancelled(request.job_id) {
            return Outcome::one(failure(
                Some(request.request_id),
                Some(request.job_id),
                ErrorCode::Cancelled,
                "recognition was cancelled",
                true,
            ));
        }

        let mut responses = vec![Frame::new(Message::Progress(Progress {
            request_id: request.request_id,
            job_id: Some(request.job_id),
            phase: ProgressPhase::Decoding,
            message: "正在解码图片".to_owned(),
            model_index: None,
            model_count: None,
            model_id: None,
            bytes_completed: None,
            total_bytes: None,
        }))];
        match self.engine.recognize(&request, &attachment) {
            Ok(output) => {
                if self.take_cancelled(request.job_id) {
                    responses.push(failure(
                        Some(request.request_id),
                        Some(request.job_id),
                        ErrorCode::Cancelled,
                        "recognition result was discarded after cancellation",
                        true,
                    ));
                } else {
                    let attached = !output.processed_png.is_empty();
                    responses.push(Frame::with_attachment(
                        Message::RecognitionResult(RecognitionResult {
                            request_id: request.request_id,
                            job_id: request.job_id,
                            configuration_id: request.configuration_id,
                            elapsed_ms: output.elapsed_ms,
                            metadata: output.metadata,
                            processed_image_attached: attached,
                        }),
                        output.processed_png,
                    ));
                }
            }
            Err(error) => responses.push(engine_failure(
                Some(request.request_id),
                Some(request.job_id),
                error,
            )),
        }
        Outcome {
            responses,
            exit: false,
        }
    }

    fn cancel(&mut self, request: Cancel, attachment: Vec<u8>) -> Outcome {
        if !attachment.is_empty() {
            return Outcome::one(failure(
                Some(request.request_id),
                Some(request.job_id),
                ErrorCode::InvalidRequest,
                "cancel must not contain an attachment",
                false,
            ));
        }
        Outcome::one(failure(
            Some(request.request_id),
            Some(request.job_id),
            ErrorCode::Cancelled,
            "cancellation recorded",
            true,
        ))
    }

    fn take_cancelled(&self, job_id: u64) -> bool {
        self.cancelled_jobs
            .lock()
            .is_ok_and(|mut cancelled| cancelled.remove(&job_id))
    }
}

fn failure(
    request_id: Option<u64>,
    job_id: Option<u64>,
    code: ErrorCode,
    message: impl Into<String>,
    retryable: bool,
) -> Frame {
    Frame::new(Message::RequestFailed(RequestFailed {
        request_id,
        job_id,
        code,
        message: message.into(),
        retryable,
    }))
}

fn engine_failure(request_id: Option<u64>, job_id: Option<u64>, error: EngineError) -> Frame {
    failure(
        request_id,
        job_id,
        error.code,
        error.message,
        error.retryable,
    )
}

fn protocol_progress(request_id: u64, progress: EngineProgress) -> Progress {
    let message = match (
        progress.phase,
        progress.bytes_completed,
        progress.total_bytes,
    ) {
        (ProgressPhase::PreparingModels, Some(done), Some(total)) => format!(
            "正在下载模型（{}/{}）{}/{} {}",
            progress.model_index,
            progress.model_count,
            human_bytes(done),
            human_bytes(total),
            progress.model_id
        ),
        (ProgressPhase::InitializingSessions, _, _) => format!(
            "正在加载模型（{}/{}）{}",
            progress.model_index, progress.model_count, progress.model_id
        ),
        _ => progress.model_id.clone(),
    };
    Progress {
        request_id,
        job_id: None,
        phase: progress.phase,
        message,
        model_index: Some(progress.model_index),
        model_count: Some(progress.model_count),
        model_id: Some(progress.model_id),
        bytes_completed: progress.bytes_completed,
        total_bytes: progress.total_bytes,
    }
}

fn human_bytes(bytes: u64) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    if bytes < 1024 * 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MiB", bytes as f64 / MIB)
    }
}

fn request_id(message: &Message) -> Option<u64> {
    match message {
        Message::Ready(value) => Some(value.request_id),
        Message::Configured(value) => Some(value.request_id),
        Message::Progress(value) => Some(value.request_id),
        Message::RecognitionResult(value) => Some(value.request_id),
        Message::RequestFailed(value) => value.request_id,
        Message::Pong(value) => Some(value.request_id),
        Message::ShutdownComplete(value) => Some(value.request_id),
        _ => None,
    }
}

#[derive(Debug, Error)]
pub enum ServerError {
    #[error(transparent)]
    Codec(#[from] CodecError),
    #[error("failed to create IPC thread: {0}")]
    Thread(#[from] io::Error),
}

#[cfg(test)]
mod tests {
    use ppocr_app_protocol::{
        DetectionSizeLimit, EffectiveDetectionParameters, ImageDescriptor, ImageSize, ModelIds,
        ModelSettings, OcrMetadata, ProcessedImagePolicy, ProviderConfig,
    };

    use super::*;

    #[derive(Default)]
    struct FakeEngine {
        configure_count: usize,
        recognize_count: usize,
    }

    impl Engine for FakeEngine {
        fn capabilities(&self) -> Vec<ProviderCapability> {
            Vec::new()
        }

        fn configure(
            &mut self,
            _: &PipelineConfig,
            _: EngineProgressCallback,
        ) -> Result<ProviderKind, EngineError> {
            self.configure_count += 1;
            Ok(ProviderKind::Cpu)
        }

        fn recognize(&mut self, _: &Recognize, _: &[u8]) -> Result<EngineOutput, EngineError> {
            self.recognize_count += 1;
            Ok(EngineOutput {
                elapsed_ms: 7,
                metadata: empty_metadata(),
                processed_png: vec![1, 2, 3],
            })
        }
    }

    fn empty_metadata() -> OcrMetadata {
        OcrMetadata {
            input_size: ImageSize {
                width: 1,
                height: 1,
            },
            processed_size: ImageSize {
                width: 1,
                height: 1,
            },
            model_settings: ModelSettings {
                use_document_orientation: false,
                use_document_rectification: false,
                use_textline_orientation: false,
            },
            model_ids: ModelIds {
                document_orientation: None,
                document_rectification: None,
                text_detection: "det".to_owned(),
                textline_orientation: None,
                text_recognition: "rec".to_owned(),
            },
            text_detection_parameters: EffectiveDetectionParameters {
                limit_side_len: 64,
                limit_type: DetectionSizeLimit::Min,
                max_side_limit: 4000,
                threshold: 0.3,
                box_threshold: 0.6,
                unclip_ratio: 1.5,
            },
            text_recognition_score_threshold: 0.0,
            return_word_boxes: false,
            document_orientation: None,
            detections: Vec::new(),
            textline_orientations: Vec::new(),
            text_lines: Vec::new(),
        }
    }

    fn initialize(machine: &mut WorkerMachine<FakeEngine>) {
        let result = machine.handle(Frame::new(Message::Initialize(Initialize {
            request_id: 1,
            protocol_minor: ppocr_app_protocol::PROTOCOL_MINOR,
            idle_timeout_ms: None,
        })));
        assert!(matches!(result.responses[0].message, Message::Ready(_)));
    }

    fn configure(machine: &mut WorkerMachine<FakeEngine>, id: u64) -> Outcome {
        machine.handle(Frame::new(Message::Configure(Configure {
            request_id: id + 10,
            configuration_id: id,
            config: PipelineConfig::default(),
        })))
    }

    #[test]
    fn requires_initialize_and_keeps_configuration_immutable() {
        let mut machine = WorkerMachine::new(FakeEngine::default());
        let rejected = configure(&mut machine, 1);
        assert!(matches!(
            rejected.responses[0].message,
            Message::RequestFailed(RequestFailed {
                code: ErrorCode::InvalidState,
                ..
            })
        ));

        initialize(&mut machine);
        let configured = configure(&mut machine, 1);
        assert!(
            configured
                .responses
                .iter()
                .any(|frame| matches!(frame.message, Message::Configured(_)))
        );
        assert_eq!(machine.engine.configure_count, 1);

        let repeated = configure(&mut machine, 1);
        assert!(matches!(
            repeated.responses[0].message,
            Message::Configured(_)
        ));
        assert_eq!(machine.engine.configure_count, 1);

        let different = configure(&mut machine, 2);
        assert!(matches!(
            different.responses[0].message,
            Message::RequestFailed(RequestFailed {
                code: ErrorCode::ConfigurationRequiresRestart,
                ..
            })
        ));
    }

    #[test]
    fn recognizes_only_the_active_configuration() {
        let mut machine = WorkerMachine::new(FakeEngine::default());
        initialize(&mut machine);
        configure(&mut machine, 5);
        let request = Recognize {
            request_id: 20,
            job_id: 30,
            configuration_id: 5,
            source_name: None,
            image: ImageDescriptor::Encoded {
                format: ppocr_app_protocol::EncodedImageFormat::Png,
            },
            processed_image: ProcessedImagePolicy::Always,
        };
        let outcome = machine.handle(Frame::with_attachment(Message::Recognize(request), vec![9]));
        let result = outcome.responses.last().unwrap();
        assert!(matches!(result.message, Message::RecognitionResult(_)));
        assert_eq!(result.attachment, vec![1, 2, 3]);
        assert_eq!(machine.engine.recognize_count, 1);
    }

    #[test]
    fn ping_and_shutdown_do_not_require_a_pipeline() {
        let mut machine = WorkerMachine::new(FakeEngine::default());
        let pong = machine.handle(Frame::new(Message::Ping(Ping { request_id: 9 })));
        assert!(matches!(pong.responses[0].message, Message::Pong(_)));
        let stopped = machine.handle(Frame::new(Message::Shutdown(
            ppocr_app_protocol::Shutdown { request_id: 10 },
        )));
        assert!(stopped.exit);
    }

    #[test]
    fn config_dto_supports_explicit_provider() {
        let config = PipelineConfig {
            provider: ProviderConfig::Cpu,
            ..PipelineConfig::default()
        };
        assert_eq!(config.provider, ProviderConfig::Cpu);
    }

    #[test]
    fn model_progress_contains_position_bytes_and_full_model_id() {
        let progress = protocol_progress(
            7,
            EngineProgress {
                phase: ProgressPhase::PreparingModels,
                model_index: 1,
                model_count: 5,
                model_id: "PP-OCRv6_mobile_rec".to_owned(),
                bytes_completed: Some(11_943_936),
                total_bytes: Some(53_897_626),
            },
        );
        assert_eq!(progress.model_index, Some(1));
        assert_eq!(progress.model_count, Some(5));
        assert_eq!(progress.model_id.as_deref(), Some("PP-OCRv6_mobile_rec"));
        assert!(progress.message.contains("（1/5）"));
        assert!(progress.message.contains("PP-OCRv6_mobile_rec"));
        assert!(progress.message.contains("MiB"));
    }
}
