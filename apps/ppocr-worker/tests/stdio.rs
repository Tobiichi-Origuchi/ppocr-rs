use std::{
    fs,
    io::Read,
    path::Path,
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
};

use ppocr_app_protocol::{
    Configure, EncodedImageFormat, Frame, ImageDescriptor, Initialize, Message, PipelineConfig,
    ProcessedImagePolicy, Recognize, Shutdown, read_frame, write_frame,
};

struct WorkerProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
}

impl WorkerProcess {
    fn spawn() -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_ppocr-worker"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        Self {
            child,
            stdin,
            stdout,
        }
    }

    fn send(&mut self, frame: Frame) {
        write_frame(&mut self.stdin, &frame).unwrap();
    }

    fn receive(&mut self) -> Frame {
        read_frame(&mut self.stdout).unwrap().unwrap()
    }

    fn shutdown(mut self, request_id: u64) {
        self.send(Frame::new(Message::Shutdown(Shutdown { request_id })));
        assert!(matches!(
            self.receive().message,
            Message::ShutdownComplete(_)
        ));
        drop(self.stdin);
        assert!(self.child.wait().unwrap().success());
    }
}

fn initialize(worker: &mut WorkerProcess) {
    assert!(matches!(worker.receive().message, Message::WorkerHello(_)));
    worker.send(Frame::new(Message::Initialize(Initialize {
        request_id: 1,
        protocol_minor: ppocr_app_protocol::PROTOCOL_MINOR,
        idle_timeout_ms: None,
    })));
    assert!(matches!(worker.receive().message, Message::Ready(_)));
}

fn configure(worker: &mut WorkerProcess, config: PipelineConfig) {
    worker.send(Frame::new(Message::Configure(Configure {
        request_id: 2,
        configuration_id: 1,
        config,
    })));
    assert!(matches!(worker.receive().message, Message::Progress(_)));
    let response = worker.receive();
    if let Message::RequestFailed(error) = &response.message {
        panic!("worker configuration failed: {error:?}");
    }
    assert!(matches!(response.message, Message::Configured(_)));
}

fn encoded_format(bytes: &[u8]) -> EncodedImageFormat {
    match image::guess_format(bytes).unwrap() {
        image::ImageFormat::Png => EncodedImageFormat::Png,
        image::ImageFormat::Jpeg => EncodedImageFormat::Jpeg,
        image::ImageFormat::WebP => EncodedImageFormat::WebP,
        image::ImageFormat::Tiff => EncodedImageFormat::Tiff,
        image::ImageFormat::Bmp => EncodedImageFormat::Bmp,
        image::ImageFormat::Gif => EncodedImageFormat::Gif,
        format => panic!("unsupported test image format: {format:?}"),
    }
}

#[test]
fn child_stdio_handshake_and_shutdown() {
    let mut worker = WorkerProcess::spawn();
    initialize(&mut worker);
    worker.send(Frame::new(Message::Ping(ppocr_app_protocol::Ping {
        request_id: 3,
    })));
    assert!(matches!(worker.receive().message, Message::Pong(_)));
    worker.shutdown(4);
}

#[test]
#[ignore = "downloads models and requires PPOCR_SMOKE_DOCUMENT"]
fn full_pipeline_runs_over_stdio() {
    let path = std::env::var_os("PPOCR_SMOKE_DOCUMENT")
        .expect("PPOCR_SMOKE_DOCUMENT must point to the official OCR image");
    let bytes = fs::read(Path::new(&path)).unwrap();
    let mut worker = WorkerProcess::spawn();
    initialize(&mut worker);
    configure(&mut worker, PipelineConfig::default());
    let format = encoded_format(&bytes);
    worker.send(Frame::with_attachment(
        Message::Recognize(Recognize {
            request_id: 3,
            job_id: 1,
            configuration_id: 1,
            source_name: Some("official.png".to_owned()),
            image: ImageDescriptor::Encoded { format },
            processed_image: ProcessedImagePolicy::Always,
        }),
        bytes,
    ));
    assert!(matches!(worker.receive().message, Message::Progress(_)));
    let result = worker.receive();
    let Message::RecognitionResult(metadata) = result.message else {
        panic!("expected recognition result, received {:?}", result.message);
    };
    assert!(metadata.processed_image_attached);
    assert!(!result.attachment.is_empty());
    assert_eq!(
        metadata
            .metadata
            .text_lines
            .iter()
            .map(|line| line.text.as_str())
            .collect::<Vec<_>>(),
        [
            "上海斯格威铂尔大酒店",
            "打浦路15号",
            "绿洲仕格维花园公寓",
            "打浦路252935号",
        ]
    );
    assert!(metadata.metadata.document_orientation.is_some());
    assert!(metadata.metadata.return_word_boxes);
    assert!(
        metadata
            .metadata
            .text_lines
            .iter()
            .all(|line| line.words.is_some())
    );
    worker.shutdown(4);
}

#[test]
fn protocol_frames_are_not_mixed_with_stderr() {
    let mut worker = Command::new(env!("CARGO_BIN_EXE_ppocr-worker"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdout = worker.stdout.take().unwrap();
    assert!(matches!(
        read_frame(&mut stdout).unwrap().unwrap().message,
        Message::WorkerHello(_)
    ));
    drop(worker.stdin.take());
    assert!(worker.wait().unwrap().success());
    let mut stderr = String::new();
    worker
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut stderr)
        .unwrap();
    assert!(stderr.is_empty());
}
