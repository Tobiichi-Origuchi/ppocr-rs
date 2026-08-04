use serde::{Deserialize, Serialize};

pub const PROTOCOL_MINOR: u16 = 0;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "payload", rename_all = "snake_case")]
pub enum Message {
    WorkerHello(WorkerHello),
    Initialize(Initialize),
    Ready(Ready),
    Configure(Configure),
    Configured(Configured),
    Recognize(Recognize),
    Cancel(Cancel),
    Progress(Progress),
    RecognitionResult(RecognitionResult),
    RequestFailed(RequestFailed),
    Ping(Ping),
    Pong(Pong),
    Unload(Unload),
    Shutdown(Shutdown),
    ShutdownComplete(ShutdownComplete),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtocolLimits {
    pub max_json_bytes: u32,
    pub max_attachment_bytes: u64,
    pub max_image_dimension: u32,
    pub max_image_pixels: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerHello {
    pub worker_version: String,
    pub target_os: String,
    pub target_arch: String,
    pub protocol_minor: u16,
    pub limits: ProtocolLimits,
    pub providers: Vec<ProviderCapability>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderCapability {
    pub provider: ProviderKind,
    pub compiled: bool,
    pub ort_reported_available: bool,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    Cpu,
    Cuda,
    DirectMl,
    CoreMl,
    WebGpu,
    MiGraphX,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Initialize {
    pub request_id: u64,
    pub protocol_minor: u16,
    /// `None` keeps the worker warm until explicitly stopped.
    pub idle_timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ready {
    pub request_id: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Configure {
    pub request_id: u64,
    pub configuration_id: u64,
    pub config: PipelineConfig,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub document_orientation_model: Option<String>,
    pub document_rectification_model: Option<String>,
    pub text_detection_model: String,
    pub textline_orientation_model: Option<String>,
    pub text_recognition_model: String,
    pub provider: ProviderConfig,
    pub runtime: RuntimeConfig,
    pub detection: DetectionConfig,
    pub textline_orientation_batch_size: usize,
    pub text_recognition_batch_size: usize,
    pub text_recognition_score_threshold: f32,
    pub return_word_boxes: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            document_orientation_model: Some("PP-LCNet_x1_0_doc_ori".to_owned()),
            document_rectification_model: Some("UVDoc".to_owned()),
            text_detection_model: "PP-OCRv6_medium_det".to_owned(),
            textline_orientation_model: Some("PP-LCNet_x1_0_textline_ori".to_owned()),
            text_recognition_model: "PP-OCRv6_medium_rec".to_owned(),
            provider: ProviderConfig::Cpu,
            runtime: RuntimeConfig::default(),
            detection: DetectionConfig::default(),
            textline_orientation_batch_size: 6,
            text_recognition_batch_size: 6,
            text_recognition_score_threshold: 0.0,
            return_word_boxes: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProviderConfig {
    Cpu,
    Cuda { device_id: i32 },
    DirectMl { device_id: i32 },
    CoreMl,
    WebGpu { device_id: i32 },
    MiGraphX { device_id: i32 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeConfig {
    pub intra_threads: Option<usize>,
    pub inter_threads: Option<usize>,
    pub parallel_execution: bool,
    pub optimization_level: OptimizationLevel,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            intra_threads: None,
            inter_threads: None,
            parallel_execution: false,
            optimization_level: OptimizationLevel::All,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizationLevel {
    Disable,
    Level1,
    Level2,
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DetectionConfig {
    pub limit_side_len: Option<usize>,
    pub limit_type: Option<DetectionSizeLimit>,
    pub max_side_limit: usize,
    pub threshold: Option<f32>,
    pub box_threshold: Option<f32>,
    pub unclip_ratio: Option<f32>,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            limit_side_len: Some(64),
            limit_type: Some(DetectionSizeLimit::Min),
            max_side_limit: 4000,
            threshold: Some(0.3),
            box_threshold: Some(0.6),
            unclip_ratio: Some(1.5),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DetectionSizeLimit {
    Max,
    Min,
    ResizeLong,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Configured {
    pub request_id: u64,
    pub configuration_id: u64,
    pub provider: ProviderKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Recognize {
    pub request_id: u64,
    pub job_id: u64,
    pub configuration_id: u64,
    pub source_name: Option<String>,
    pub image: ImageDescriptor,
    pub processed_image: ProcessedImagePolicy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ImageDescriptor {
    Encoded {
        format: EncodedImageFormat,
    },
    RawRgba8 {
        width: u32,
        height: u32,
        stride: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EncodedImageFormat {
    Png,
    Jpeg,
    WebP,
    Tiff,
    Bmp,
    Gif,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessedImagePolicy {
    Never,
    WhenTransformed,
    Always,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cancel {
    pub request_id: u64,
    pub job_id: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Progress {
    pub request_id: u64,
    pub job_id: Option<u64>,
    pub phase: ProgressPhase,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProgressPhase {
    PreparingModels,
    InitializingSessions,
    Decoding,
    Recognizing,
    EncodingResult,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecognitionResult {
    pub request_id: u64,
    pub job_id: u64,
    pub configuration_id: u64,
    pub elapsed_ms: u64,
    pub metadata: OcrMetadata,
    pub processed_image_attached: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OcrMetadata {
    pub input_size: ImageSize,
    pub processed_size: ImageSize,
    pub model_settings: ModelSettings,
    pub model_ids: ModelIds,
    pub text_detection_parameters: EffectiveDetectionParameters,
    pub text_recognition_score_threshold: f32,
    pub return_word_boxes: bool,
    pub document_orientation: Option<ClassificationResult>,
    pub detections: Vec<DetectedRegion>,
    pub textline_orientations: Vec<Option<ClassificationResult>>,
    pub text_lines: Vec<TextLine>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageSize {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSettings {
    pub use_document_orientation: bool,
    pub use_document_rectification: bool,
    pub use_textline_orientation: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelIds {
    pub document_orientation: Option<String>,
    pub document_rectification: Option<String>,
    pub text_detection: String,
    pub textline_orientation: Option<String>,
    pub text_recognition: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EffectiveDetectionParameters {
    pub limit_side_len: usize,
    pub limit_type: DetectionSizeLimit,
    pub max_side_limit: usize,
    pub threshold: f32,
    pub box_threshold: f32,
    pub unclip_ratio: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassificationResult {
    pub class_id: usize,
    pub angle: u16,
    pub label: String,
    pub score: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectedRegion {
    pub points: [Point; 4],
    pub score: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextLine {
    pub detection_index: usize,
    pub polygon: [Point; 4],
    pub bounding_box: [f32; 4],
    pub detection_score: f32,
    pub textline_orientation: Option<ClassificationResult>,
    pub text: String,
    pub recognition_score: f32,
    pub words: Option<Vec<Word>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Word {
    pub text: String,
    pub polygon: [Point; 4],
    pub bounding_box: [f32; 4],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestFailed {
    pub request_id: Option<u64>,
    pub job_id: Option<u64>,
    pub code: ErrorCode,
    pub message: String,
    pub retryable: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    Protocol,
    InvalidState,
    InvalidRequest,
    ConfigurationRequiresRestart,
    UnsupportedProvider,
    ModelDownload,
    ImageDecode,
    PipelineConfiguration,
    Inference,
    Cancelled,
    Internal,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ping {
    pub request_id: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pong {
    pub request_id: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Unload {
    pub request_id: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shutdown {
    pub request_id: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShutdownComplete {
    pub request_id: u64,
}
