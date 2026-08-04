//! PaddleOCR model management and ONNX Runtime inference implemented in Rust.

mod classification;
mod config;
mod error;
mod image_ops;
mod runtime;

pub mod document_orientation;
pub mod document_rectification;
pub mod image_geometry;
pub mod models;
pub mod text_detection;
pub mod text_recognition;
pub mod textline_orientation;

pub use document_orientation::{
    DocumentOrientationClassifier, DocumentOrientationClassifierOptions, DocumentOrientationResult,
};
pub use document_rectification::{DocumentRectifier, DocumentRectifierOptions};
pub use error::{OcrError, Result};
pub use image::RgbImage;
pub use runtime::OrtSessionOptions;
pub use text_detection::{
    DetectedTextRegion, DetectionSizeLimit, Point, TextDetectionResult, TextDetector,
    TextDetectorOptions,
};
pub use text_recognition::{TextRecognitionResult, TextRecognizer, TextRecognizerOptions};
pub use textline_orientation::{
    TextlineOrientationClassifier, TextlineOrientationClassifierOptions, TextlineOrientationResult,
};

pub use models::{
    ArtifactOrigin, ArtifactStatus, ConversionValidation, DownloadedModel, ModelArtifact,
    ModelConversion, ModelDownloadError, ModelDownloader, ModelDownloaderOptions, ModelFile,
    ModelKind, ModelManifest, ModelSpec, manifest, resolve_hf_hub_cache_dir,
};
