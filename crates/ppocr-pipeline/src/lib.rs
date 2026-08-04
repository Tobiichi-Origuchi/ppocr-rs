//! PaddleOCR model management and ONNX Runtime inference implemented in Rust.
//!
//! The inference boundary is a decoded [`RgbImage`]. Local file traversal, URL
//! retrieval, PDF rasterization, page limits, request timeouts, and SSRF policy
//! belong in an application or a separate input-adapter crate. This keeps the
//! core usable without an HTTP client or a platform-specific PDF renderer.
//!
//! ONNX Runtime execution providers are configured through
//! [`OrtSessionOptions::execution_providers`]. The [`execution_providers`]
//! re-export exposes the provider builders compiled into `ort`; this crate has
//! convenience Cargo features for `cuda`, `webgpu`, `coreml`, `directml`, and
//! `migraphx`:
//!
//! - `cuda` is the NVIDIA path and requires matching CUDA and cuDNN user-space
//!   libraries at runtime.
//! - `webgpu` uses ONNX Runtime's Dawn-based WebGPU provider. Dawn maps to
//!   Vulkan on Linux, D3D12 or Vulkan on Windows, and Metal on Apple systems;
//!   Vulkan and Metal are not separate ONNX Runtime execution providers.
//! - `coreml` is the native Apple CPU/GPU/Neural Engine path.
//! - `directml` is the native Windows GPU path.
//! - `migraphx` exposes AMD's preferred provider binding, but does not supply
//!   its required custom ONNX Runtime/ROCm/MIGraphX runtime.
//!
//! ONNX Runtime deprecated its legacy ROCm provider. AMD deployments should
//! use MIGraphX, which currently requires a custom ONNX Runtime build because
//! `ort` does not distribute a prebuilt MIGraphX runtime.
//!
//! Provider features expose configuration APIs; actual availability remains
//! target- and runtime-specific. Use
//! [`ExecutionProviderDispatch::error_on_failure`] when selecting an accelerator
//! so a missing provider cannot silently turn into CPU execution.

mod classification;
mod config;
mod error;
mod image_ops;
mod runtime;

pub mod document_orientation;
pub mod document_rectification;
pub mod image_geometry;
pub mod models;
pub mod pipeline;
pub mod text_detection;
pub mod text_recognition;
pub mod textline_orientation;

pub use document_orientation::{
    DocumentOrientationClassifier, DocumentOrientationClassifierOptions, DocumentOrientationResult,
};
pub use document_rectification::{DocumentRectifier, DocumentRectifierOptions};
pub use error::{OcrError, Result};
pub use image::RgbImage;
pub use pipeline::{
    OcrImageSize, OcrPipeline, OcrPipelineBuilder, OcrPipelineModelIds, OcrPipelineModelSettings,
    OcrPipelineOptions, OcrPipelinePool, OcrPipelineResult, OcrTextLine, OcrWord,
};
pub use runtime::{
    ExecutionProviderDispatch, GraphOptimizationLevel, OrtSessionOptions, execution_providers,
};
pub use text_detection::{
    DetectedTextRegion, DetectionSizeLimit, Point, TextDetectionParameters, TextDetectionResult,
    TextDetector, TextDetectorOptions,
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
