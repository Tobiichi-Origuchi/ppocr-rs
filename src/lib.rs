//! Rust support for PaddleOCR models and pipelines.
//!
//! Only the model manifest and lazy Hugging Face download layer are implemented
//! at this stage. Inference and OCR pipeline APIs will be added separately.

pub mod models;

pub use models::{
    ArtifactOrigin, ArtifactStatus, ConversionValidation, DownloadedModel, ModelArtifact,
    ModelConversion, ModelDownloadError, ModelDownloader, ModelDownloaderOptions, ModelFile,
    ModelKind, ModelManifest, ModelSpec, manifest, resolve_hf_hub_cache_dir,
};
