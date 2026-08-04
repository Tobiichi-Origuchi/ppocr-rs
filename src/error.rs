use std::path::PathBuf;

use thiserror::Error;

use crate::models::{ModelDownloadError, ModelKind};

/// Errors produced while loading or running PaddleOCR modules.
#[derive(Debug, Error)]
pub enum OcrError {
    #[error(transparent)]
    Download(#[from] ModelDownloadError),
    #[error("failed to read {path}: {source}")]
    ReadConfig {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse inference configuration {path}: {source}")]
    ParseConfig {
        path: PathBuf,
        #[source]
        source: Box<serde_saphyr::Error>,
    },
    #[error("model {model_id:?} has kind {actual:?}, expected {expected:?}")]
    WrongModelKind {
        model_id: String,
        actual: ModelKind,
        expected: ModelKind,
    },
    #[error("invalid configuration for model {model_id:?}: {reason}")]
    InvalidConfig { model_id: String, reason: String },
    #[error("invalid image: width and height must both be non-zero")]
    EmptyImage,
    #[error("invalid image geometry: {reason}")]
    InvalidGeometry { reason: String },
    #[error("invalid OCR pipeline configuration: {reason}")]
    InvalidPipelineConfig { reason: String },
    #[error("OCR concurrency error: {reason}")]
    Concurrency { reason: String },
    #[error("invalid inference output for model {model_id:?}: {reason}")]
    InvalidOutput { model_id: String, reason: String },
    #[error(transparent)]
    Ort(#[from] ort::Error),
}

/// Result type used by inference modules.
pub type Result<T> = std::result::Result<T, OcrError>;
