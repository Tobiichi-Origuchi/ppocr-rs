use std::path::Path;

use image::RgbImage;
use serde::{Deserialize, Serialize};

use crate::classification::{ClassifierOptions, ImageClassifier};
use crate::models::{DownloadedModel, ModelDownloader, ModelKind};
use crate::{OrtSessionOptions, Result};

#[derive(Debug, Clone, Default)]
pub struct DocumentOrientationClassifierOptions {
    pub session: OrtSessionOptions,
}

/// Top-1 document orientation prediction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocumentOrientationResult {
    pub class_id: usize,
    /// Counter-clockwise correction angle in degrees.
    pub angle: u16,
    /// Exact PaddleX label (`"0"`, `"90"`, `"180"`, or `"270"`).
    pub label: String,
    pub score: f32,
}

/// PaddleOCR document image orientation classifier.
pub struct DocumentOrientationClassifier {
    inner: ImageClassifier,
}

impl std::fmt::Debug for DocumentOrientationClassifier {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DocumentOrientationClassifier")
            .field("model_id", &self.model_id())
            .finish_non_exhaustive()
    }
}

impl DocumentOrientationClassifier {
    pub fn from_model(
        model_id: &str,
        downloader: &ModelDownloader,
        options: DocumentOrientationClassifierOptions,
    ) -> Result<Self> {
        Ok(Self {
            inner: ImageClassifier::from_model(
                model_id,
                ModelKind::DocumentOrientation,
                downloader,
                ClassifierOptions {
                    session: options.session,
                },
            )?,
        })
    }

    pub fn from_downloaded(
        model: &DownloadedModel,
        options: DocumentOrientationClassifierOptions,
    ) -> Result<Self> {
        Ok(Self {
            inner: ImageClassifier::from_downloaded(
                model,
                ModelKind::DocumentOrientation,
                ClassifierOptions {
                    session: options.session,
                },
            )?,
        })
    }

    pub fn from_files(
        model_id: &str,
        onnx_path: &Path,
        config_path: &Path,
        options: DocumentOrientationClassifierOptions,
    ) -> Result<Self> {
        Ok(Self {
            inner: ImageClassifier::from_files(
                model_id,
                ModelKind::DocumentOrientation,
                onnx_path,
                config_path,
                ClassifierOptions {
                    session: options.session,
                },
            )?,
        })
    }

    pub fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    pub fn classify(&mut self, image: &RgbImage) -> Result<DocumentOrientationResult> {
        let result = self.inner.classify(image)?;
        convert_result(self.model_id(), result)
    }

    pub fn classify_batch(
        &mut self,
        images: &[RgbImage],
    ) -> Result<Vec<DocumentOrientationResult>> {
        let model_id = self.model_id().to_owned();
        self.inner
            .classify_batch(images)?
            .into_iter()
            .map(|result| convert_result(&model_id, result))
            .collect()
    }
}

fn convert_result(
    model_id: &str,
    result: crate::classification::ClassificationResult,
) -> Result<DocumentOrientationResult> {
    let angle = result
        .label
        .parse::<u16>()
        .ok()
        .filter(|angle| matches!(angle, 0 | 90 | 180 | 270))
        .ok_or_else(|| crate::OcrError::InvalidOutput {
            model_id: model_id.to_owned(),
            reason: format!("unsupported document orientation label {:?}", result.label),
        })?;
    Ok(DocumentOrientationResult {
        class_id: result.class_id,
        angle,
        label: result.label,
        score: result.score,
    })
}
