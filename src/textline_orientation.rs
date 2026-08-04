use std::path::Path;

use image::RgbImage;
use serde::{Deserialize, Serialize};

use crate::classification::{ClassifierOptions, ImageClassifier};
use crate::models::{DownloadedModel, ModelDownloader, ModelKind};
use crate::{OrtSessionOptions, Result};

#[derive(Debug, Clone, Default)]
pub struct TextlineOrientationClassifierOptions {
    pub session: OrtSessionOptions,
}

/// Top-1 text-line orientation prediction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextlineOrientationResult {
    pub class_id: usize,
    /// Counter-clockwise correction angle in degrees (`0` or `180`).
    pub angle: u16,
    /// Exact PaddleX label (`"0_degree"` or `"180_degree"`).
    pub label: String,
    pub score: f32,
}

/// PaddleOCR text-line orientation classifier.
pub struct TextlineOrientationClassifier {
    inner: ImageClassifier,
}

impl std::fmt::Debug for TextlineOrientationClassifier {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TextlineOrientationClassifier")
            .field("model_id", &self.model_id())
            .finish_non_exhaustive()
    }
}

impl TextlineOrientationClassifier {
    pub fn from_model(
        model_id: &str,
        downloader: &ModelDownloader,
        options: TextlineOrientationClassifierOptions,
    ) -> Result<Self> {
        Ok(Self {
            inner: ImageClassifier::from_model(
                model_id,
                ModelKind::TextlineOrientation,
                downloader,
                ClassifierOptions {
                    session: options.session,
                },
            )?,
        })
    }

    pub fn from_downloaded(
        model: &DownloadedModel,
        options: TextlineOrientationClassifierOptions,
    ) -> Result<Self> {
        Ok(Self {
            inner: ImageClassifier::from_downloaded(
                model,
                ModelKind::TextlineOrientation,
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
        options: TextlineOrientationClassifierOptions,
    ) -> Result<Self> {
        Ok(Self {
            inner: ImageClassifier::from_files(
                model_id,
                ModelKind::TextlineOrientation,
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

    pub fn classify(&mut self, image: &RgbImage) -> Result<TextlineOrientationResult> {
        let result = self.inner.classify(image)?;
        convert_result(self.model_id(), result)
    }

    pub fn classify_batch(
        &mut self,
        images: &[RgbImage],
    ) -> Result<Vec<TextlineOrientationResult>> {
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
) -> Result<TextlineOrientationResult> {
    let angle = match result.label.as_str() {
        "0_degree" => 0,
        "180_degree" => 180,
        _ => {
            return Err(crate::OcrError::InvalidOutput {
                model_id: model_id.to_owned(),
                reason: format!("unsupported text-line orientation label {:?}", result.label),
            });
        }
    };
    Ok(TextlineOrientationResult {
        class_id: result.class_id,
        angle,
        label: result.label,
        score: result.score,
    })
}
