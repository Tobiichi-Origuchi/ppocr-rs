//! Composition of the five PaddleOCR general OCR module roles.

use std::cmp::Ordering;

use image::RgbImage;

use crate::document_orientation::{
    DocumentOrientationClassifier, DocumentOrientationClassifierOptions, DocumentOrientationResult,
};
use crate::document_rectification::{DocumentRectifier, DocumentRectifierOptions};
use crate::image_geometry::{crop_text_region, rotate_image, sort_text_regions};
use crate::models::{ModelDownloadError, ModelDownloader, ModelKind, manifest};
use crate::text_detection::{DetectedTextRegion, TextDetector, TextDetectorOptions};
use crate::text_recognition::{TextRecognitionResult, TextRecognizer, TextRecognizerOptions};
use crate::textline_orientation::{
    TextlineOrientationClassifier, TextlineOrientationClassifierOptions, TextlineOrientationResult,
};
use crate::{OcrError, Result};

pub const DEFAULT_DOCUMENT_ORIENTATION_MODEL: &str = "PP-LCNet_x1_0_doc_ori";
pub const DEFAULT_DOCUMENT_RECTIFICATION_MODEL: &str = "UVDoc";
pub const DEFAULT_TEXTLINE_ORIENTATION_MODEL: &str = "PP-LCNet_x1_0_textline_ori";
pub const DEFAULT_TEXT_DETECTION_MODEL: &str = "PP-OCRv6_medium_det";
pub const DEFAULT_TEXT_RECOGNITION_MODEL: &str = "PP-OCRv6_medium_rec";

/// Runtime composition settings independent of the selected model artifacts.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct OcrPipelineOptions {
    /// Maximum number of crops passed to one text-line orientation call.
    pub textline_orientation_batch_size: usize,
    /// Maximum number of aspect-ratio-sorted crops passed to one recognition call.
    pub text_recognition_batch_size: usize,
}

impl Default for OcrPipelineOptions {
    fn default() -> Self {
        Self {
            textline_orientation_batch_size: 6,
            text_recognition_batch_size: 6,
        }
    }
}

/// Unfiltered result for one detected text region in reading order.
#[derive(Debug, Clone)]
pub struct OcrPipelineRegion {
    pub detection: DetectedTextRegion,
    pub textline_orientation: Option<TextlineOrientationResult>,
    pub recognition: TextRecognitionResult,
}

/// One raw pipeline result before result-score filtering and metadata expansion.
#[derive(Debug, Clone)]
pub struct OcrPipelineResult {
    /// Image after enabled document orientation and rectification stages.
    pub processed_image: RgbImage,
    pub document_orientation: Option<DocumentOrientationResult>,
    /// Regions are restored to detector reading order after recognition batching.
    pub regions: Vec<OcrPipelineRegion>,
}

/// Lazily constructs only the explicitly enabled OCR modules.
#[derive(Debug)]
pub struct OcrPipelineBuilder<'a> {
    downloader: &'a ModelDownloader,
    document_orientation_model: Option<String>,
    document_rectification_model: Option<String>,
    textline_orientation_model: Option<String>,
    text_detection_model: String,
    text_recognition_model: String,
    document_orientation_options: DocumentOrientationClassifierOptions,
    document_rectification_options: DocumentRectifierOptions,
    textline_orientation_options: TextlineOrientationClassifierOptions,
    text_detection_options: TextDetectorOptions,
    text_recognition_options: TextRecognizerOptions,
    pipeline_options: OcrPipelineOptions,
}

impl<'a> OcrPipelineBuilder<'a> {
    /// Use the current PaddleX general OCR defaults with all optional stages enabled.
    pub fn new(downloader: &'a ModelDownloader) -> Self {
        let text_detection_options = TextDetectorOptions {
            limit_side_len: Some(64),
            limit_type: Some(crate::DetectionSizeLimit::Min),
            threshold: Some(0.3),
            box_threshold: Some(0.6),
            unclip_ratio: Some(1.5),
            ..TextDetectorOptions::default()
        };
        Self {
            downloader,
            document_orientation_model: Some(DEFAULT_DOCUMENT_ORIENTATION_MODEL.to_owned()),
            document_rectification_model: Some(DEFAULT_DOCUMENT_RECTIFICATION_MODEL.to_owned()),
            textline_orientation_model: Some(DEFAULT_TEXTLINE_ORIENTATION_MODEL.to_owned()),
            text_detection_model: DEFAULT_TEXT_DETECTION_MODEL.to_owned(),
            text_recognition_model: DEFAULT_TEXT_RECOGNITION_MODEL.to_owned(),
            document_orientation_options: DocumentOrientationClassifierOptions::default(),
            document_rectification_options: DocumentRectifierOptions::default(),
            textline_orientation_options: TextlineOrientationClassifierOptions::default(),
            text_detection_options,
            text_recognition_options: TextRecognizerOptions::default(),
            pipeline_options: OcrPipelineOptions::default(),
        }
    }

    pub fn text_detection_model(mut self, model_id: impl Into<String>) -> Self {
        self.text_detection_model = model_id.into();
        self
    }

    pub fn text_recognition_model(mut self, model_id: impl Into<String>) -> Self {
        self.text_recognition_model = model_id.into();
        self
    }

    pub fn document_orientation_model(mut self, model_id: impl Into<String>) -> Self {
        self.document_orientation_model = Some(model_id.into());
        self
    }

    pub fn disable_document_orientation(mut self) -> Self {
        self.document_orientation_model = None;
        self
    }

    pub fn document_rectification_model(mut self, model_id: impl Into<String>) -> Self {
        self.document_rectification_model = Some(model_id.into());
        self
    }

    pub fn disable_document_rectification(mut self) -> Self {
        self.document_rectification_model = None;
        self
    }

    pub fn textline_orientation_model(mut self, model_id: impl Into<String>) -> Self {
        self.textline_orientation_model = Some(model_id.into());
        self
    }

    pub fn disable_textline_orientation(mut self) -> Self {
        self.textline_orientation_model = None;
        self
    }

    pub fn document_orientation_options(
        mut self,
        options: DocumentOrientationClassifierOptions,
    ) -> Self {
        self.document_orientation_options = options;
        self
    }

    pub fn document_rectification_options(mut self, options: DocumentRectifierOptions) -> Self {
        self.document_rectification_options = options;
        self
    }

    pub fn textline_orientation_options(
        mut self,
        options: TextlineOrientationClassifierOptions,
    ) -> Self {
        self.textline_orientation_options = options;
        self
    }

    pub fn text_detection_options(mut self, options: TextDetectorOptions) -> Self {
        self.text_detection_options = options;
        self
    }

    pub fn text_recognition_options(mut self, options: TextRecognizerOptions) -> Self {
        self.text_recognition_options = options;
        self
    }

    pub fn pipeline_options(mut self, options: OcrPipelineOptions) -> Self {
        self.pipeline_options = options;
        self
    }

    /// Validate every selection before downloading, then initialize only enabled modules.
    pub fn build(self) -> Result<OcrPipeline> {
        self.validate()?;
        let document_orientation = self
            .document_orientation_model
            .as_deref()
            .map(|model_id| {
                DocumentOrientationClassifier::from_model(
                    model_id,
                    self.downloader,
                    self.document_orientation_options,
                )
            })
            .transpose()?;
        let document_rectifier = self
            .document_rectification_model
            .as_deref()
            .map(|model_id| {
                DocumentRectifier::from_model(
                    model_id,
                    self.downloader,
                    self.document_rectification_options,
                )
            })
            .transpose()?;
        let detector = TextDetector::from_model(
            &self.text_detection_model,
            self.downloader,
            self.text_detection_options,
        )?;
        let textline_orientation = self
            .textline_orientation_model
            .as_deref()
            .map(|model_id| {
                TextlineOrientationClassifier::from_model(
                    model_id,
                    self.downloader,
                    self.textline_orientation_options,
                )
            })
            .transpose()?;
        let recognizer = TextRecognizer::from_model(
            &self.text_recognition_model,
            self.downloader,
            self.text_recognition_options,
        )?;
        OcrPipeline::from_components(
            detector,
            recognizer,
            document_orientation,
            document_rectifier,
            textline_orientation,
            self.pipeline_options,
        )
    }

    fn validate(&self) -> Result<()> {
        validate_options(self.pipeline_options)?;
        validate_model_kind(&self.text_detection_model, ModelKind::TextDetection)?;
        validate_model_kind(&self.text_recognition_model, ModelKind::TextRecognition)?;
        if let Some(model_id) = self.document_orientation_model.as_deref() {
            validate_model_kind(model_id, ModelKind::DocumentOrientation)?;
        }
        if let Some(model_id) = self.document_rectification_model.as_deref() {
            validate_model_kind(model_id, ModelKind::DocumentRectification)?;
        }
        if let Some(model_id) = self.textline_orientation_model.as_deref() {
            validate_model_kind(model_id, ModelKind::TextlineOrientation)?;
        }
        Ok(())
    }
}

/// CPU ONNX Runtime implementation of the PaddleOCR general OCR pipeline.
pub struct OcrPipeline {
    detector: TextDetector,
    recognizer: TextRecognizer,
    document_orientation: Option<DocumentOrientationClassifier>,
    document_rectifier: Option<DocumentRectifier>,
    textline_orientation: Option<TextlineOrientationClassifier>,
    options: OcrPipelineOptions,
}

impl std::fmt::Debug for OcrPipeline {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OcrPipeline")
            .field("text_detection_model", &self.detector.model_id())
            .field("text_recognition_model", &self.recognizer.model_id())
            .field(
                "document_orientation_model",
                &self
                    .document_orientation
                    .as_ref()
                    .map(DocumentOrientationClassifier::model_id),
            )
            .field(
                "document_rectification_model",
                &self
                    .document_rectifier
                    .as_ref()
                    .map(DocumentRectifier::model_id),
            )
            .field(
                "textline_orientation_model",
                &self
                    .textline_orientation
                    .as_ref()
                    .map(TextlineOrientationClassifier::model_id),
            )
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}

impl OcrPipeline {
    pub fn builder(downloader: &ModelDownloader) -> OcrPipelineBuilder<'_> {
        OcrPipelineBuilder::new(downloader)
    }

    /// Compose already initialized modules without any cache or network access.
    pub fn from_components(
        detector: TextDetector,
        recognizer: TextRecognizer,
        document_orientation: Option<DocumentOrientationClassifier>,
        document_rectifier: Option<DocumentRectifier>,
        textline_orientation: Option<TextlineOrientationClassifier>,
        options: OcrPipelineOptions,
    ) -> Result<Self> {
        validate_options(options)?;
        Ok(Self {
            detector,
            recognizer,
            document_orientation,
            document_rectifier,
            textline_orientation,
            options,
        })
    }

    /// Run all enabled modules in PaddleX order for one decoded RGB image.
    pub fn predict(&mut self, image: &RgbImage) -> Result<OcrPipelineResult> {
        if image.width() == 0 || image.height() == 0 {
            return Err(OcrError::EmptyImage);
        }

        let document_orientation = self
            .document_orientation
            .as_mut()
            .map(|classifier| classifier.classify(image))
            .transpose()?;
        let mut processed_image = match &document_orientation {
            Some(result) => rotate_image(image, result.angle)?,
            None => image.clone(),
        };
        if let Some(rectifier) = self.document_rectifier.as_mut() {
            processed_image = rectifier.rectify(&processed_image)?;
        }

        let detections = self.detector.detect(&processed_image)?;
        let sorted_regions = sort_text_regions(&detections.regions);
        let mut regions = Vec::with_capacity(sorted_regions.len());
        let mut crops = Vec::with_capacity(sorted_regions.len());
        for region in sorted_regions {
            match crop_text_region(&processed_image, &region.points) {
                Ok(crop) => {
                    regions.push(region);
                    crops.push(crop);
                }
                Err(OcrError::InvalidGeometry { .. }) => {}
                Err(error) => return Err(error),
            }
        }

        let textline_orientations = self.classify_and_rotate_textlines(&mut crops)?;
        let recognitions = self.recognize_in_aspect_ratio_order(&crops)?;
        let regions = regions
            .into_iter()
            .zip(textline_orientations)
            .zip(recognitions)
            .map(
                |((detection, textline_orientation), recognition)| OcrPipelineRegion {
                    detection,
                    textline_orientation,
                    recognition,
                },
            )
            .collect();

        Ok(OcrPipelineResult {
            processed_image,
            document_orientation,
            regions,
        })
    }

    /// Run documents sequentially so each detector/UVDoc invocation keeps its natural shape.
    pub fn predict_batch(&mut self, images: &[RgbImage]) -> Result<Vec<OcrPipelineResult>> {
        images.iter().map(|image| self.predict(image)).collect()
    }

    fn classify_and_rotate_textlines(
        &mut self,
        crops: &mut [RgbImage],
    ) -> Result<Vec<Option<TextlineOrientationResult>>> {
        let Some(classifier) = self.textline_orientation.as_mut() else {
            return Ok(vec![None; crops.len()]);
        };
        let mut results = Vec::with_capacity(crops.len());
        for chunk in crops.chunks(self.options.textline_orientation_batch_size) {
            results.extend(classifier.classify_batch(chunk)?);
        }
        for (crop, result) in crops.iter_mut().zip(&results) {
            if result.angle != 0 {
                *crop = rotate_image(crop, result.angle)?;
            }
        }
        Ok(results.into_iter().map(Some).collect())
    }

    fn recognize_in_aspect_ratio_order(
        &mut self,
        crops: &[RgbImage],
    ) -> Result<Vec<TextRecognitionResult>> {
        let mut indices: Vec<usize> = (0..crops.len()).collect();
        indices.sort_by(|left, right| {
            aspect_ratio(&crops[*left])
                .partial_cmp(&aspect_ratio(&crops[*right]))
                .unwrap_or(Ordering::Equal)
        });
        let mut restored = vec![None; crops.len()];
        for chunk in indices.chunks(self.options.text_recognition_batch_size) {
            let images: Vec<&RgbImage> = chunk.iter().map(|index| &crops[*index]).collect();
            let results = self.recognizer.recognize_batch_refs(&images)?;
            for (&original_index, result) in chunk.iter().zip(results) {
                restored[original_index] = Some(result);
            }
        }
        Ok(restored
            .into_iter()
            .map(|result| result.expect("every recognition result is restored exactly once"))
            .collect())
    }
}

fn aspect_ratio(image: &RgbImage) -> f64 {
    f64::from(image.width()) / f64::from(image.height())
}

fn validate_options(options: OcrPipelineOptions) -> Result<()> {
    if options.textline_orientation_batch_size == 0 {
        return Err(OcrError::InvalidPipelineConfig {
            reason: "textline_orientation_batch_size must be greater than zero".to_owned(),
        });
    }
    if options.text_recognition_batch_size == 0 {
        return Err(OcrError::InvalidPipelineConfig {
            reason: "text_recognition_batch_size must be greater than zero".to_owned(),
        });
    }
    Ok(())
}

fn validate_model_kind(model_id: &str, expected: ModelKind) -> Result<()> {
    let spec = manifest()
        .get(model_id)
        .ok_or_else(|| ModelDownloadError::UnknownModel(model_id.to_owned()))?;
    if spec.kind != expected {
        return Err(OcrError::WrongModelKind {
            model_id: model_id.to_owned(),
            actual: spec.kind,
            expected,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;
    use crate::models::ModelDownloaderOptions;

    fn downloader() -> ModelDownloader {
        ModelDownloader::new(ModelDownloaderOptions {
            cache_dir: Some(tempdir().unwrap().keep()),
            local_files_only: true,
            force_download: false,
        })
        .unwrap()
    }

    #[test]
    fn defaults_match_current_paddlex_general_ocr_config() {
        let downloader = downloader();
        let builder = OcrPipelineBuilder::new(&downloader);
        assert_eq!(
            builder.document_orientation_model.as_deref(),
            Some(DEFAULT_DOCUMENT_ORIENTATION_MODEL)
        );
        assert_eq!(
            builder.document_rectification_model.as_deref(),
            Some(DEFAULT_DOCUMENT_RECTIFICATION_MODEL)
        );
        assert_eq!(
            builder.textline_orientation_model.as_deref(),
            Some(DEFAULT_TEXTLINE_ORIENTATION_MODEL)
        );
        assert_eq!(builder.text_detection_model, DEFAULT_TEXT_DETECTION_MODEL);
        assert_eq!(
            builder.text_recognition_model,
            DEFAULT_TEXT_RECOGNITION_MODEL
        );
        assert_eq!(builder.pipeline_options, OcrPipelineOptions::default());
        builder.validate().unwrap();
    }

    #[test]
    fn optional_models_can_all_be_excluded_before_download() {
        let downloader = downloader();
        let builder = OcrPipelineBuilder::new(&downloader)
            .disable_document_orientation()
            .disable_document_rectification()
            .disable_textline_orientation();
        assert!(builder.document_orientation_model.is_none());
        assert!(builder.document_rectification_model.is_none());
        assert!(builder.textline_orientation_model.is_none());
        builder.validate().unwrap();
    }

    #[test]
    fn selection_is_validated_without_cache_or_network_access() {
        let downloader = downloader();
        let error = OcrPipelineBuilder::new(&downloader)
            .text_detection_model(DEFAULT_TEXT_RECOGNITION_MODEL)
            .validate()
            .unwrap_err();
        assert!(matches!(error, OcrError::WrongModelKind { .. }));

        let error = OcrPipelineBuilder::new(&downloader)
            .text_recognition_model("unknown-model")
            .validate()
            .unwrap_err();
        assert!(matches!(
            error,
            OcrError::Download(ModelDownloadError::UnknownModel(_))
        ));
    }

    #[test]
    fn rejects_zero_module_batch_sizes() {
        let downloader = downloader();
        let error = OcrPipelineBuilder::new(&downloader)
            .pipeline_options(OcrPipelineOptions {
                textline_orientation_batch_size: 0,
                text_recognition_batch_size: 6,
            })
            .validate()
            .unwrap_err();
        assert!(matches!(error, OcrError::InvalidPipelineConfig { .. }));
    }
}
