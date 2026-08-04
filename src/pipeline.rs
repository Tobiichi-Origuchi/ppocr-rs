//! Composition of the five PaddleOCR general OCR module roles.

use std::cmp::Ordering;

use image::RgbImage;
use serde::{Deserialize, Serialize};

use crate::document_orientation::{
    DocumentOrientationClassifier, DocumentOrientationClassifierOptions, DocumentOrientationResult,
};
use crate::document_rectification::{DocumentRectifier, DocumentRectifierOptions};
use crate::image_geometry::{crop_text_region, rotate_image, sort_text_regions};
use crate::models::{ModelDownloadError, ModelDownloader, ModelKind, manifest};
use crate::text_detection::{
    DetectedTextRegion, Point, TextDetectionParameters, TextDetector, TextDetectorOptions,
};
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OcrPipelineOptions {
    /// Maximum number of crops passed to one text-line orientation call.
    pub textline_orientation_batch_size: usize,
    /// Maximum number of aspect-ratio-sorted crops passed to one recognition call.
    pub text_recognition_batch_size: usize,
    /// Keep recognized lines whose mean CTC confidence is at least this value.
    pub text_recognition_score_threshold: f32,
}

impl Default for OcrPipelineOptions {
    fn default() -> Self {
        Self {
            textline_orientation_batch_size: 6,
            text_recognition_batch_size: 6,
            text_recognition_score_threshold: 0.0,
        }
    }
}

/// Width and height metadata that can be serialized without image pixels.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct OcrImageSize {
    pub width: u32,
    pub height: u32,
}

impl From<&RgbImage> for OcrImageSize {
    fn from(image: &RgbImage) -> Self {
        Self {
            width: image.width(),
            height: image.height(),
        }
    }
}

/// Enabled/disabled state of every optional pipeline stage.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct OcrPipelineModelSettings {
    pub use_document_orientation: bool,
    pub use_document_rectification: bool,
    pub use_textline_orientation: bool,
}

/// Actual manifest IDs backing all required and enabled pipeline stages.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct OcrPipelineModelIds {
    pub document_orientation: Option<String>,
    pub document_rectification: Option<String>,
    pub text_detection: String,
    pub textline_orientation: Option<String>,
    pub text_recognition: String,
}

/// One score-filtered recognized line in detector reading order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OcrTextLine {
    /// Index into [`OcrPipelineResult::detections`] and
    /// [`OcrPipelineResult::textline_orientations`].
    pub detection_index: usize,
    pub polygon: [Point; 4],
    /// `[left, top, right, bottom]` enclosing `polygon`.
    pub bounding_box: [f32; 4],
    pub detection_score: f32,
    pub textline_orientation: Option<TextlineOrientationResult>,
    pub text: String,
    pub recognition_score: f32,
}

/// Aggregate PaddleOCR output for one input image.
#[derive(Debug, Clone)]
pub struct OcrPipelineResult {
    pub input_size: OcrImageSize,
    pub processed_size: OcrImageSize,
    /// Image after enabled document orientation and rectification stages.
    pub processed_image: RgbImage,
    pub model_settings: OcrPipelineModelSettings,
    pub model_ids: OcrPipelineModelIds,
    pub text_detection_parameters: TextDetectionParameters,
    pub text_recognition_score_threshold: f32,
    pub document_orientation: Option<DocumentOrientationResult>,
    /// Every sorted, geometrically valid detection, regardless of recognition score.
    pub detections: Vec<DetectedTextRegion>,
    /// One entry per detection; `None` when the optional classifier is disabled.
    pub textline_orientations: Vec<Option<TextlineOrientationResult>>,
    /// Score-filtered lines, restored to detector reading order after batching.
    pub text_lines: Vec<OcrTextLine>,
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

    pub fn text_recognition_score_threshold(mut self, threshold: f32) -> Self {
        self.pipeline_options.text_recognition_score_threshold = threshold;
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
    model_settings: OcrPipelineModelSettings,
    model_ids: OcrPipelineModelIds,
    text_detection_parameters: TextDetectionParameters,
}

impl std::fmt::Debug for OcrPipeline {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OcrPipeline")
            .field("model_settings", &self.model_settings)
            .field("model_ids", &self.model_ids)
            .field("text_detection_parameters", &self.text_detection_parameters)
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
        let model_settings = OcrPipelineModelSettings {
            use_document_orientation: document_orientation.is_some(),
            use_document_rectification: document_rectifier.is_some(),
            use_textline_orientation: textline_orientation.is_some(),
        };
        let model_ids = OcrPipelineModelIds {
            document_orientation: document_orientation
                .as_ref()
                .map(DocumentOrientationClassifier::model_id)
                .map(str::to_owned),
            document_rectification: document_rectifier
                .as_ref()
                .map(DocumentRectifier::model_id)
                .map(str::to_owned),
            text_detection: detector.model_id().to_owned(),
            textline_orientation: textline_orientation
                .as_ref()
                .map(TextlineOrientationClassifier::model_id)
                .map(str::to_owned),
            text_recognition: recognizer.model_id().to_owned(),
        };
        let text_detection_parameters = detector.parameters();
        Ok(Self {
            detector,
            recognizer,
            document_orientation,
            document_rectifier,
            textline_orientation,
            options,
            model_settings,
            model_ids,
            text_detection_parameters,
        })
    }

    pub fn model_settings(&self) -> OcrPipelineModelSettings {
        self.model_settings
    }

    pub fn model_ids(&self) -> &OcrPipelineModelIds {
        &self.model_ids
    }

    pub fn text_detection_parameters(&self) -> TextDetectionParameters {
        self.text_detection_parameters
    }

    /// Run all enabled modules in PaddleX order for one decoded RGB image.
    pub fn predict(&mut self, image: &RgbImage) -> Result<OcrPipelineResult> {
        if image.width() == 0 || image.height() == 0 {
            return Err(OcrError::EmptyImage);
        }

        let input_size = OcrImageSize::from(image);
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
        let (detections, textline_orientations, text_lines) = aggregate_text_lines(
            regions,
            textline_orientations,
            recognitions,
            self.options.text_recognition_score_threshold,
        );

        Ok(OcrPipelineResult {
            input_size,
            processed_size: OcrImageSize::from(&processed_image),
            processed_image,
            model_settings: self.model_settings,
            model_ids: self.model_ids.clone(),
            text_detection_parameters: self.text_detection_parameters,
            text_recognition_score_threshold: self.options.text_recognition_score_threshold,
            document_orientation,
            detections,
            textline_orientations,
            text_lines,
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

fn aggregate_text_lines(
    detections: Vec<DetectedTextRegion>,
    textline_orientations: Vec<Option<TextlineOrientationResult>>,
    recognitions: Vec<TextRecognitionResult>,
    score_threshold: f32,
) -> (
    Vec<DetectedTextRegion>,
    Vec<Option<TextlineOrientationResult>>,
    Vec<OcrTextLine>,
) {
    assert_eq!(detections.len(), textline_orientations.len());
    assert_eq!(detections.len(), recognitions.len());
    let text_lines = detections
        .iter()
        .zip(&textline_orientations)
        .zip(recognitions)
        .enumerate()
        .filter_map(
            |(detection_index, ((detection, textline_orientation), recognition))| {
                (recognition.score >= score_threshold).then(|| OcrTextLine {
                    detection_index,
                    polygon: detection.points,
                    bounding_box: enclosing_box(&detection.points),
                    detection_score: detection.score,
                    textline_orientation: textline_orientation.clone(),
                    text: recognition.text,
                    recognition_score: recognition.score,
                })
            },
        )
        .collect();
    (detections, textline_orientations, text_lines)
}

fn enclosing_box(points: &[Point; 4]) -> [f32; 4] {
    let mut left = f32::INFINITY;
    let mut top = f32::INFINITY;
    let mut right = f32::NEG_INFINITY;
    let mut bottom = f32::NEG_INFINITY;
    for point in points {
        left = left.min(point.x);
        top = top.min(point.y);
        right = right.max(point.x);
        bottom = bottom.max(point.y);
    }
    [left, top, right, bottom]
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
    if !options.text_recognition_score_threshold.is_finite() {
        return Err(OcrError::InvalidPipelineConfig {
            reason: "text_recognition_score_threshold must be finite".to_owned(),
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
                text_recognition_score_threshold: 0.0,
            })
            .validate()
            .unwrap_err();
        assert!(matches!(error, OcrError::InvalidPipelineConfig { .. }));
    }

    #[test]
    fn filters_recognition_scores_without_losing_detection_associations() {
        fn detection(x: f32, score: f32) -> DetectedTextRegion {
            DetectedTextRegion {
                points: [
                    Point { x, y: 3.0 },
                    Point {
                        x: x + 10.0,
                        y: 2.0,
                    },
                    Point {
                        x: x + 11.0,
                        y: 7.0,
                    },
                    Point { x: x + 1.0, y: 8.0 },
                ],
                score,
            }
        }

        let detections = vec![
            detection(1.0, 0.91),
            detection(20.0, 0.82),
            detection(40.0, 0.73),
        ];
        let orientations = vec![
            Some(TextlineOrientationResult {
                class_id: 0,
                angle: 0,
                label: "0_degree".to_owned(),
                score: 0.99,
            }),
            Some(TextlineOrientationResult {
                class_id: 1,
                angle: 180,
                label: "180_degree".to_owned(),
                score: 0.98,
            }),
            None,
        ];
        let recognitions = vec![
            TextRecognitionResult {
                text: "first".to_owned(),
                score: 0.9,
            },
            TextRecognitionResult {
                text: "filtered".to_owned(),
                score: 0.79,
            },
            TextRecognitionResult {
                text: "threshold".to_owned(),
                score: 0.8,
            },
        ];

        let (all_detections, all_orientations, lines) =
            aggregate_text_lines(detections, orientations, recognitions, 0.8);
        assert_eq!(all_detections.len(), 3);
        assert_eq!(all_orientations.len(), 3);
        assert_eq!(
            lines
                .iter()
                .map(|line| (line.detection_index, line.text.as_str()))
                .collect::<Vec<_>>(),
            vec![(0, "first"), (2, "threshold")]
        );
        assert_eq!(lines[0].bounding_box, [1.0, 2.0, 12.0, 8.0]);
        assert_eq!(lines[0].detection_score, 0.91);
        assert_eq!(lines[0].textline_orientation.as_ref().unwrap().angle, 0);
        assert!(lines[1].textline_orientation.is_none());
    }

    #[test]
    fn rejects_non_finite_score_threshold() {
        let downloader = downloader();
        let error = OcrPipelineBuilder::new(&downloader)
            .text_recognition_score_threshold(f32::NAN)
            .validate()
            .unwrap_err();
        assert!(matches!(error, OcrError::InvalidPipelineConfig { .. }));
    }

    #[test]
    fn serializes_result_metadata_types() {
        let settings = OcrPipelineModelSettings {
            use_document_orientation: true,
            use_document_rectification: false,
            use_textline_orientation: true,
        };
        let model_ids = OcrPipelineModelIds {
            document_orientation: Some(DEFAULT_DOCUMENT_ORIENTATION_MODEL.to_owned()),
            document_rectification: None,
            text_detection: DEFAULT_TEXT_DETECTION_MODEL.to_owned(),
            textline_orientation: Some(DEFAULT_TEXTLINE_ORIENTATION_MODEL.to_owned()),
            text_recognition: DEFAULT_TEXT_RECOGNITION_MODEL.to_owned(),
        };
        let value = serde_json::json!({
            "size": OcrImageSize { width: 720, height: 1150 },
            "settings": settings,
            "model_ids": model_ids,
        });
        assert_eq!(value["size"]["width"], 720);
        assert_eq!(value["settings"]["use_document_rectification"], false);
        assert_eq!(
            value["model_ids"]["text_recognition"],
            DEFAULT_TEXT_RECOGNITION_MODEL
        );
    }
}
