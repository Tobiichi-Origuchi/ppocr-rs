//! Composition of the five PaddleOCR general OCR module roles.

use std::{
    cmp::Ordering,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering as AtomicOrdering},
    },
};

use image::RgbImage;
use serde::{Deserialize, Serialize};

use crate::document_orientation::{
    DocumentOrientationClassifier, DocumentOrientationClassifierOptions, DocumentOrientationResult,
};
use crate::document_rectification::{DocumentRectifier, DocumentRectifierOptions};
use crate::image_geometry::{crop_text_region, rotate_image, sort_text_regions};
use crate::models::{
    DownloadedModel, ModelDownloadError, ModelDownloadProgressCallback, ModelDownloader, ModelKind,
    manifest,
};
use crate::text_detection::{
    DetectedTextRegion, Point, TextDetectionParameters, TextDetector, TextDetectorOptions,
};
use crate::text_recognition::{
    CtcWordAlignment, CtcWordKind, TextRecognitionResult, TextRecognizer, TextRecognizerOptions,
};
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
    /// Return PaddleX-compatible word boxes derived from CTC timestep alignment.
    pub return_word_boxes: bool,
}

impl Default for OcrPipelineOptions {
    fn default() -> Self {
        Self {
            textline_orientation_batch_size: 6,
            text_recognition_batch_size: 6,
            text_recognition_score_threshold: 0.0,
            return_word_boxes: false,
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
    /// Word/character boxes in processed-document coordinates when requested.
    pub words: Option<Vec<OcrWord>>,
}

/// One word or CJK character inferred from CTC alignment inside a text line.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OcrWord {
    pub text: String,
    pub polygon: [Point; 4],
    /// `[left, top, right, bottom]` enclosing `polygon`.
    pub bounding_box: [f32; 4],
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
    pub return_word_boxes: bool,
    pub document_orientation: Option<DocumentOrientationResult>,
    /// Every sorted, geometrically valid detection, regardless of recognition score.
    pub detections: Vec<DetectedTextRegion>,
    /// One entry per detection; `None` when the optional classifier is disabled.
    pub textline_orientations: Vec<Option<TextlineOrientationResult>>,
    /// Score-filtered lines, restored to detector reading order after batching.
    pub text_lines: Vec<OcrTextLine>,
}

/// Lazily constructs only the explicitly enabled OCR modules.
#[derive(Debug, Clone)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineBuildPhase {
    Downloading,
    Initializing,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineBuildProgress {
    pub model_index: usize,
    pub model_count: usize,
    pub model_id: String,
    pub phase: PipelineBuildPhase,
    pub bytes_completed: Option<u64>,
    pub total_bytes: Option<u64>,
}

pub type PipelineBuildProgressCallback = Arc<dyn Fn(PipelineBuildProgress) + Send + Sync + 'static>;

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

    /// Apply one ONNX Runtime configuration to every enabled pipeline module.
    ///
    /// This is the convenient path for selecting one execution-provider
    /// priority list and one threading policy for the complete pipeline.
    pub fn session_options(mut self, options: crate::OrtSessionOptions) -> Self {
        self.document_orientation_options.session = options.clone();
        self.document_rectification_options.session = options.clone();
        self.textline_orientation_options.session = options.clone();
        self.text_detection_options.session = options.clone();
        self.text_recognition_options.session = options;
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
        self.build_inner(None)
    }

    pub fn build_with_progress(
        self,
        progress: PipelineBuildProgressCallback,
    ) -> Result<OcrPipeline> {
        self.build_inner(Some(progress))
    }

    fn build_inner(self, progress: Option<PipelineBuildProgressCallback>) -> Result<OcrPipeline> {
        self.validate()?;
        let model_count = 2
            + usize::from(self.document_orientation_model.is_some())
            + usize::from(self.document_rectification_model.is_some())
            + usize::from(self.textline_orientation_model.is_some());
        let mut model_index = 0;
        let document_orientation = self
            .document_orientation_model
            .as_deref()
            .map(|model_id| {
                model_index += 1;
                let model = download_for_build(
                    self.downloader,
                    model_id,
                    model_index,
                    model_count,
                    progress.as_ref(),
                )?;
                DocumentOrientationClassifier::from_downloaded(
                    &model,
                    self.document_orientation_options,
                )
            })
            .transpose()?;
        let document_rectifier = self
            .document_rectification_model
            .as_deref()
            .map(|model_id| {
                model_index += 1;
                let model = download_for_build(
                    self.downloader,
                    model_id,
                    model_index,
                    model_count,
                    progress.as_ref(),
                )?;
                DocumentRectifier::from_downloaded(&model, self.document_rectification_options)
            })
            .transpose()?;
        model_index += 1;
        let detector_model = download_for_build(
            self.downloader,
            &self.text_detection_model,
            model_index,
            model_count,
            progress.as_ref(),
        )?;
        let detector = TextDetector::from_downloaded(&detector_model, self.text_detection_options)?;
        let textline_orientation = self
            .textline_orientation_model
            .as_deref()
            .map(|model_id| {
                model_index += 1;
                let model = download_for_build(
                    self.downloader,
                    model_id,
                    model_index,
                    model_count,
                    progress.as_ref(),
                )?;
                TextlineOrientationClassifier::from_downloaded(
                    &model,
                    self.textline_orientation_options,
                )
            })
            .transpose()?;
        let mut text_recognition_options = self.text_recognition_options;
        text_recognition_options.return_word_alignment |= self.pipeline_options.return_word_boxes;
        model_index += 1;
        let recognizer_model = download_for_build(
            self.downloader,
            &self.text_recognition_model,
            model_index,
            model_count,
            progress.as_ref(),
        )?;
        let recognizer =
            TextRecognizer::from_downloaded(&recognizer_model, text_recognition_options)?;
        OcrPipeline::from_components(
            detector,
            recognizer,
            document_orientation,
            document_rectifier,
            textline_orientation,
            self.pipeline_options,
        )
    }

    /// Build independent pipelines for concurrent callers.
    ///
    /// Each worker owns its own five-or-fewer ORT sessions. On CPU, cap
    /// `intra_threads` when using several workers to avoid oversubscription. On
    /// a GPU, a single pipeline with larger OCR batches is normally preferable
    /// because every additional worker duplicates session and device memory.
    pub fn build_pool(self, worker_count: usize) -> Result<OcrPipelinePool> {
        if worker_count == 0 {
            return Err(OcrError::InvalidPipelineConfig {
                reason: "worker_count must be greater than zero".to_owned(),
            });
        }
        self.validate()?;
        let mut pipelines = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            pipelines.push(self.clone().build()?);
        }
        OcrPipelinePool::from_pipelines(pipelines)
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

fn download_for_build(
    downloader: &ModelDownloader,
    model_id: &str,
    model_index: usize,
    model_count: usize,
    progress: Option<&PipelineBuildProgressCallback>,
) -> Result<DownloadedModel> {
    let download_progress: Option<ModelDownloadProgressCallback> = progress.map(|progress| {
        let progress = progress.clone();
        Arc::new(move |download: crate::ModelDownloadProgress| {
            progress(PipelineBuildProgress {
                model_index,
                model_count,
                model_id: download.model_id,
                phase: PipelineBuildPhase::Downloading,
                bytes_completed: Some(download.bytes_completed),
                total_bytes: Some(download.total_bytes),
            });
        }) as ModelDownloadProgressCallback
    });
    let model = downloader.download_with_progress(model_id, download_progress)?;
    if let Some(progress) = progress {
        progress(PipelineBuildProgress {
            model_index,
            model_count,
            model_id: model_id.to_owned(),
            phase: PipelineBuildPhase::Initializing,
            bytes_completed: None,
            total_bytes: None,
        });
    }
    Ok(model)
}

/// ONNX Runtime implementation of the PaddleOCR general OCR pipeline.
///
/// CPU is the default. Configure another execution provider through the
/// module options or [`OcrPipelineBuilder::session_options`].
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

/// A set of independent pipelines for safe request-level concurrency.
///
/// `ort::Session::run` requires exclusive access because some provider
/// allocators and statistics are not thread-safe. This pool never runs the
/// same session concurrently: each worker is protected by its own mutex, while
/// separate workers may execute in parallel.
pub struct OcrPipelinePool {
    workers: Box<[Mutex<OcrPipeline>]>,
    next_worker: AtomicUsize,
}

impl std::fmt::Debug for OcrPipelinePool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OcrPipelinePool")
            .field("worker_count", &self.workers.len())
            .finish_non_exhaustive()
    }
}

impl OcrPipelinePool {
    /// Construct a pool from already initialized pipelines.
    ///
    /// This also permits assigning different device IDs to different workers.
    pub fn from_pipelines(pipelines: Vec<OcrPipeline>) -> Result<Self> {
        if pipelines.is_empty() {
            return Err(OcrError::InvalidPipelineConfig {
                reason: "an OCR pipeline pool requires at least one worker".to_owned(),
            });
        }
        Ok(Self {
            workers: pipelines
                .into_iter()
                .map(Mutex::new)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            next_worker: AtomicUsize::new(0),
        })
    }

    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Run one image on a round-robin worker.
    pub fn predict(&self, image: &RgbImage) -> Result<OcrPipelineResult> {
        let index = self.next_worker.fetch_add(1, AtomicOrdering::Relaxed) % self.workers.len();
        let mut worker = self.workers[index]
            .lock()
            .map_err(|_| OcrError::Concurrency {
                reason: format!("pipeline worker {index} is poisoned"),
            })?;
        worker.predict(image)
    }

    /// Distribute images over all workers and restore input order.
    pub fn predict_batch(&self, images: &[RgbImage]) -> Result<Vec<OcrPipelineResult>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        let next_image = AtomicUsize::new(0);
        let results = Mutex::new(
            std::iter::repeat_with(|| None)
                .take(images.len())
                .collect::<Vec<Option<Result<OcrPipelineResult>>>>(),
        );
        std::thread::scope(|scope| {
            for worker in &self.workers {
                let next_image = &next_image;
                let results = &results;
                scope.spawn(move || {
                    let Ok(mut worker) = worker.lock() else {
                        return;
                    };
                    loop {
                        let index = next_image.fetch_add(1, AtomicOrdering::Relaxed);
                        let Some(image) = images.get(index) else {
                            break;
                        };
                        let result = worker.predict(image);
                        results
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)[index] =
                            Some(result);
                    }
                });
            }
        });

        results
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .into_iter()
            .enumerate()
            .map(|(index, result)| {
                result.unwrap_or_else(|| {
                    Err(OcrError::Concurrency {
                        reason: format!("no healthy pipeline worker processed image {index}"),
                    })
                })
            })
            .collect()
    }
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
        if options.return_word_boxes && !recognizer.returns_word_alignment() {
            return Err(OcrError::InvalidPipelineConfig {
                reason: "return_word_boxes requires a recognizer built with return_word_alignment"
                    .to_owned(),
            });
        }
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
            self.options.return_word_boxes,
        );

        Ok(OcrPipelineResult {
            input_size,
            processed_size: OcrImageSize::from(&processed_image),
            processed_image,
            model_settings: self.model_settings,
            model_ids: self.model_ids.clone(),
            text_detection_parameters: self.text_detection_parameters,
            text_recognition_score_threshold: self.options.text_recognition_score_threshold,
            return_word_boxes: self.options.return_word_boxes,
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
    return_word_boxes: bool,
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
                (recognition.score >= score_threshold).then(|| {
                    let words = return_word_boxes.then(|| {
                        word_boxes(
                            &recognition.text,
                            &detection.points,
                            recognition
                                .word_alignment
                                .as_ref()
                                .expect("word alignment is enabled with word boxes"),
                        )
                    });
                    OcrTextLine {
                        detection_index,
                        polygon: detection.points,
                        bounding_box: enclosing_box(&detection.points),
                        detection_score: detection.score,
                        textline_orientation: textline_orientation.clone(),
                        text: recognition.text,
                        recognition_score: recognition.score,
                        words,
                    }
                })
            },
        )
        .collect();
    (detections, textline_orientations, text_lines)
}

fn word_boxes(
    recognized_text: &str,
    line: &[Point; 4],
    alignment: &CtcWordAlignment,
) -> Vec<OcrWord> {
    if alignment.column_count <= 0.0 {
        return Vec::new();
    }
    let width = line
        .iter()
        .map(|point| point.x)
        .fold(f32::NEG_INFINITY, f32::max)
        - line
            .iter()
            .map(|point| point.x)
            .fold(f32::INFINITY, f32::min);
    let height = line
        .iter()
        .map(|point| point.y)
        .fold(f32::NEG_INFINITY, f32::max)
        - line
            .iter()
            .map(|point| point.y)
            .fold(f32::INFINITY, f32::min);
    let vertical = width == 0.0 || height / width > 1.5;
    let (axis_start, axis_end, cross_start, cross_end) = if vertical {
        (line[0].y, line[2].y, line[0].x, line[1].x)
    } else {
        (line[0].x, line[1].x, line[0].y, line[2].y)
    };
    let axis_size = axis_end - axis_start;
    let cell_size = axis_size / alignment.column_count;
    let mut words = Vec::new();
    let mut cjk = Vec::new();
    let mut cjk_sizes = Vec::new();

    for group in &alignment.groups {
        if group.columns.is_empty() {
            continue;
        }
        if group.kind == CtcWordKind::Cjk {
            if group.columns.len() != 1 {
                let sequence_size = (group.columns[group.columns.len() - 1] - group.columns[0] + 1)
                    as f32
                    * cell_size;
                cjk_sizes.push(sequence_size / (group.columns.len() - 1) as f32);
            }
            cjk.extend(group.text.chars().zip(group.columns.iter().copied()));
            continue;
        }

        let start = axis_start + (group.columns[0] as f32 * cell_size) as i32 as f32;
        let end = axis_start
            + ((group.columns[group.columns.len() - 1] + 1) as f32 * cell_size) as i32 as f32;
        words.push(make_word(
            group.text.clone(),
            start,
            end,
            cross_start,
            cross_end,
            vertical,
        ));
    }

    if !cjk.is_empty() {
        let average_size = if cjk_sizes.is_empty() {
            axis_size / recognized_text.chars().count().max(1) as f32
        } else {
            cjk_sizes.iter().sum::<f32>() / cjk_sizes.len() as f32
        };
        for (character, column) in cjk {
            let center = (column as f32 + 0.5) * cell_size;
            let start = axis_start + ((center - average_size / 2.0) as i32).max(0) as f32;
            let end =
                axis_start + ((center + average_size / 2.0) as i32).min(axis_size as i32) as f32;
            words.push(make_word(
                character.to_string(),
                start,
                end,
                cross_start,
                cross_end,
                vertical,
            ));
        }
    }

    words.sort_by(|left, right| {
        word_center(left)
            .1
            .partial_cmp(&word_center(right).1)
            .unwrap_or(Ordering::Equal)
    });
    let mut sorted = Vec::with_capacity(words.len());
    let mut current_line = Vec::new();
    let mut last_y: Option<f32> = None;
    for word in words {
        let y = word_center(&word).1;
        if last_y.is_some_and(|last| (y - last).abs() >= 12.0) {
            append_x_sorted(&mut current_line, &mut sorted);
        }
        current_line.push(word);
        last_y = Some(y);
    }
    append_x_sorted(&mut current_line, &mut sorted);
    sorted
}

fn word_center(word: &OcrWord) -> (f32, f32) {
    (
        (word.bounding_box[0] + word.bounding_box[2]) / 2.0,
        (word.bounding_box[1] + word.bounding_box[3]) / 2.0,
    )
}

fn append_x_sorted(line: &mut Vec<OcrWord>, output: &mut Vec<OcrWord>) {
    line.sort_by(|left, right| {
        word_center(left)
            .0
            .partial_cmp(&word_center(right).0)
            .unwrap_or(Ordering::Equal)
    });
    output.append(line);
}

fn make_word(
    text: String,
    start: f32,
    end: f32,
    cross_start: f32,
    cross_end: f32,
    vertical: bool,
) -> OcrWord {
    let polygon = if vertical {
        [
            Point::new(cross_start, start),
            Point::new(cross_end, start),
            Point::new(cross_end, end),
            Point::new(cross_start, end),
        ]
    } else {
        [
            Point::new(start, cross_start),
            Point::new(end, cross_start),
            Point::new(end, cross_end),
            Point::new(start, cross_end),
        ]
    };
    OcrWord {
        text,
        bounding_box: enclosing_box(&polygon),
        polygon,
    }
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
    fn pool_rejects_zero_workers_before_download() {
        let downloader = downloader();
        let error = OcrPipelineBuilder::new(&downloader)
            .build_pool(0)
            .unwrap_err();
        assert!(matches!(error, OcrError::InvalidPipelineConfig { .. }));
    }

    #[test]
    fn runtime_options_apply_to_every_module() {
        let downloader = downloader();
        let options = crate::OrtSessionOptions {
            intra_threads: Some(2),
            inter_threads: Some(1),
            ..crate::OrtSessionOptions::default()
        };
        let builder = OcrPipelineBuilder::new(&downloader).session_options(options);
        assert_eq!(
            builder.document_orientation_options.session.intra_threads,
            Some(2)
        );
        assert_eq!(
            builder.document_rectification_options.session.intra_threads,
            Some(2)
        );
        assert_eq!(
            builder.textline_orientation_options.session.intra_threads,
            Some(2)
        );
        assert_eq!(
            builder.text_detection_options.session.intra_threads,
            Some(2)
        );
        assert_eq!(
            builder.text_recognition_options.session.intra_threads,
            Some(2)
        );
    }

    #[test]
    fn pipelines_and_pools_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<OcrPipeline>();
        assert_send_sync::<OcrPipelinePool>();
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
                return_word_boxes: false,
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
                ..TextRecognitionResult::default()
            },
            TextRecognitionResult {
                text: "filtered".to_owned(),
                score: 0.79,
                ..TextRecognitionResult::default()
            },
            TextRecognitionResult {
                text: "threshold".to_owned(),
                score: 0.8,
                ..TextRecognitionResult::default()
            },
        ];

        let (all_detections, all_orientations, lines) =
            aggregate_text_lines(detections, orientations, recognitions, 0.8, false);
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
    fn word_boxes_follow_ctc_columns_in_document_coordinates() {
        let line = [
            Point::new(0.0, 0.0),
            Point::new(100.0, 0.0),
            Point::new(100.0, 20.0),
            Point::new(0.0, 20.0),
        ];
        let alignment = CtcWordAlignment {
            column_count: 10.0,
            groups: vec![
                crate::text_recognition::CtcWordGroup {
                    text: "中文".to_owned(),
                    columns: vec![1, 3],
                    kind: CtcWordKind::Cjk,
                },
                crate::text_recognition::CtcWordGroup {
                    text: "A-1".to_owned(),
                    columns: vec![5, 6, 7],
                    kind: CtcWordKind::Alphanumeric,
                },
            ],
        };
        let words = word_boxes("中文A-1", &line, &alignment);
        assert_eq!(
            words
                .iter()
                .map(|word| (word.text.as_str(), word.bounding_box))
                .collect::<Vec<_>>(),
            vec![
                ("中", [0.0, 0.0, 30.0, 20.0]),
                ("文", [20.0, 0.0, 50.0, 20.0]),
                ("A-1", [50.0, 0.0, 80.0, 20.0]),
            ]
        );
    }

    #[test]
    fn word_boxes_support_vertical_text_lines() {
        let line = [
            Point::new(10.0, 0.0),
            Point::new(30.0, 0.0),
            Point::new(30.0, 100.0),
            Point::new(10.0, 100.0),
        ];
        let alignment = CtcWordAlignment {
            column_count: 10.0,
            groups: vec![
                crate::text_recognition::CtcWordGroup {
                    text: "中文".to_owned(),
                    columns: vec![1, 3],
                    kind: CtcWordKind::Cjk,
                },
                crate::text_recognition::CtcWordGroup {
                    text: "A1".to_owned(),
                    columns: vec![5, 6],
                    kind: CtcWordKind::Alphanumeric,
                },
            ],
        };
        let words = word_boxes("中文A1", &line, &alignment);
        assert_eq!(
            words
                .iter()
                .map(|word| (word.text.as_str(), word.bounding_box))
                .collect::<Vec<_>>(),
            vec![
                ("中", [10.0, 0.0, 30.0, 30.0]),
                ("文", [10.0, 20.0, 30.0, 50.0]),
                ("A1", [10.0, 50.0, 30.0, 70.0]),
            ]
        );
        assert_eq!(
            words[0].polygon,
            [
                Point::new(10.0, 0.0),
                Point::new(30.0, 0.0),
                Point::new(30.0, 30.0),
                Point::new(10.0, 30.0),
            ]
        );
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
