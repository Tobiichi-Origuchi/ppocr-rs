mod geometry;
mod postprocess;

use std::path::Path;

use image::RgbImage;
use ort::{session::Session, value::TensorRef};
use serde::{Deserialize, Serialize};

use crate::config::InferenceConfig;
use crate::image_ops::resize_rgb_to_bgr_chw;
use crate::models::{DownloadedModel, ModelDownloader, ModelKind, manifest};
use crate::runtime::{OrtSessionOptions, load_session};
use crate::{OcrError, Result};

pub use geometry::Point;
use postprocess::DbOptions;

/// How the detector constrains an input image before rounding to multiples of 32.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum DetectionSizeLimit {
    Max,
    Min,
    ResizeLong,
}

/// Runtime and PaddleX-compatible detector overrides.
#[derive(Debug, Clone)]
pub struct TextDetectorOptions {
    pub session: OrtSessionOptions,
    pub limit_side_len: Option<usize>,
    pub limit_type: Option<DetectionSizeLimit>,
    pub max_side_limit: usize,
    pub threshold: Option<f32>,
    pub box_threshold: Option<f32>,
    pub unclip_ratio: Option<f32>,
}

impl Default for TextDetectorOptions {
    fn default() -> Self {
        Self {
            session: OrtSessionOptions::default(),
            limit_side_len: None,
            limit_type: None,
            max_side_limit: 4000,
            threshold: None,
            box_threshold: None,
            unclip_ratio: None,
        }
    }
}

/// One quadrilateral produced by DB post-processing, ordered TL, TR, BR, BL.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectedTextRegion {
    pub points: [Point; 4],
    pub score: f32,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TextDetectionResult {
    pub regions: Vec<DetectedTextRegion>,
}

/// PaddleOCR DB text detector backed by ONNX Runtime.
pub struct TextDetector {
    model_id: String,
    session: Session,
    mean: [f32; 3],
    std: [f32; 3],
    scale: f32,
    limit_side_len: usize,
    limit_type: DetectionSizeLimit,
    max_side_limit: usize,
    postprocess: DbOptions,
}

impl std::fmt::Debug for TextDetector {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TextDetector")
            .field("model_id", &self.model_id)
            .field("limit_side_len", &self.limit_side_len)
            .field("limit_type", &self.limit_type)
            .field("max_side_limit", &self.max_side_limit)
            .field("postprocess", &self.postprocess)
            .finish_non_exhaustive()
    }
}

impl TextDetector {
    /// Download only `model_id` and initialize its detector.
    pub fn from_model(
        model_id: &str,
        downloader: &ModelDownloader,
        options: TextDetectorOptions,
    ) -> Result<Self> {
        let downloaded = downloader.download(model_id)?;
        Self::from_downloaded(&downloaded, options)
    }

    pub fn from_downloaded(model: &DownloadedModel, options: TextDetectorOptions) -> Result<Self> {
        Self::from_files(
            &model.model_id,
            &model.onnx_path,
            &model.config_path,
            options,
        )
    }

    pub fn from_files(
        model_id: &str,
        onnx_path: &Path,
        config_path: &Path,
        options: TextDetectorOptions,
    ) -> Result<Self> {
        require_kind(model_id, ModelKind::TextDetection)?;
        let config = InferenceConfig::read(config_path)?;
        validate_common_config(model_id, &config)?;
        if config.post_process.name != "DBPostProcess" {
            return invalid_config(model_id, "PostProcess.name must be DBPostProcess");
        }
        if config.post_process.score_mode.as_deref().unwrap_or("fast") != "fast" {
            return invalid_config(model_id, "only PaddleX DB score_mode=fast is supported");
        }
        if config.post_process.box_type.as_deref().unwrap_or("quad") != "quad" {
            return invalid_config(model_id, "only PaddleX DB box_type=quad is supported");
        }
        let normalization = config
            .normalization()
            .ok_or_else(|| invalid_config_error(model_id, "NormalizeImage is missing"))?;
        let mean = three_values(model_id, "NormalizeImage.mean", &normalization.mean)?;
        let std = three_values(model_id, "NormalizeImage.std", &normalization.std)?;
        if !normalization.order.eq_ignore_ascii_case("hwc") {
            return invalid_config(model_id, "NormalizeImage.order must be hwc");
        }
        if options.max_side_limit < 32 {
            return invalid_config(model_id, "max_side_limit must be at least 32");
        }

        let resize = config.detection_resize();
        let limit_side_len = options
            .limit_side_len
            .or_else(|| resize.and_then(|value| value.resize_long))
            .unwrap_or(960);
        let limit_type = options.limit_type.unwrap_or_else(|| {
            resize
                .and_then(|value| value.limit_type.as_deref())
                .and_then(parse_limit_type)
                .unwrap_or(DetectionSizeLimit::Max)
        });
        let session = load_session(onnx_path, &options.session)?;
        validate_session(model_id, &session)?;

        Ok(Self {
            model_id: model_id.to_owned(),
            session,
            mean,
            std,
            scale: normalization.scale.0,
            limit_side_len,
            limit_type,
            max_side_limit: options.max_side_limit,
            postprocess: DbOptions {
                threshold: options
                    .threshold
                    .or(config.post_process.thresh)
                    .unwrap_or(0.3),
                box_threshold: options
                    .box_threshold
                    .or(config.post_process.box_thresh)
                    .unwrap_or(0.6),
                max_candidates: config.post_process.max_candidates.unwrap_or(1000),
                unclip_ratio: options
                    .unclip_ratio
                    .or(config.post_process.unclip_ratio)
                    .unwrap_or(2.0),
                use_dilation: config.post_process.use_dilation.unwrap_or(false),
            },
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn detect(&mut self, image: &RgbImage) -> Result<TextDetectionResult> {
        if image.width() == 0 || image.height() == 0 {
            return Err(OcrError::EmptyImage);
        }
        let (source_width, source_height) = if image.width() + image.height() < 64 {
            (image.width().max(32), image.height().max(32))
        } else {
            (image.width(), image.height())
        };
        let (resize_width, resize_height) = self.resized_dimensions(source_width, source_height);
        let data = resize_rgb_to_bgr_chw(
            image,
            source_width,
            source_height,
            resize_width,
            resize_height,
            |channel, value| (value * self.scale - self.mean[channel]) / self.std[channel],
        );
        let input = TensorRef::from_array_view((
            [1usize, 3, resize_height, resize_width],
            data.as_slice(),
        ))?;
        let outputs = self.session.run(ort::inputs![input])?;
        let output = outputs[0].try_extract_array::<f32>()?;
        if output.ndim() != 4 || output.shape()[0] != 1 || output.shape()[1] != 1 {
            return Err(OcrError::InvalidOutput {
                model_id: self.model_id.clone(),
                reason: format!("expected [1, 1, H, W], received {:?}", output.shape()),
            });
        }
        let batch_output = output.index_axis(ndarray::Axis(0), 0);
        let prediction = batch_output
            .index_axis(ndarray::Axis(0), 0)
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| OcrError::InvalidOutput {
                model_id: self.model_id.clone(),
                reason: "failed to view detector output as a 2-D map".to_owned(),
            })?;
        Ok(postprocess::process(
            prediction,
            image.width(),
            image.height(),
            &self.postprocess,
        ))
    }

    /// Detection images generally have different tensor shapes, so a batch is run sequentially.
    pub fn detect_batch(&mut self, images: &[RgbImage]) -> Result<Vec<TextDetectionResult>> {
        images.iter().map(|image| self.detect(image)).collect()
    }

    fn resized_dimensions(&self, width: u32, height: u32) -> (usize, usize) {
        let width = width as usize;
        let height = height as usize;
        let ratio = match self.limit_type {
            DetectionSizeLimit::Max if width.max(height) > self.limit_side_len => {
                self.limit_side_len as f64 / width.max(height) as f64
            }
            DetectionSizeLimit::Min if width.min(height) < self.limit_side_len => {
                self.limit_side_len as f64 / width.min(height) as f64
            }
            DetectionSizeLimit::ResizeLong => self.limit_side_len as f64 / width.max(height) as f64,
            _ => 1.0,
        };
        let mut resize_width = (width as f64 * ratio) as usize;
        let mut resize_height = (height as f64 * ratio) as usize;
        if resize_width.max(resize_height) > self.max_side_limit {
            let ratio = self.max_side_limit as f64 / resize_width.max(resize_height) as f64;
            resize_width = (resize_width as f64 * ratio) as usize;
            resize_height = (resize_height as f64 * ratio) as usize;
        }
        (
            round_to_stride(resize_width, 32),
            round_to_stride(resize_height, 32),
        )
    }
}

fn validate_common_config(model_id: &str, config: &InferenceConfig) -> Result<()> {
    if config.global.model_name != model_id {
        return invalid_config(
            model_id,
            "Global.model_name does not match the selected model",
        );
    }
    let decode = config
        .decode_image()
        .ok_or_else(|| invalid_config_error(model_id, "DecodeImage is missing"))?;
    if decode.channel_first || !decode.img_mode.eq_ignore_ascii_case("BGR") {
        return invalid_config(model_id, "DecodeImage must use channel_first=false and BGR");
    }
    Ok(())
}

fn validate_session(model_id: &str, session: &Session) -> Result<()> {
    if session.inputs().len() != 1 || session.outputs().len() != 1 {
        return invalid_config(
            model_id,
            "text detection ONNX must have exactly one input and one output",
        );
    }
    Ok(())
}

fn require_kind(model_id: &str, expected: ModelKind) -> Result<()> {
    let spec = manifest()
        .get(model_id)
        .ok_or_else(|| crate::models::ModelDownloadError::UnknownModel(model_id.to_owned()))?;
    if spec.kind != expected {
        return Err(OcrError::WrongModelKind {
            model_id: model_id.to_owned(),
            actual: spec.kind,
            expected,
        });
    }
    Ok(())
}

fn three_values(model_id: &str, name: &str, values: &[f32]) -> Result<[f32; 3]> {
    values.try_into().map_err(|_| {
        invalid_config_error(
            model_id,
            &format!("{name} must contain exactly three values"),
        )
    })
}

fn parse_limit_type(value: &str) -> Option<DetectionSizeLimit> {
    match value {
        "max" => Some(DetectionSizeLimit::Max),
        "min" => Some(DetectionSizeLimit::Min),
        "resize_long" => Some(DetectionSizeLimit::ResizeLong),
        _ => None,
    }
}

fn round_to_stride(value: usize, stride: usize) -> usize {
    (((value as f64 / stride as f64).round_ties_even() as usize) * stride).max(stride)
}

fn invalid_config<T>(model_id: &str, reason: &str) -> Result<T> {
    Err(invalid_config_error(model_id, reason))
}

fn invalid_config_error(model_id: &str, reason: &str) -> OcrError {
    OcrError::InvalidConfig {
        model_id: model_id.to_owned(),
        reason: reason.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounds_dimensions_like_python() {
        assert_eq!(round_to_stride(48, 32), 64);
        assert_eq!(round_to_stride(80, 32), 64);
        assert_eq!(round_to_stride(15, 32), 32);
    }
}
