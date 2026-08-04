use std::path::Path;

use image::RgbImage;
use ort::{session::Session, value::TensorRef};

use crate::config::{InferenceConfig, NormalizeImageConfig};
use crate::image_ops::{resize_rgb, rgb_region_to_chw};
use crate::models::{DownloadedModel, ModelDownloader, ModelKind, manifest};
use crate::runtime::{OrtSessionOptions, load_session};
use crate::{OcrError, Result};

#[derive(Debug, Clone)]
pub(crate) struct ClassifierOptions {
    pub session: OrtSessionOptions,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ClassificationResult {
    pub class_id: usize,
    pub label: String,
    pub score: f32,
}

#[derive(Debug, Clone, Copy)]
enum ResizeMode {
    Exact { width: usize, height: usize },
    ShortEdge(usize),
}

pub(crate) struct ImageClassifier {
    model_id: String,
    session: Session,
    resize: ResizeMode,
    crop: Option<[usize; 2]>,
    normalization: Normalization,
    labels: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
struct Normalization {
    mean: [f32; 3],
    std: [f32; 3],
    scale: f32,
}

impl std::fmt::Debug for ImageClassifier {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ImageClassifier")
            .field("model_id", &self.model_id)
            .field("resize", &self.resize)
            .field("crop", &self.crop)
            .field("labels", &self.labels)
            .finish_non_exhaustive()
    }
}

impl ImageClassifier {
    pub fn from_model(
        model_id: &str,
        expected_kind: ModelKind,
        downloader: &ModelDownloader,
        options: ClassifierOptions,
    ) -> Result<Self> {
        let downloaded = downloader.download(model_id)?;
        Self::from_downloaded(&downloaded, expected_kind, options)
    }

    pub fn from_downloaded(
        model: &DownloadedModel,
        expected_kind: ModelKind,
        options: ClassifierOptions,
    ) -> Result<Self> {
        Self::from_files(
            &model.model_id,
            expected_kind,
            &model.onnx_path,
            &model.config_path,
            options,
        )
    }

    pub fn from_files(
        model_id: &str,
        expected_kind: ModelKind,
        onnx_path: &Path,
        config_path: &Path,
        options: ClassifierOptions,
    ) -> Result<Self> {
        require_kind(model_id, expected_kind)?;
        let config = InferenceConfig::read(config_path)?;
        if config.global.model_name != model_id {
            return invalid_config(
                model_id,
                "Global.model_name does not match the selected model",
            );
        }

        let resize_config = config
            .classification_resize()
            .ok_or_else(|| invalid_config_error(model_id, "ResizeImage is missing"))?;
        if !resize_config.backend.eq_ignore_ascii_case("cv2")
            || !resize_config.interpolation.eq_ignore_ascii_case("LINEAR")
        {
            return invalid_config(model_id, "only cv2 LINEAR ResizeImage is supported");
        }
        let resize = match (resize_config.resize_short, resize_config.size) {
            (Some(short_edge), None) if short_edge > 0 => ResizeMode::ShortEdge(short_edge),
            (None, Some(size)) => {
                let [width, height] = size.width_height();
                if width == 0 || height == 0 {
                    return invalid_config(model_id, "ResizeImage size must be positive");
                }
                ResizeMode::Exact { width, height }
            }
            _ => {
                return invalid_config(
                    model_id,
                    "ResizeImage must define exactly one of resize_short or size",
                );
            }
        };
        let crop = config
            .classification_crop()
            .map(|config| config.size.width_height());
        if crop.is_some_and(|[width, height]| width == 0 || height == 0) {
            return invalid_config(model_id, "CropImage size must be positive");
        }

        let normalization = parse_normalization(
            model_id,
            config
                .normalization()
                .ok_or_else(|| invalid_config_error(model_id, "NormalizeImage is missing"))?,
        )?;
        let topk = config
            .post_process
            .topk
            .ok_or_else(|| invalid_config_error(model_id, "PostProcess.Topk is missing"))?;
        if topk.topk != 1 {
            return invalid_config(model_id, "PostProcess.Topk.topk must be 1");
        }
        let labels = topk.label_list.ok_or_else(|| {
            invalid_config_error(model_id, "PostProcess.Topk.label_list is missing")
        })?;
        if labels.is_empty() {
            return invalid_config(model_id, "PostProcess.Topk.label_list must not be empty");
        }

        let session = load_session(onnx_path, &options.session)?;
        if session.inputs().len() != 1 || session.outputs().len() != 1 {
            return invalid_config(
                model_id,
                "classification ONNX must have exactly one input and one output",
            );
        }

        Ok(Self {
            model_id: model_id.to_owned(),
            session,
            resize,
            crop,
            normalization,
            labels,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn classify(&mut self, image: &RgbImage) -> Result<ClassificationResult> {
        let mut results = self.classify_batch(std::slice::from_ref(image))?;
        Ok(results.pop().expect("a one-image batch has one result"))
    }

    pub fn classify_batch(&mut self, images: &[RgbImage]) -> Result<Vec<ClassificationResult>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        if images
            .iter()
            .any(|image| image.width() == 0 || image.height() == 0)
        {
            return Err(OcrError::EmptyImage);
        }

        let (input_width, input_height) = self.output_size(&images[0])?;
        let sample_len = 3 * input_width * input_height;
        let mut batch = Vec::with_capacity(sample_len * images.len());
        for image in images {
            let (width, height) = self.output_size(image)?;
            if (width, height) != (input_width, input_height) {
                return Err(OcrError::InvalidConfig {
                    model_id: self.model_id.clone(),
                    reason: "classification preprocessing produced inconsistent batch shapes"
                        .to_owned(),
                });
            }
            batch.extend(self.preprocess(image)?);
        }

        let input = TensorRef::from_array_view((
            [images.len(), 3, input_height, input_width],
            batch.as_slice(),
        ))?;
        let outputs = self.session.run(ort::inputs![input])?;
        let output = outputs[0].try_extract_array::<f32>()?;
        if output.ndim() != 2
            || output.shape()[0] != images.len()
            || output.shape()[1] != self.labels.len()
        {
            return Err(OcrError::InvalidOutput {
                model_id: self.model_id.clone(),
                reason: format!(
                    "expected [batch, {}] classification scores, received {:?}",
                    self.labels.len(),
                    output.shape()
                ),
            });
        }

        let mut results = Vec::with_capacity(images.len());
        for scores in output.outer_iter() {
            let (class_id, score) = scores
                .iter()
                .copied()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .expect("validated classification output has at least one class");
            results.push(ClassificationResult {
                class_id,
                label: self.labels[class_id].clone(),
                // PaddleX Topk applies np.around(..., decimals=5).
                score: (score * 100_000.0).round_ties_even() / 100_000.0,
            });
        }
        Ok(results)
    }

    fn resized_size(&self, image: &RgbImage) -> (usize, usize) {
        match self.resize {
            ResizeMode::Exact { width, height } => (width, height),
            ResizeMode::ShortEdge(target) => {
                let width = image.width() as usize;
                let height = image.height() as usize;
                let scale = target as f64 / width.min(height) as f64;
                (
                    (width as f64 * scale).round_ties_even().max(1.0) as usize,
                    (height as f64 * scale).round_ties_even().max(1.0) as usize,
                )
            }
        }
    }

    fn output_size(&self, image: &RgbImage) -> Result<(usize, usize)> {
        let (resized_width, resized_height) = self.resized_size(image);
        let [width, height] = self.crop.unwrap_or([resized_width, resized_height]);
        if width > resized_width || height > resized_height {
            return Err(OcrError::InvalidConfig {
                model_id: self.model_id.clone(),
                reason: format!(
                    "input resized to {resized_width}x{resized_height}, smaller than the configured {width}x{height} crop"
                ),
            });
        }
        Ok((width, height))
    }

    fn preprocess(&self, image: &RgbImage) -> Result<Vec<f32>> {
        let (resized_width, resized_height) = self.resized_size(image);
        let resized = resize_rgb(image, resized_width, resized_height);
        let (width, height) = self.output_size(image)?;
        let x = (resized_width - width) / 2;
        let y = (resized_height - height) / 2;
        let normalization = self.normalization;
        Ok(rgb_region_to_chw(
            &resized,
            x,
            y,
            width,
            height,
            move |channel, value| {
                (value * normalization.scale - normalization.mean[channel])
                    / normalization.std[channel]
            },
        ))
    }
}

fn parse_normalization(model_id: &str, config: &NormalizeImageConfig) -> Result<Normalization> {
    if config.mean.len() != 3 || config.std.len() != 3 {
        return invalid_config(model_id, "NormalizeImage mean and std must have 3 values");
    }
    if config.std.contains(&0.0) {
        return invalid_config(model_id, "NormalizeImage std values must be non-zero");
    }
    if !config.order.is_empty() {
        return invalid_config(model_id, "NormalizeImage order must be empty");
    }
    if config.channel_num.is_some_and(|channels| channels != 3) {
        return invalid_config(model_id, "NormalizeImage channel_num must be 3");
    }
    Ok(Normalization {
        mean: [config.mean[0], config.mean[1], config.mean[2]],
        std: [config.std[0], config.std[1], config.std[2]],
        scale: config.scale.0,
    })
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

fn invalid_config<T>(model_id: &str, reason: &str) -> Result<T> {
    Err(invalid_config_error(model_id, reason))
}

fn invalid_config_error(model_id: &str, reason: &str) -> OcrError {
    OcrError::InvalidConfig {
        model_id: model_id.to_owned(),
        reason: reason.to_owned(),
    }
}
