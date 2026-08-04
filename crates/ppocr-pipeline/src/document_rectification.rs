use std::path::Path;

use image::{Rgb, RgbImage};
use ort::{session::Session, value::TensorRef};

use crate::config::ModelIdentityConfig;
use crate::image_ops::resize_rgb_to_bgr_chw;
use crate::models::{DownloadedModel, ModelDownloader, ModelKind, manifest};
use crate::runtime::{OrtSessionOptions, load_session};
use crate::{OcrError, Result};

#[derive(Debug, Clone, Default)]
pub struct DocumentRectifierOptions {
    pub session: OrtSessionOptions,
}

/// UVDoc text-image rectification with the official ONNX model.
pub struct DocumentRectifier {
    model_id: String,
    session: Session,
}

impl std::fmt::Debug for DocumentRectifier {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DocumentRectifier")
            .field("model_id", &self.model_id)
            .finish_non_exhaustive()
    }
}

impl DocumentRectifier {
    /// Download only `model_id` and initialize the rectifier.
    pub fn from_model(
        model_id: &str,
        downloader: &ModelDownloader,
        options: DocumentRectifierOptions,
    ) -> Result<Self> {
        let downloaded = downloader.download(model_id)?;
        Self::from_downloaded(&downloaded, options)
    }

    pub fn from_downloaded(
        model: &DownloadedModel,
        options: DocumentRectifierOptions,
    ) -> Result<Self> {
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
        options: DocumentRectifierOptions,
    ) -> Result<Self> {
        require_kind(model_id)?;
        let config = ModelIdentityConfig::read(config_path)?;
        if config.global.model_name != model_id {
            return Err(OcrError::InvalidConfig {
                model_id: model_id.to_owned(),
                reason: "Global.model_name does not match the selected model".to_owned(),
            });
        }
        let session = load_session(onnx_path, &options.session)?;
        if session.inputs().len() != 1 || session.outputs().len() != 1 {
            return Err(OcrError::InvalidConfig {
                model_id: model_id.to_owned(),
                reason: "document rectification ONNX must have exactly one input and one output"
                    .to_owned(),
            });
        }
        Ok(Self {
            model_id: model_id.to_owned(),
            session,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Rectify one image while preserving its width and height.
    pub fn rectify(&mut self, image: &RgbImage) -> Result<RgbImage> {
        if image.width() == 0 || image.height() == 0 {
            return Err(OcrError::EmptyImage);
        }
        let width = image.width() as usize;
        let height = image.height() as usize;
        let batch = resize_rgb_to_bgr_chw(
            image,
            image.width(),
            image.height(),
            width,
            height,
            |_, value| value / 255.0,
        );
        let input = TensorRef::from_array_view(([1, 3, height, width], batch.as_slice()))?;
        let outputs = self.session.run(ort::inputs![input])?;
        let output = outputs[0].try_extract_array::<f32>()?;
        if output.ndim() != 4 || output.shape() != [1, 3, height, width] {
            return Err(OcrError::InvalidOutput {
                model_id: self.model_id.clone(),
                reason: format!(
                    "expected rectified image [1, 3, {height}, {width}], received {:?}",
                    output.shape()
                ),
            });
        }

        let mut rectified = RgbImage::new(image.width(), image.height());
        for y in 0..height {
            for x in 0..width {
                // UVDoc operates on PaddleX BGR input. Its postprocessor scales,
                // truncates to u8, then reverses the channels back to RGB.
                let red = (output[[0, 2, y, x]] * 255.0) as u8;
                let green = (output[[0, 1, y, x]] * 255.0) as u8;
                let blue = (output[[0, 0, y, x]] * 255.0) as u8;
                rectified.put_pixel(x as u32, y as u32, Rgb([red, green, blue]));
            }
        }
        Ok(rectified)
    }

    /// Rectify images of arbitrary dimensions in input order.
    ///
    /// UVDoc's tensor dimensions are dynamic, so differently-sized images are
    /// deliberately run one at a time instead of being padded or resized.
    pub fn rectify_batch(&mut self, images: &[RgbImage]) -> Result<Vec<RgbImage>> {
        images.iter().map(|image| self.rectify(image)).collect()
    }
}

fn require_kind(model_id: &str) -> Result<()> {
    let spec = manifest()
        .get(model_id)
        .ok_or_else(|| crate::models::ModelDownloadError::UnknownModel(model_id.to_owned()))?;
    if spec.kind != ModelKind::DocumentRectification {
        return Err(OcrError::WrongModelKind {
            model_id: model_id.to_owned(),
            actual: spec.kind,
            expected: ModelKind::DocumentRectification,
        });
    }
    Ok(())
}
