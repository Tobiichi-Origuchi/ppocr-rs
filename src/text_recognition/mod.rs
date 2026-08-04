use std::path::Path;

use image::RgbImage;
use ort::{session::Session, value::TensorRef};
use serde::{Deserialize, Serialize};
use unicode_bidi::BidiInfo;

use crate::config::InferenceConfig;
use crate::image_ops::resize_rgb_to_bgr_chw;
use crate::models::{DownloadedModel, ModelDownloader, ModelKind, manifest};
use crate::runtime::{OrtSessionOptions, load_session};
use crate::{OcrError, Result};

const MAX_RECOGNITION_WIDTH: usize = 3200;

#[derive(Debug, Clone, Default)]
pub struct TextRecognizerOptions {
    pub session: OrtSessionOptions,
    /// Preserve CTC timestep alignment for pipeline-level word boxes.
    ///
    /// The standalone result continues to expose only text and confidence;
    /// document-coordinate boxes require the detector polygon and are produced
    /// by [`crate::OcrPipeline`].
    pub return_word_alignment: bool,
}

/// Text and mean CTC confidence for one input crop.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TextRecognitionResult {
    pub text: String,
    pub score: f32,
    #[serde(skip)]
    pub(crate) word_alignment: Option<CtcWordAlignment>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CtcWordAlignment {
    pub column_count: f32,
    pub groups: Vec<CtcWordGroup>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CtcWordGroup {
    pub text: String,
    pub columns: Vec<usize>,
    pub kind: CtcWordKind,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum CtcWordKind {
    Cjk,
    Alphanumeric,
    Symbol,
}

/// PaddleOCR text recognizer with config-defined CTC vocabulary.
pub struct TextRecognizer {
    model_id: String,
    session: Session,
    image_height: usize,
    base_image_width: usize,
    vocabulary: Vec<String>,
    apply_arabic_bidi: bool,
    return_word_alignment: bool,
}

impl std::fmt::Debug for TextRecognizer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TextRecognizer")
            .field("model_id", &self.model_id)
            .field("image_height", &self.image_height)
            .field("base_image_width", &self.base_image_width)
            .field("vocabulary_size", &self.vocabulary.len())
            .field("apply_arabic_bidi", &self.apply_arabic_bidi)
            .finish_non_exhaustive()
    }
}

impl TextRecognizer {
    /// Download only `model_id` and initialize its recognizer.
    pub fn from_model(
        model_id: &str,
        downloader: &ModelDownloader,
        options: TextRecognizerOptions,
    ) -> Result<Self> {
        let downloaded = downloader.download(model_id)?;
        Self::from_downloaded(&downloaded, options)
    }

    pub fn from_downloaded(
        model: &DownloadedModel,
        options: TextRecognizerOptions,
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
        options: TextRecognizerOptions,
    ) -> Result<Self> {
        require_kind(model_id, ModelKind::TextRecognition)?;
        let config = InferenceConfig::read(config_path)?;
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
        if config.post_process.name != "CTCLabelDecode" {
            return invalid_config(model_id, "PostProcess.name must be CTCLabelDecode");
        }
        let [channels, image_height, base_image_width] = config
            .recognition_shape()
            .ok_or_else(|| invalid_config_error(model_id, "RecResizeImg is missing"))?;
        if channels != 3 || image_height == 0 || base_image_width == 0 {
            return invalid_config(
                model_id,
                "RecResizeImg.image_shape must be [3, positive height, positive width]",
            );
        }
        let characters = config
            .post_process
            .character_dict
            .ok_or_else(|| invalid_config_error(model_id, "character_dict is missing"))?;
        let mut vocabulary = Vec::with_capacity(characters.len() + 2);
        vocabulary.push("blank".to_owned());
        vocabulary.extend(characters);
        if config.post_process.use_space_char.unwrap_or(true) {
            vocabulary.push(" ".to_owned());
        }

        let session = load_session(onnx_path, &options.session)?;
        if session.inputs().len() != 1 || session.outputs().len() != 1 {
            return invalid_config(
                model_id,
                "text recognition ONNX must have exactly one input and one output",
            );
        }

        Ok(Self {
            model_id: model_id.to_owned(),
            session,
            image_height,
            base_image_width,
            vocabulary,
            apply_arabic_bidi: matches!(
                model_id,
                "arabic_PP-OCRv3_mobile_rec" | "arabic_PP-OCRv5_mobile_rec"
            ),
            return_word_alignment: options.return_word_alignment,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub(crate) fn returns_word_alignment(&self) -> bool {
        self.return_word_alignment
    }

    pub fn recognize(&mut self, image: &RgbImage) -> Result<TextRecognitionResult> {
        let mut results = self.recognize_batch(std::slice::from_ref(image))?;
        Ok(results.pop().expect("a one-image batch has one result"))
    }

    /// Recognize a batch, padding only this batch to its widest selected crop.
    pub fn recognize_batch(&mut self, images: &[RgbImage]) -> Result<Vec<TextRecognitionResult>> {
        let images: Vec<&RgbImage> = images.iter().collect();
        self.recognize_batch_refs(&images)
    }

    /// Recognize borrowed crops without cloning them when their order is indirect.
    pub fn recognize_batch_refs(
        &mut self,
        images: &[&RgbImage],
    ) -> Result<Vec<TextRecognitionResult>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        if images
            .iter()
            .any(|image| image.width() == 0 || image.height() == 0)
        {
            return Err(OcrError::EmptyImage);
        }

        let padded_widths: Vec<usize> = images
            .iter()
            .map(|image| self.padded_width(image))
            .collect();
        let batch_width = *padded_widths
            .iter()
            .max()
            .expect("non-empty image batch has a maximum width");
        let sample_len = 3 * self.image_height * batch_width;
        let mut batch = vec![0.0f32; sample_len * images.len()];

        for (batch_index, &image) in images.iter().enumerate() {
            let resized_width = self.resized_width(image, padded_widths[batch_index]);
            let resized = resize_rgb_to_bgr_chw(
                image,
                image.width(),
                image.height(),
                resized_width,
                self.image_height,
                |_, value| value / 127.5 - 1.0,
            );
            let source_plane = resized_width * self.image_height;
            let destination_plane = batch_width * self.image_height;
            let sample_start = batch_index * sample_len;
            for channel in 0..3 {
                for row in 0..self.image_height {
                    let source_start = channel * source_plane + row * resized_width;
                    let destination_start =
                        sample_start + channel * destination_plane + row * batch_width;
                    batch[destination_start..destination_start + resized_width]
                        .copy_from_slice(&resized[source_start..source_start + resized_width]);
                }
            }
        }

        let input = TensorRef::from_array_view((
            [images.len(), 3, self.image_height, batch_width],
            batch.as_slice(),
        ))?;
        let outputs = self.session.run(ort::inputs![input])?;
        let output = outputs[0].try_extract_array::<f32>()?;
        if output.ndim() != 3 || output.shape()[0] != images.len() {
            return Err(OcrError::InvalidOutput {
                model_id: self.model_id.clone(),
                reason: format!(
                    "expected [batch, time, classes], received {:?}",
                    output.shape()
                ),
            });
        }
        if output.shape()[2] != self.vocabulary.len() {
            return Err(OcrError::InvalidOutput {
                model_id: self.model_id.clone(),
                reason: format!(
                    "output has {} CTC classes but inference.yml defines {}",
                    output.shape()[2],
                    self.vocabulary.len()
                ),
            });
        }

        let mut results = Vec::with_capacity(images.len());
        let max_wh_ratio = images.iter().fold(
            self.base_image_width as f64 / self.image_height as f64,
            |maximum, image| maximum.max(f64::from(image.width()) / f64::from(image.height())),
        );
        for (batch_index, image) in images.iter().enumerate() {
            let sample = output.index_axis(ndarray::Axis(0), batch_index);
            let mut previous = None;
            let mut text = String::new();
            let mut score_sum = 0.0;
            let mut character_count = 0usize;
            let mut decoded = Vec::new();
            for (column, timestep) in sample.outer_iter().enumerate() {
                let mut best_index = 0usize;
                let mut best_score = f32::NEG_INFINITY;
                for (index, score) in timestep.iter().copied().enumerate() {
                    if score > best_score {
                        best_index = index;
                        best_score = score;
                    }
                }
                let duplicate = previous == Some(best_index);
                previous = Some(best_index);
                if best_index != 0 && !duplicate {
                    text.push_str(&self.vocabulary[best_index]);
                    if self.return_word_alignment {
                        decoded.push((self.vocabulary[best_index].clone(), column));
                    }
                    score_sum += best_score;
                    character_count += 1;
                }
            }
            let word_alignment = self.return_word_alignment.then(|| {
                build_word_alignment(
                    &decoded,
                    sample.shape()[0],
                    f64::from(image.width()) / f64::from(image.height()),
                    max_wh_ratio,
                )
            });
            if self.apply_arabic_bidi {
                text = bidi_display(&text);
            }
            results.push(TextRecognitionResult {
                text,
                score: if character_count == 0 {
                    0.0
                } else {
                    score_sum / character_count as f32
                },
                word_alignment,
            });
        }
        Ok(results)
    }

    fn padded_width(&self, image: &RgbImage) -> usize {
        padded_width_for(self.image_height, self.base_image_width, image)
    }

    fn resized_width(&self, image: &RgbImage, padded_width: usize) -> usize {
        let aspect_width = (self.image_height as f64 * image.width() as f64 / image.height() as f64)
            .ceil() as usize;
        aspect_width.min(padded_width).max(1)
    }
}

fn build_word_alignment(
    decoded: &[(String, usize)],
    output_columns: usize,
    image_wh_ratio: f64,
    max_wh_ratio: f64,
) -> CtcWordAlignment {
    let mut groups: Vec<CtcWordGroup> = Vec::new();
    let mut current_kind = None;
    for (index, (token, column)) in decoded.iter().enumerate() {
        let mut kind = token_kind(token);
        if token == "."
            && current_kind == Some(CtcWordKind::Alphanumeric)
            && decoded
                .get(index + 1)
                .and_then(|(next, _)| next.chars().next())
                .is_some_and(|character| character.is_ascii_digit())
        {
            kind = CtcWordKind::Alphanumeric;
        }
        if token == "-" && current_kind == Some(CtcWordKind::Alphanumeric) {
            kind = CtcWordKind::Alphanumeric;
        }

        if current_kind != Some(kind) {
            groups.push(CtcWordGroup {
                text: String::new(),
                columns: Vec::new(),
                kind,
            });
            current_kind = Some(kind);
        }
        let group = groups
            .last_mut()
            .expect("a group is created before appending a decoded token");
        group.text.push_str(token);
        group.columns.push(*column);
    }

    CtcWordAlignment {
        column_count: output_columns as f32 * (image_wh_ratio / max_wh_ratio) as f32,
        groups,
    }
}

fn token_kind(token: &str) -> CtcWordKind {
    match token.chars().next() {
        Some(character) if ('\u{4e00}'..='\u{9fff}').contains(&character) => CtcWordKind::Cjk,
        Some(character) if character.is_ascii_alphanumeric() => CtcWordKind::Alphanumeric,
        _ => CtcWordKind::Symbol,
    }
}

fn padded_width_for(image_height: usize, base_image_width: usize, image: &RgbImage) -> usize {
    let aspect_width =
        (image_height as f64 * image.width() as f64 / image.height() as f64).floor() as usize;
    base_image_width
        .max(aspect_width)
        .min(MAX_RECOGNITION_WIDTH)
}

fn bidi_display(text: &str) -> String {
    let bidi = BidiInfo::new(text, None);
    let mut display = String::with_capacity(text.len());
    for paragraph in &bidi.paragraphs {
        display.push_str(&bidi.reorder_line(paragraph, paragraph.range.clone()));
    }
    display
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bidi_reorders_arabic_for_display() {
        assert_eq!(bidi_display("مرحبا 123"), "123 ابحرم");
    }

    #[test]
    fn batch_width_uses_paddlex_floor_before_resize_ceil() {
        let image = RgbImage::new(514, 64);
        assert_eq!(padded_width_for(48, 320, &image), 385);

        let narrow = RgbImage::new(100, 100);
        assert_eq!(padded_width_for(48, 320, &narrow), 320);
    }

    #[test]
    fn word_alignment_groups_paddlex_ctc_columns() {
        let decoded = vec![
            ("A".to_owned(), 1),
            ("-".to_owned(), 2),
            ("1".to_owned(), 3),
            (".".to_owned(), 4),
            ("5".to_owned(), 5),
            (" ".to_owned(), 6),
            ("中".to_owned(), 8),
            ("文".to_owned(), 10),
        ];
        let alignment = build_word_alignment(&decoded, 20, 2.0, 4.0);
        assert_eq!(alignment.column_count, 10.0);
        assert_eq!(alignment.groups.len(), 3);
        assert_eq!(alignment.groups[0].text, "A-1.5");
        assert_eq!(alignment.groups[0].columns, [1, 2, 3, 4, 5]);
        assert_eq!(alignment.groups[0].kind, CtcWordKind::Alphanumeric);
        assert_eq!(alignment.groups[1].kind, CtcWordKind::Symbol);
        assert_eq!(alignment.groups[2].text, "中文");
        assert_eq!(alignment.groups[2].kind, CtcWordKind::Cjk);
    }
}
