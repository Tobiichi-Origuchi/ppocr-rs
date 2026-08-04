use std::fs::File;
use std::path::Path;

use serde::Deserialize;

use crate::{OcrError, Result};

#[derive(Debug, Deserialize)]
pub(crate) struct InferenceConfig {
    #[serde(rename = "Global")]
    pub global: GlobalConfig,
    #[serde(rename = "PreProcess")]
    pub pre_process: PreProcessConfig,
    #[serde(rename = "PostProcess")]
    pub post_process: PostProcessConfig,
}

impl InferenceConfig {
    pub fn read(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|source| OcrError::ReadConfig {
            path: path.to_owned(),
            source,
        })?;
        serde_saphyr::from_reader(file).map_err(|source| OcrError::ParseConfig {
            path: path.to_owned(),
            source: Box::new(source),
        })
    }

    pub fn decode_image(&self) -> Option<&DecodeImageConfig> {
        self.pre_process
            .transform_ops
            .iter()
            .find_map(|op| match op {
                TransformOp::DecodeImage(config) => Some(config),
                _ => None,
            })
    }

    pub fn detection_resize(&self) -> Option<&DetResizeConfig> {
        self.pre_process
            .transform_ops
            .iter()
            .find_map(|op| match op {
                TransformOp::DetResizeForTest(config) => config.as_ref(),
                _ => None,
            })
    }

    pub fn normalization(&self) -> Option<&NormalizeImageConfig> {
        self.pre_process
            .transform_ops
            .iter()
            .find_map(|op| match op {
                TransformOp::NormalizeImage(config) => Some(config),
                _ => None,
            })
    }

    pub fn recognition_shape(&self) -> Option<[usize; 3]> {
        self.pre_process
            .transform_ops
            .iter()
            .find_map(|op| match op {
                TransformOp::RecResizeImg(config) => Some(config.image_shape),
                _ => None,
            })
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct GlobalConfig {
    pub model_name: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PreProcessConfig {
    pub transform_ops: Vec<TransformOp>,
}

#[derive(Debug, Deserialize)]
pub(crate) enum TransformOp {
    DecodeImage(DecodeImageConfig),
    DetLabelEncode(()),
    DetResizeForTest(Option<DetResizeConfig>),
    NormalizeImage(NormalizeImageConfig),
    ToCHWImage(()),
    KeepKeys(serde::de::IgnoredAny),
    MultiLabelEncode(serde::de::IgnoredAny),
    RecResizeImg(RecResizeConfig),
    ResizeImage(serde::de::IgnoredAny),
    CropImage(serde::de::IgnoredAny),
}

#[derive(Debug, Deserialize)]
pub(crate) struct DecodeImageConfig {
    pub channel_first: bool,
    pub img_mode: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct DetResizeConfig {
    pub resize_long: Option<usize>,
    pub limit_type: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct NormalizeImageConfig {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub scale: Scale,
    pub order: String,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Scale(pub f32);

impl<'de> Deserialize<'de> for Scale {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Repr {
            Number(f32),
            Text(String),
        }

        let value = match Repr::deserialize(deserializer)? {
            Repr::Number(value) => value,
            Repr::Text(value) => match value.trim() {
                "1./255." | "1/255" | "1.0/255.0" => 1.0 / 255.0,
                other => {
                    return Err(serde::de::Error::custom(format!(
                        "unsupported normalization scale expression {other:?}"
                    )));
                }
            },
        };
        Ok(Self(value))
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct RecResizeConfig {
    pub image_shape: [usize; 3],
}

#[derive(Debug, Deserialize)]
pub(crate) struct PostProcessConfig {
    #[serde(default)]
    pub name: String,
    pub thresh: Option<f32>,
    pub box_thresh: Option<f32>,
    pub max_candidates: Option<usize>,
    pub unclip_ratio: Option<f32>,
    pub use_dilation: Option<bool>,
    pub score_mode: Option<String>,
    pub box_type: Option<String>,
    pub character_dict: Option<Vec<String>>,
    pub use_space_char: Option<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_paddlex_scale_expression() {
        #[derive(Deserialize)]
        struct Wrapper {
            scale: Scale,
        }
        let parsed: Wrapper = serde_saphyr::from_str("scale: 1./255.\n").unwrap();
        assert!((parsed.scale.0 - 1.0 / 255.0).abs() < f32::EPSILON);
    }
}
