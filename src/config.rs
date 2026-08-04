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

    pub fn classification_resize(&self) -> Option<&ResizeImageConfig> {
        self.pre_process
            .transform_ops
            .iter()
            .find_map(|op| match op {
                TransformOp::ResizeImage(config) => Some(config),
                _ => None,
            })
    }

    pub fn classification_crop(&self) -> Option<&CropImageConfig> {
        self.pre_process
            .transform_ops
            .iter()
            .find_map(|op| match op {
                TransformOp::CropImage(config) => Some(config),
                _ => None,
            })
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct ModelIdentityConfig {
    #[serde(rename = "Global")]
    pub global: GlobalConfig,
}

impl ModelIdentityConfig {
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
    ResizeImage(ResizeImageConfig),
    CropImage(CropImageConfig),
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
    pub channel_num: Option<usize>,
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
pub(crate) struct ResizeImageConfig {
    pub resize_short: Option<usize>,
    pub size: Option<ImageSize>,
    #[serde(default = "default_resize_backend")]
    pub backend: String,
    #[serde(default = "default_resize_interpolation")]
    pub interpolation: String,
}

fn default_resize_backend() -> String {
    "cv2".to_owned()
}

fn default_resize_interpolation() -> String {
    "LINEAR".to_owned()
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(untagged)]
pub(crate) enum ImageSize {
    Square(usize),
    WidthHeight([usize; 2]),
}

impl ImageSize {
    pub fn width_height(self) -> [usize; 2] {
        match self {
            Self::Square(size) => [size, size],
            Self::WidthHeight(size) => size,
        }
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct CropImageConfig {
    pub size: ImageSize,
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
    #[serde(rename = "Topk")]
    pub topk: Option<TopkConfig>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct TopkConfig {
    pub topk: usize,
    pub label_list: Option<Vec<String>>,
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

    #[test]
    fn parses_classification_shapes_and_labels() {
        let parsed: InferenceConfig = serde_saphyr::from_str(
            "Global:\n  model_name: demo\nPreProcess:\n  transform_ops:\n  - ResizeImage:\n      resize_short: 256\n  - CropImage:\n      size: 224\n  - NormalizeImage:\n      mean: [0.485, 0.456, 0.406]\n      std: [0.229, 0.224, 0.225]\n      scale: 0.00392156862745098\n      order: ''\n  - ToCHWImage: null\nPostProcess:\n  Topk:\n    topk: 1\n    label_list: ['0', '90', '180', '270']\n",
        )
        .unwrap();
        assert_eq!(
            parsed.classification_resize().unwrap().resize_short,
            Some(256)
        );
        assert_eq!(
            parsed.classification_crop().unwrap().size.width_height(),
            [224, 224]
        );
        assert_eq!(
            parsed.post_process.topk.unwrap().label_list.unwrap()[1],
            "90"
        );
    }
}
