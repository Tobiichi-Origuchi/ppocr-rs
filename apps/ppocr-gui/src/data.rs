pub const DETECTION_MODELS: &[&str] = &[
    "PP-OCRv6_medium_det",
    "PP-OCRv6_small_det",
    "PP-OCRv6_tiny_det",
    "PP-OCRv5_server_det",
    "PP-OCRv5_mobile_det",
    "PP-OCRv4_server_det",
    "PP-OCRv4_mobile_det",
];

pub const RECOGNITION_MODELS: &[&str] = &[
    "PP-OCRv6_medium_rec",
    "PP-OCRv6_small_rec",
    "PP-OCRv6_tiny_rec",
    "PP-OCRv5_server_rec",
    "PP-OCRv5_mobile_rec",
    "PP-OCRv4_server_rec_doc",
    "PP-OCRv4_mobile_rec",
    "PP-OCRv4_server_rec",
    "PP-OCRv3_mobile_rec",
    "ch_SVTRv2_rec",
    "ch_RepSVTR_rec",
    "en_PP-OCRv5_mobile_rec",
    "en_PP-OCRv4_mobile_rec",
    "en_PP-OCRv3_mobile_rec",
    "korean_PP-OCRv5_mobile_rec",
    "latin_PP-OCRv5_mobile_rec",
    "eslav_PP-OCRv5_mobile_rec",
    "th_PP-OCRv5_mobile_rec",
    "el_PP-OCRv5_mobile_rec",
    "arabic_PP-OCRv5_mobile_rec",
    "cyrillic_PP-OCRv5_mobile_rec",
    "devanagari_PP-OCRv5_mobile_rec",
    "te_PP-OCRv5_mobile_rec",
    "ta_PP-OCRv5_mobile_rec",
    "korean_PP-OCRv3_mobile_rec",
    "japan_PP-OCRv3_mobile_rec",
    "chinese_cht_PP-OCRv3_mobile_rec",
    "te_PP-OCRv3_mobile_rec",
    "ka_PP-OCRv3_mobile_rec",
    "ta_PP-OCRv3_mobile_rec",
    "latin_PP-OCRv3_mobile_rec",
    "arabic_PP-OCRv3_mobile_rec",
    "cyrillic_PP-OCRv3_mobile_rec",
    "devanagari_PP-OCRv3_mobile_rec",
];

pub const DOCUMENT_ORIENTATION_MODELS: &[&str] = &["PP-LCNet_x1_0_doc_ori"];
pub const DOCUMENT_RECTIFICATION_MODELS: &[&str] = &["UVDoc"];
pub const TEXTLINE_ORIENTATION_MODELS: &[&str] =
    &["PP-LCNet_x1_0_textline_ori", "PP-LCNet_x0_25_textline_ori"];

pub const RUNTIME_PROVIDERS: &[&str] = &[
    "CPU",
    "NVIDIA CUDA",
    "DirectML (Windows)",
    "CoreML (macOS)",
    "WebGPU (Vulkan / Metal / DX12)",
    "MIGraphX (AMD ROCm)",
];

pub fn export_formats() -> Vec<String> {
    vec![
        ppocr_i18n::tr!("Plain text (.txt)"),
        ppocr_i18n::tr!("Markdown (.md)"),
        ppocr_i18n::tr!("JSON (.json)"),
        ppocr_i18n::tr!("CSV (.csv)"),
        ppocr_i18n::tr!("TSV (.tsv)"),
        ppocr_i18n::tr!("hOCR (.html)"),
        ppocr_i18n::tr!("ALTO XML (.xml)"),
        ppocr_i18n::tr!("Annotated image (.png)"),
    ]
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RectF {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl RectF {
    pub fn contains(self, x: f64, y: f64) -> bool {
        x >= self.x && x <= self.x + self.width && y >= self.y && y <= self.y + self.height
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_choices_match_expected_inventory() {
        assert_eq!(DETECTION_MODELS.len(), 7);
        assert_eq!(RECOGNITION_MODELS.len(), 34);
        assert_eq!(TEXTLINE_ORIENTATION_MODELS.len(), 2);
        assert_eq!(DOCUMENT_ORIENTATION_MODELS.len(), 1);
        assert_eq!(DOCUMENT_RECTIFICATION_MODELS.len(), 1);
    }

    #[test]
    fn rectangle_hit_testing_handles_inside_and_outside_points() {
        let rectangle = RectF {
            x: 10.0,
            y: 20.0,
            width: 100.0,
            height: 200.0,
        };
        assert!(rectangle.contains(50.0, 100.0));
        assert!(!rectangle.contains(5.0, 100.0));
    }
}
