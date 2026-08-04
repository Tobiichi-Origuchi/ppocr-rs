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

pub const EXPORT_FORMATS: &[&str] = &[
    "纯文本 (.txt)",
    "Markdown (.md)",
    "JSON (.json)",
    "CSV (.csv)",
    "TSV (.tsv)",
    "hOCR (.html)",
    "ALTO XML (.xml)",
    "标注图片 (.png)",
];

#[derive(Clone, Copy, Debug)]
pub struct DemoLine {
    pub text: &'static str,
    pub confidence: f32,
    pub bounds: [f64; 4],
}

impl DemoLine {
    const fn new(text: &'static str, confidence: f32, bounds: [f64; 4]) -> Self {
        Self {
            text,
            confidence,
            bounds,
        }
    }
}

pub const DEMO_LINES: &[DemoLine] = &[
    DemoLine::new("PaddleOCR 通用文字识别", 0.998, [0.13, 0.10, 0.86, 0.16]),
    DemoLine::new("原生 GTK4 界面预览", 0.995, [0.26, 0.19, 0.74, 0.24]),
    DemoLine::new("文件：示例文档.png", 0.991, [0.13, 0.31, 0.53, 0.35]),
    DemoLine::new(
        "检测模型：PP-OCRv6_medium_det",
        0.987,
        [0.13, 0.39, 0.82, 0.43],
    ),
    DemoLine::new(
        "识别模型：PP-OCRv6_medium_rec",
        0.984,
        [0.13, 0.47, 0.82, 0.51],
    ),
    DemoLine::new(
        "支持截图、图片导入与多格式导出",
        0.976,
        [0.13, 0.59, 0.87, 0.64],
    ),
    DemoLine::new(
        "前端与推理进程将在下一阶段通过 IPC 连接",
        0.968,
        [0.13, 0.67, 0.90, 0.72],
    ),
    DemoLine::new("置信度 96.8%", 0.968, [0.13, 0.82, 0.45, 0.86]),
];

pub fn demo_text() -> String {
    DEMO_LINES
        .iter()
        .map(|line| line.text)
        .collect::<Vec<_>>()
        .join("\n")
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

    pub fn map_normalized(self, bounds: [f64; 4]) -> Self {
        Self {
            x: self.x + self.width * bounds[0],
            y: self.y + self.height * bounds[1],
            width: self.width * (bounds[2] - bounds[0]),
            height: self.height * (bounds[3] - bounds[1]),
        }
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
    fn demo_text_preserves_line_order() {
        let text = demo_text();
        assert_eq!(text.lines().count(), DEMO_LINES.len());
        assert!(text.starts_with(DEMO_LINES[0].text));
        assert!(text.ends_with(DEMO_LINES[DEMO_LINES.len() - 1].text));
    }

    #[test]
    fn normalized_rectangles_map_and_hit_test() {
        let page = RectF {
            x: 10.0,
            y: 20.0,
            width: 100.0,
            height: 200.0,
        };
        let mapped = page.map_normalized([0.1, 0.2, 0.8, 0.9]);
        assert_eq!(
            mapped,
            RectF {
                x: 20.0,
                y: 60.0,
                width: 70.0,
                height: 140.0,
            }
        );
        assert!(mapped.contains(50.0, 100.0));
        assert!(!mapped.contains(5.0, 100.0));
    }
}
