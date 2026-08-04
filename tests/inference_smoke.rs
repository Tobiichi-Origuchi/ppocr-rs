use std::{env, fs, path::Path};

use image::{Rgb, RgbImage};
use ppocr::{
    DocumentOrientationClassifier, DocumentOrientationClassifierOptions, DocumentRectifier,
    DocumentRectifierOptions, ModelDownloader, ModelDownloaderOptions, OcrPipeline,
    OcrPipelineOptions, OrtSessionOptions, TextDetector, TextDetectorOptions, TextRecognizer,
    TextRecognizerOptions, TextlineOrientationClassifier, TextlineOrientationClassifierOptions,
};

fn load_smoke_document() -> RgbImage {
    let path = env::var_os("PPOCR_SMOKE_DOCUMENT")
        .expect("PPOCR_SMOKE_DOCUMENT must point to a real OCR image");
    image::load_from_memory(&fs::read(Path::new(&path)).unwrap())
        .unwrap()
        .into_rgb8()
}

#[cfg(any(
    feature = "cuda",
    feature = "webgpu",
    feature = "coreml",
    feature = "directml",
    feature = "migraphx"
))]
fn assert_full_pipeline_result(result: &ppocr::OcrPipelineResult) {
    assert!(result.document_orientation.is_some());
    assert_eq!(result.textline_orientations.len(), result.detections.len());
    assert!(result.textline_orientations.iter().all(Option::is_some));
    assert!(!result.text_lines.is_empty());
    assert!(
        result
            .text_lines
            .iter()
            .all(|line| line.recognition_score.is_finite())
    );
    assert_eq!(
        result
            .text_lines
            .iter()
            .map(|line| line.text.as_str())
            .collect::<Vec<_>>(),
        [
            "上海斯格威铂尔大酒店",
            "打浦路15号",
            "绿洲仕格维花园公寓",
            "打浦路252935号",
        ]
    );
}

#[test]
#[ignore = "downloads models from Hugging Face"]
fn tiny_detector_runs_end_to_end() {
    let downloader = ModelDownloader::new(ModelDownloaderOptions::default()).unwrap();
    let mut detector = TextDetector::from_model(
        "PP-OCRv6_tiny_det",
        &downloader,
        TextDetectorOptions::default(),
    )
    .unwrap();
    let image = RgbImage::from_pixel(128, 64, Rgb([255, 255, 255]));
    let result = detector.detect(&image).unwrap();
    assert!(result.regions.iter().all(|region| region.score.is_finite()));
}

#[test]
#[ignore = "downloads models from Hugging Face"]
fn tiny_recognizer_runs_end_to_end() {
    let downloader = ModelDownloader::new(ModelDownloaderOptions::default()).unwrap();
    let mut recognizer = TextRecognizer::from_model(
        "PP-OCRv6_tiny_rec",
        &downloader,
        TextRecognizerOptions::default(),
    )
    .unwrap();
    let image = RgbImage::from_pixel(160, 48, Rgb([255, 255, 255]));
    let result = recognizer.recognize(&image).unwrap();
    assert!(result.score.is_finite());
}

#[test]
#[ignore = "downloads models from Hugging Face"]
fn optional_classifiers_run_end_to_end() {
    let downloader = ModelDownloader::new(ModelDownloaderOptions::default()).unwrap();
    let image = RgbImage::from_pixel(320, 160, Rgb([255, 255, 255]));

    let mut document = DocumentOrientationClassifier::from_model(
        "PP-LCNet_x1_0_doc_ori",
        &downloader,
        DocumentOrientationClassifierOptions::default(),
    )
    .unwrap();
    assert!(document.classify(&image).unwrap().score.is_finite());

    let mut textline = TextlineOrientationClassifier::from_model(
        "PP-LCNet_x0_25_textline_ori",
        &downloader,
        TextlineOrientationClassifierOptions::default(),
    )
    .unwrap();
    assert!(textline.classify(&image).unwrap().score.is_finite());
}

#[test]
#[ignore = "downloads models from Hugging Face"]
fn document_rectifier_runs_end_to_end() {
    let downloader = ModelDownloader::new(ModelDownloaderOptions::default()).unwrap();
    let mut rectifier =
        DocumentRectifier::from_model("UVDoc", &downloader, DocumentRectifierOptions::default())
            .unwrap();
    let image = RgbImage::from_pixel(128, 64, Rgb([255, 255, 255]));
    let output = rectifier.rectify(&image).unwrap();
    assert_eq!(output.dimensions(), image.dimensions());
}

#[test]
#[ignore = "downloads only the selected detection and recognition models from Hugging Face"]
fn selected_minimal_pipeline_runs_end_to_end() {
    let downloader = ModelDownloader::new(ModelDownloaderOptions::default()).unwrap();
    let mut pipeline = OcrPipeline::builder(&downloader)
        .text_detection_model("PP-OCRv6_tiny_det")
        .text_recognition_model("PP-OCRv6_tiny_rec")
        .disable_document_orientation()
        .disable_document_rectification()
        .disable_textline_orientation()
        .build()
        .unwrap();
    let image = RgbImage::from_pixel(128, 64, Rgb([255, 255, 255]));
    let result = pipeline.predict(&image).unwrap();
    assert_eq!(result.processed_image.dimensions(), image.dimensions());
    assert!(result.document_orientation.is_none());
    assert!(
        result
            .text_lines
            .iter()
            .all(|line| line.recognition_score.is_finite())
    );
    let batch = pipeline
        .predict_batch(&[image.clone(), image.clone()])
        .unwrap();
    assert_eq!(batch.len(), 2);
    assert!(pipeline.predict_batch(&[]).unwrap().is_empty());
}

#[test]
#[ignore = "downloads models and requires PPOCR_SMOKE_DOCUMENT"]
fn word_boxes_and_pipeline_pool_run_end_to_end() {
    let image = load_smoke_document();
    let downloader = ModelDownloader::new(ModelDownloaderOptions::default()).unwrap();
    let pool = OcrPipeline::builder(&downloader)
        .disable_document_orientation()
        .disable_document_rectification()
        .disable_textline_orientation()
        .session_options(OrtSessionOptions {
            intra_threads: Some(1),
            ..OrtSessionOptions::default()
        })
        .pipeline_options(OcrPipelineOptions {
            return_word_boxes: true,
            ..OcrPipelineOptions::default()
        })
        .build_pool(2)
        .unwrap();
    assert_eq!(pool.worker_count(), 2);

    let results = pool.predict_batch(&[image.clone(), image]).unwrap();
    assert_eq!(results.len(), 2);
    let expected_words = [
        vec!["上", "海", "斯", "格", "威", "铂", "尔", "大", "酒", "店"],
        vec!["打", "浦", "路", "15", "号"],
        vec!["绿", "洲", "仕", "格", "维", "花", "园", "公", "寓"],
        vec!["打", "浦", "路", "252935", "号"],
    ];
    let expected_boxes = [
        vec![
            [34.0, 409.0, 81.0, 432.0],
            [72.0, 409.0, 118.0, 432.0],
            [117.0, 409.0, 163.0, 432.0],
            [154.0, 409.0, 200.0, 432.0],
            [199.0, 409.0, 245.0, 432.0],
            [236.0, 409.0, 283.0, 432.0],
            [281.0, 409.0, 327.0, 432.0],
            [363.0, 409.0, 409.0, 432.0],
            [400.0, 409.0, 447.0, 432.0],
            [445.0, 409.0, 486.0, 432.0],
        ],
        vec![
            [198.0, 453.0, 242.0, 483.0],
            [236.0, 453.0, 280.0, 483.0],
            [280.0, 453.0, 324.0, 483.0],
            [324.0, 453.0, 356.0, 483.0],
            [362.0, 453.0, 399.0, 483.0],
        ],
        vec![
            [14.0, 505.0, 70.0, 533.0],
            [76.0, 505.0, 134.0, 533.0],
            [124.0, 505.0, 182.0, 533.0],
            [188.0, 505.0, 246.0, 533.0],
            [244.0, 505.0, 302.0, 533.0],
            [300.0, 505.0, 358.0, 533.0],
            [356.0, 505.0, 414.0, 533.0],
            [404.0, 505.0, 462.0, 533.0],
            [468.0, 505.0, 517.0, 533.0],
        ],
        vec![
            [76.0, 551.0, 122.0, 575.0],
            [116.0, 551.0, 161.0, 575.0],
            [161.0, 551.0, 206.0, 575.0],
            [215.0, 551.0, 345.0, 575.0],
            [353.0, 551.0, 396.0, 575.0],
        ],
    ];
    for result in results {
        assert!(result.return_word_boxes);
        assert_eq!(result.text_lines.len(), expected_words.len());
        for (line_index, line) in result.text_lines.into_iter().enumerate() {
            let words = line.words.expect("word boxes were requested");
            assert!(!words.is_empty());
            assert_eq!(
                words
                    .iter()
                    .map(|word| word.text.as_str())
                    .collect::<String>(),
                line.text
            );
            assert_eq!(
                words
                    .iter()
                    .map(|word| word.text.as_str())
                    .collect::<Vec<_>>(),
                expected_words[line_index]
            );
            assert_eq!(
                words
                    .iter()
                    .map(|word| word.bounding_box)
                    .collect::<Vec<_>>(),
                expected_boxes[line_index]
            );
            assert!(words.iter().all(|word| {
                word.polygon
                    .iter()
                    .all(|point| point.x.is_finite() && point.y.is_finite())
            }));
        }
    }
}

#[cfg(all(feature = "webgpu", any(target_os = "linux", target_os = "windows")))]
#[test]
#[ignore = "requires a Vulkan-capable device, WebGPU-enabled ORT, models, and PPOCR_SMOKE_DOCUMENT"]
fn webgpu_vulkan_provider_runs_the_full_pipeline() {
    use ppocr::execution_providers::{
        ExecutionProvider, WebGPU,
        webgpu::{DawnBackendType, PreferredLayout},
    };

    let device_id = env::var("PPOCR_WEBGPU_DEVICE_ID")
        .ok()
        .map(|value| value.parse::<i32>().expect("valid WebGPU device ID"))
        .unwrap_or(0);
    let provider = WebGPU::default()
        .with_dawn_backend_type(DawnBackendType::Vulkan)
        .with_device_id(device_id)
        .with_preferred_layout(PreferredLayout::NCHW);
    assert!(provider.is_available().unwrap());

    let downloader = ModelDownloader::new(ModelDownloaderOptions::default()).unwrap();
    let mut pipeline = OcrPipeline::builder(&downloader)
        .session_options(OrtSessionOptions {
            execution_providers: vec![provider.build().error_on_failure()],
            ..OrtSessionOptions::default()
        })
        .build()
        .unwrap();
    let result = pipeline.predict(&load_smoke_document()).unwrap();
    assert_full_pipeline_result(&result);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an NVIDIA GPU, matching CUDA/cuDNN libraries, models, and PPOCR_SMOKE_DOCUMENT"]
fn cuda_provider_runs_the_full_pipeline() {
    use ppocr::execution_providers::{CUDA, ExecutionProvider};

    let provider = CUDA::default().with_device_id(0);
    assert!(provider.is_available().unwrap());

    let downloader = ModelDownloader::new(ModelDownloaderOptions::default()).unwrap();
    let mut pipeline = OcrPipeline::builder(&downloader)
        .session_options(OrtSessionOptions {
            execution_providers: vec![provider.build().error_on_failure()],
            ..OrtSessionOptions::default()
        })
        .build()
        .unwrap();
    let result = pipeline.predict(&load_smoke_document()).unwrap();
    assert_full_pipeline_result(&result);
}

#[cfg(all(feature = "webgpu", target_os = "macos"))]
#[test]
#[ignore = "requires an Apple Metal device, WebGPU-enabled ORT, models, and PPOCR_SMOKE_DOCUMENT"]
fn webgpu_metal_provider_runs_the_full_pipeline() {
    use ppocr::execution_providers::{ExecutionProvider, WebGPU};

    // Leaving Dawn's backend unset selects the native Metal backend on Apple.
    let provider = WebGPU::default();
    assert!(provider.is_available().unwrap());

    let downloader = ModelDownloader::new(ModelDownloaderOptions::default()).unwrap();
    let mut pipeline = OcrPipeline::builder(&downloader)
        .session_options(OrtSessionOptions {
            execution_providers: vec![provider.build().error_on_failure()],
            ..OrtSessionOptions::default()
        })
        .build()
        .unwrap();
    let result = pipeline.predict(&load_smoke_document()).unwrap();
    assert_full_pipeline_result(&result);
}

#[cfg(feature = "coreml")]
#[test]
#[ignore = "requires an Apple device, CoreML-enabled ORT, models, and PPOCR_SMOKE_DOCUMENT"]
fn coreml_provider_runs_the_full_pipeline() {
    use ppocr::execution_providers::{
        CoreML, ExecutionProvider,
        coreml::{ComputeUnits, ModelFormat},
    };

    let provider = CoreML::default()
        .with_compute_units(ComputeUnits::All)
        .with_model_format(ModelFormat::MLProgram);
    assert!(provider.is_available().unwrap());

    let downloader = ModelDownloader::new(ModelDownloaderOptions::default()).unwrap();
    let mut pipeline = OcrPipeline::builder(&downloader)
        .session_options(OrtSessionOptions {
            execution_providers: vec![provider.build().error_on_failure()],
            ..OrtSessionOptions::default()
        })
        .build()
        .unwrap();
    let result = pipeline.predict(&load_smoke_document()).unwrap();
    assert_full_pipeline_result(&result);
}

#[cfg(feature = "directml")]
#[test]
#[ignore = "requires a DirectX 12 GPU, DirectML-enabled ORT, models, and PPOCR_SMOKE_DOCUMENT"]
fn directml_provider_runs_the_full_pipeline() {
    use ppocr::execution_providers::{DirectML, ExecutionProvider};

    let provider = DirectML::default();
    assert!(provider.is_available().unwrap());

    let downloader = ModelDownloader::new(ModelDownloaderOptions::default()).unwrap();
    let mut pipeline = OcrPipeline::builder(&downloader)
        .session_options(OrtSessionOptions {
            execution_providers: vec![provider.build().error_on_failure()],
            parallel_execution: false,
            ..OrtSessionOptions::default()
        })
        .build()
        .unwrap();
    let result = pipeline.predict(&load_smoke_document()).unwrap();
    assert_full_pipeline_result(&result);
}

#[cfg(feature = "migraphx")]
#[test]
#[ignore = "requires a custom MIGraphX-enabled ORT, ROCm, an AMD GPU, models, and PPOCR_SMOKE_DOCUMENT"]
fn migraphx_provider_runs_the_full_pipeline() {
    use ppocr::execution_providers::{ExecutionProvider, MIGraphX};

    let provider = MIGraphX::default().with_device_id(0);
    assert!(provider.is_available().unwrap());

    let downloader = ModelDownloader::new(ModelDownloaderOptions::default()).unwrap();
    let mut pipeline = OcrPipeline::builder(&downloader)
        .session_options(OrtSessionOptions {
            execution_providers: vec![provider.build().error_on_failure()],
            ..OrtSessionOptions::default()
        })
        .build()
        .unwrap();
    let result = pipeline.predict(&load_smoke_document()).unwrap();
    assert_full_pipeline_result(&result);
}
