use image::{Rgb, RgbImage};
use ppocr::{
    DocumentOrientationClassifier, DocumentOrientationClassifierOptions, DocumentRectifier,
    DocumentRectifierOptions, ModelDownloader, ModelDownloaderOptions, OcrPipeline, TextDetector,
    TextDetectorOptions, TextRecognizer, TextRecognizerOptions, TextlineOrientationClassifier,
    TextlineOrientationClassifierOptions,
};

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
            .regions
            .iter()
            .all(|region| region.recognition.score.is_finite())
    );
}
