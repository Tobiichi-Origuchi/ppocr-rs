use image::{Rgb, RgbImage};
use ppocr::{
    ModelDownloader, ModelDownloaderOptions, TextDetector, TextDetectorOptions, TextRecognizer,
    TextRecognizerOptions,
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
