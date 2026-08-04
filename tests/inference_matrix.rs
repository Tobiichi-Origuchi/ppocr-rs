use std::{env, fs, path::Path};

use image::{Rgb, RgbImage, imageops};
use ppocr::image_geometry::{crop_text_region, sort_text_regions};
use ppocr::{
    DocumentOrientationClassifier, DocumentOrientationClassifierOptions, DocumentRectifier,
    DocumentRectifierOptions, ModelDownloader, ModelDownloaderOptions, ModelKind, OcrPipeline,
    TextDetector, TextDetectorOptions, TextRecognizer, TextRecognizerOptions,
    TextlineOrientationClassifier, TextlineOrientationClassifierOptions, manifest,
};

const EXPECTED_CHINESE_LINES: [&str; 4] = [
    "上海斯格威铂尔大酒店",
    "打浦路15号",
    "绿洲仕格维花园公寓",
    "打浦路252935号",
];

fn downloader() -> ModelDownloader {
    ModelDownloader::new(ModelDownloaderOptions::default()).unwrap()
}

fn synthetic_image(width: u32, height: u32) -> RgbImage {
    RgbImage::from_fn(width, height, |x, y| {
        Rgb([
            ((x * 13 + y * 3) % 256) as u8,
            ((x * 2 + y * 17) % 256) as u8,
            ((x * 7 + y * 11) % 256) as u8,
        ])
    })
}

fn fixture(variable: &str, fallback: RgbImage) -> (RgbImage, bool) {
    let Some(path) = env::var_os(variable) else {
        return (fallback, false);
    };
    let bytes = fs::read(Path::new(&path)).unwrap_or_else(|error| {
        panic!("failed to read {variable}={path:?}: {error}");
    });
    (
        image::load_from_memory(&bytes)
            .unwrap_or_else(|error| panic!("failed to decode {variable}={path:?}: {error}"))
            .into_rgb8(),
        true,
    )
}

fn language_fixture(variant: Option<&str>) -> Option<RgbImage> {
    let directory = env::var_os("PPOCR_MATRIX_LANGUAGE_FIXTURES")?;
    let path = Path::new(&directory).join(format!("{}.png", variant?));
    if !path.is_file() {
        return None;
    }
    let bytes = fs::read(&path)
        .unwrap_or_else(|error| panic!("failed to read language fixture {path:?}: {error}"));
    Some(
        image::load_from_memory(&bytes)
            .unwrap_or_else(|error| panic!("failed to decode language fixture {path:?}: {error}"))
            .into_rgb8(),
    )
}

fn assert_detection_result(result: &ppocr::TextDetectionResult, image: &RgbImage) {
    for region in &result.regions {
        assert!(region.score.is_finite());
        assert!((0.0..=1.0).contains(&region.score));
        for point in region.points {
            assert!(point.x.is_finite() && point.y.is_finite());
            assert!(point.x >= 0.0 && point.x <= image.width() as f32);
            assert!(point.y >= 0.0 && point.y <= image.height() as f32);
        }
    }
}

fn assert_recognition_results(results: &[ppocr::TextRecognitionResult], expected_len: usize) {
    assert_eq!(results.len(), expected_len);
    for result in results {
        assert!(result.score.is_finite());
        assert!((0.0..=1.0).contains(&result.score));
    }
}

#[test]
#[ignore = "downloads and executes all 7 detectors with all 34 recognizers"]
fn all_detector_recognizer_pairs_run() {
    let downloader = downloader();
    let (document, official_document) = fixture("PPOCR_MATRIX_DOCUMENT", synthetic_image(321, 193));
    let (recognition_fixture, official_recognition) =
        fixture("PPOCR_MATRIX_RECOGNITION", synthetic_image(317, 48));
    let blank = RgbImage::from_pixel(96, 64, Rgb([255, 255, 255]));
    let odd = synthetic_image(257, 193);

    let detector_specs: Vec<_> = manifest()
        .models
        .iter()
        .filter(|model| model.kind == ModelKind::TextDetection)
        .collect();
    let recognizer_specs: Vec<_> = manifest()
        .models
        .iter()
        .filter(|model| model.kind == ModelKind::TextRecognition)
        .collect();
    assert_eq!(detector_specs.len(), 7);
    assert_eq!(recognizer_specs.len(), 34);

    let mut detector_crops = Vec::with_capacity(detector_specs.len());
    for (index, spec) in detector_specs.iter().enumerate() {
        eprintln!(
            "detector {}/{}: {}",
            index + 1,
            detector_specs.len(),
            spec.id
        );
        let mut detector =
            TextDetector::from_model(&spec.id, &downloader, TextDetectorOptions::default())
                .unwrap();
        let inputs = [blank.clone(), odd.clone(), document.clone()];
        let results = detector.detect_batch(&inputs).unwrap();
        assert_eq!(results.len(), inputs.len());
        for (result, input) in results.iter().zip(&inputs) {
            assert_detection_result(result, input);
        }

        let regions = sort_text_regions(&results[2].regions);
        if official_document {
            assert_eq!(
                regions.len(),
                EXPECTED_CHINESE_LINES.len(),
                "{} returned an unexpected detection count for the official OCR document",
                spec.id
            );
        }
        let crops: Vec<_> = regions
            .iter()
            .map(|region| crop_text_region(&document, &region.points).unwrap())
            .collect();
        detector_crops.push((spec.id.clone(), crops));
    }

    let mut pair_count = 0;
    for (index, spec) in recognizer_specs.iter().enumerate() {
        eprintln!(
            "recognizer {}/{} across {} detectors: {}",
            index + 1,
            recognizer_specs.len(),
            detector_crops.len(),
            spec.id
        );
        let mut recognizer =
            TextRecognizer::from_model(&spec.id, &downloader, TextRecognizerOptions::default())
                .unwrap();
        let native_fixture = language_fixture(spec.variant.as_deref());
        let semantic_fixture = native_fixture.as_ref().unwrap_or(&recognition_fixture);
        let dynamic_inputs = [
            synthetic_image(37, 53),
            semantic_fixture.clone(),
            imageops::rotate180(semantic_fixture),
            synthetic_image(1001, 37),
        ];
        let dynamic_results = recognizer.recognize_batch(&dynamic_inputs).unwrap();
        assert_recognition_results(&dynamic_results, dynamic_inputs.len());
        let chinese_model = matches!(spec.variant.as_deref(), Some("multilingual" | "chinese"));
        if native_fixture.is_some() || (official_recognition && chinese_model) {
            assert!(
                !dynamic_results[1].text.is_empty(),
                "{} produced an empty decode for its semantic text crop",
                spec.id
            );
            eprintln!(
                "semantic crop decode for {}: {:?} ({:.6})",
                spec.id, dynamic_results[1].text, dynamic_results[1].score
            );
        }
        if official_recognition && chinese_model {
            assert_eq!(dynamic_results[1].text, EXPECTED_CHINESE_LINES[2]);
        }
        assert!(recognizer.recognize_batch(&[]).unwrap().is_empty());

        for (detector_id, crops) in &detector_crops {
            pair_count += 1;
            let results = recognizer.recognize_batch(crops).unwrap_or_else(|error| {
                panic!(
                    "detector/recognizer pair {detector_id}/{} failed: {error}",
                    spec.id
                )
            });
            assert_recognition_results(&results, crops.len());
            if official_document && chinese_model {
                // Older/tiny recognizers can reproduce PaddleX's own extra or
                // substituted glyphs on crops produced by another generation.
                assert!(
                    results.iter().all(|result| !result.text.is_empty()),
                    "detector/recognizer pair {detector_id}/{} returned an empty line",
                    spec.id,
                );
            }
        }
    }
    assert_eq!(pair_count, 7 * 34);
}

#[test]
#[ignore = "downloads and executes every optional module model"]
fn all_optional_models_run() {
    let downloader = downloader();
    let (document_rotated, official_document) = fixture(
        "PPOCR_MATRIX_DOCUMENT_ORIENTATION",
        imageops::rotate180(&synthetic_image(320, 192)),
    );
    let document_upright = imageops::rotate180(&document_rotated);
    let (textline_rotated, official_textline) = fixture(
        "PPOCR_MATRIX_TEXTLINE_ORIENTATION",
        imageops::rotate180(&synthetic_image(320, 80)),
    );
    let textline_upright = imageops::rotate180(&textline_rotated);
    let (rectification_image, _) = fixture("PPOCR_MATRIX_RECTIFICATION", synthetic_image(320, 192));

    let mut document_classifier = DocumentOrientationClassifier::from_model(
        "PP-LCNet_x1_0_doc_ori",
        &downloader,
        DocumentOrientationClassifierOptions::default(),
    )
    .unwrap();
    let document_results = document_classifier
        .classify_batch(&[document_rotated, document_upright])
        .unwrap();
    assert_eq!(document_results.len(), 2);
    assert!(
        document_results
            .iter()
            .all(|result| result.score.is_finite())
    );
    if official_document {
        assert_eq!(document_results[0].angle, 180);
        assert_eq!(document_results[1].angle, 0);
    }

    for model_id in ["PP-LCNet_x0_25_textline_ori", "PP-LCNet_x1_0_textline_ori"] {
        let mut classifier = TextlineOrientationClassifier::from_model(
            model_id,
            &downloader,
            TextlineOrientationClassifierOptions::default(),
        )
        .unwrap();
        let results = classifier
            .classify_batch(&[textline_rotated.clone(), textline_upright.clone()])
            .unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|result| result.score.is_finite()));
        if official_textline {
            assert_eq!(results[0].angle, 180, "{model_id}");
            assert_eq!(results[1].angle, 0, "{model_id}");
        }
    }

    let mut rectifier =
        DocumentRectifier::from_model("UVDoc", &downloader, DocumentRectifierOptions::default())
            .unwrap();
    let odd = synthetic_image(129, 97);
    let inputs = [odd, rectification_image];
    let results = rectifier.rectify_batch(&inputs).unwrap();
    assert_eq!(results.len(), inputs.len());
    for (result, input) in results.iter().zip(&inputs) {
        assert_eq!(result.dimensions(), input.dimensions());
    }
    assert!(rectifier.rectify_batch(&[]).unwrap().is_empty());
}

#[test]
#[ignore = "downloads and executes all 12 optional-stage pipeline configurations"]
fn all_optional_pipeline_configurations_run() {
    let downloader = downloader();
    let (document, official_document) = fixture(
        "PPOCR_MATRIX_DOCUMENT",
        RgbImage::from_pixel(320, 192, Rgb([255, 255, 255])),
    );
    let rotated_document = imageops::rotate180(&document);
    let blank = RgbImage::from_pixel(256, 128, Rgb([255, 255, 255]));
    let inputs = [document, rotated_document, blank];
    let textline_models = [
        None,
        Some("PP-LCNet_x0_25_textline_ori"),
        Some("PP-LCNet_x1_0_textline_ori"),
    ];

    let mut configuration_count = 0;
    for use_document_orientation in [false, true] {
        for use_document_rectification in [false, true] {
            for textline_model in textline_models {
                configuration_count += 1;
                eprintln!(
                    "pipeline configuration {configuration_count}/12: doc_ori={use_document_orientation} uvdoc={use_document_rectification} textline={textline_model:?}"
                );
                let mut builder = OcrPipeline::builder(&downloader);
                if !use_document_orientation {
                    builder = builder.disable_document_orientation();
                }
                if !use_document_rectification {
                    builder = builder.disable_document_rectification();
                }
                builder = match textline_model {
                    Some(model_id) => builder.textline_orientation_model(model_id),
                    None => builder.disable_textline_orientation(),
                };
                let mut pipeline = builder.build().unwrap();
                let results = pipeline.predict_batch(&inputs).unwrap();
                assert_eq!(results.len(), inputs.len());

                for result in &results {
                    assert_eq!(
                        result.model_settings.use_document_orientation,
                        use_document_orientation
                    );
                    assert_eq!(
                        result.model_settings.use_document_rectification,
                        use_document_rectification
                    );
                    assert_eq!(
                        result.model_settings.use_textline_orientation,
                        textline_model.is_some()
                    );
                    assert_eq!(
                        result.model_ids.textline_orientation.as_deref(),
                        textline_model
                    );
                    assert_eq!(result.detections.len(), result.textline_orientations.len());
                    assert!(
                        result
                            .text_lines
                            .windows(2)
                            .all(|pair| pair[0].detection_index < pair[1].detection_index)
                    );
                    assert!(result.text_lines.iter().all(|line| {
                        line.detection_index < result.detections.len()
                            && line.recognition_score.is_finite()
                            && line.detection_score.is_finite()
                            && line.bounding_box.iter().all(|value| value.is_finite())
                    }));
                    assert!(
                        result.textline_orientations.iter().all(|orientation| {
                            orientation.is_some() == textline_model.is_some()
                        })
                    );
                }

                if official_document {
                    let texts: Vec<_> = results[0]
                        .text_lines
                        .iter()
                        .map(|line| line.text.as_str())
                        .collect();
                    assert_eq!(texts, EXPECTED_CHINESE_LINES);
                }
                assert!(results[2].text_lines.is_empty());
            }
        }
    }
    assert_eq!(configuration_count, 12);
}
