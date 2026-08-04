use std::{env, fs, path::PathBuf};

use ppocr::{ModelDownloader, ModelDownloaderOptions, OcrPipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args_os().skip(1);
    let image_path = PathBuf::from(
        arguments
            .next()
            .ok_or("usage: pipeline_inference IMAGE [--minimal|--no-optional]")?,
    );
    let mode = arguments.next();
    let minimal = mode.as_deref().is_some_and(|value| value == "--minimal");
    let no_optional = mode
        .as_deref()
        .is_some_and(|value| value == "--no-optional");
    let image = image::load_from_memory(&fs::read(image_path)?)?.into_rgb8();
    let downloader = ModelDownloader::new(ModelDownloaderOptions::default())?;
    let mut builder = OcrPipeline::builder(&downloader);
    if minimal {
        builder = builder
            .text_detection_model("PP-OCRv6_tiny_det")
            .text_recognition_model("PP-OCRv6_tiny_rec")
            .disable_document_orientation()
            .disable_document_rectification()
            .disable_textline_orientation();
    } else if no_optional {
        builder = builder
            .disable_document_orientation()
            .disable_document_rectification()
            .disable_textline_orientation();
    }
    let mut pipeline = builder.build()?;
    let result = pipeline.predict(&image)?;

    if let Some(orientation) = result.document_orientation {
        println!(
            "document_orientation={} score={:.5}",
            orientation.angle, orientation.score
        );
    }
    println!(
        "processed_image={}x{} regions={}",
        result.processed_image.width(),
        result.processed_image.height(),
        result.regions.len()
    );
    for region in result.regions {
        let angle = region
            .textline_orientation
            .as_ref()
            .map_or(-1, |orientation| i32::from(orientation.angle));
        println!(
            "text={:?} score={:.6} angle={} polygon={:?}",
            region.recognition.text,
            region.recognition.score,
            angle,
            region.detection.points.map(|point| [point.x, point.y])
        );
    }
    Ok(())
}
