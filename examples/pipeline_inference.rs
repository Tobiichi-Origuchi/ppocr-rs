use std::{env, fs, path::PathBuf};

use ppocr::{ModelDownloader, ModelDownloaderOptions, OcrPipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args_os().skip(1);
    let image_path =
        PathBuf::from(arguments.next().ok_or(
            "usage: pipeline_inference IMAGE [--minimal|--no-optional] [--threshold=FLOAT]",
        )?);
    let mut minimal = false;
    let mut no_optional = false;
    let mut threshold = None;
    for argument in arguments {
        let argument = argument.to_string_lossy();
        match argument.as_ref() {
            "--minimal" => minimal = true,
            "--no-optional" => no_optional = true,
            value if value.starts_with("--threshold=") => {
                threshold = Some(value["--threshold=".len()..].parse::<f32>()?);
            }
            _ => return Err(format!("unknown argument {argument:?}").into()),
        }
    }
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
    if let Some(threshold) = threshold {
        builder = builder.text_recognition_score_threshold(threshold);
    }
    let mut pipeline = builder.build()?;
    let result = pipeline.predict(&image)?;

    println!("model_settings={:?}", result.model_settings);
    println!("model_ids={:?}", result.model_ids);
    println!(
        "text_detection_parameters={:?} recognition_threshold={}",
        result.text_detection_parameters, result.text_recognition_score_threshold
    );
    if let Some(orientation) = result.document_orientation {
        println!(
            "document_orientation={} score={:.5}",
            orientation.angle, orientation.score
        );
    }
    println!(
        "input={}x{} processed_image={}x{} detections={} recognized_lines={}",
        result.input_size.width,
        result.input_size.height,
        result.processed_size.width,
        result.processed_size.height,
        result.detections.len(),
        result.text_lines.len(),
    );
    for line in result.text_lines {
        let angle = line
            .textline_orientation
            .as_ref()
            .map_or(-1, |orientation| i32::from(orientation.angle));
        println!(
            "detection_index={} det_score={:.6} text={:?} rec_score={:.6} angle={} box={:?} polygon={:?}",
            line.detection_index,
            line.detection_score,
            line.text,
            line.recognition_score,
            angle,
            line.bounding_box,
            line.polygon.map(|point| [point.x, point.y])
        );
    }
    Ok(())
}
