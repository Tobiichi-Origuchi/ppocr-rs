use std::path::PathBuf;

use ppocr::{
    ModelDownloader, ModelDownloaderOptions, TextDetector, TextDetectorOptions, TextRecognizer,
    TextRecognizerOptions,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let module = arguments
        .next()
        .ok_or("usage: module_inference <detection|recognition> <model-id> <image> [cache-dir]")?;
    let model_id = arguments.next().ok_or("model ID is required")?;
    let image_path = PathBuf::from(arguments.next().ok_or("image path is required")?);
    let cache_dir = arguments.next().map(PathBuf::from);
    if arguments.next().is_some() {
        return Err("too many arguments".into());
    }

    // Some official examples use an extension that differs from their encoded
    // format, so decode from the content instead of trusting the path suffix.
    let image = image::load_from_memory(&std::fs::read(image_path)?)?.to_rgb8();
    let downloader = ModelDownloader::new(ModelDownloaderOptions {
        cache_dir,
        ..ModelDownloaderOptions::default()
    })?;

    match module.as_str() {
        "detection" => {
            let mut detector =
                TextDetector::from_model(&model_id, &downloader, TextDetectorOptions::default())?;
            println!(
                "{}",
                serde_json::to_string_pretty(&detector.detect(&image)?)?
            );
        }
        "recognition" => {
            let mut recognizer = TextRecognizer::from_model(
                &model_id,
                &downloader,
                TextRecognizerOptions::default(),
            )?;
            println!(
                "{}",
                serde_json::to_string_pretty(&recognizer.recognize(&image)?)?
            );
        }
        _ => return Err(format!("unknown module {module:?}").into()),
    }
    Ok(())
}
