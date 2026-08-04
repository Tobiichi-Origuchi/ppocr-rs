use std::path::PathBuf;

use ppocr::{
    DocumentOrientationClassifier, DocumentOrientationClassifierOptions, DocumentRectifier,
    DocumentRectifierOptions, TextlineOrientationClassifier, TextlineOrientationClassifierOptions,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let module = arguments.next().ok_or(
        "usage: optional_module_inference <document-orientation|textline-orientation|rectification> <model-id> <model-dir> <image> [output-image]",
    )?;
    let model_id = arguments.next().ok_or("model ID is required")?;
    let model_dir = PathBuf::from(arguments.next().ok_or("model directory is required")?);
    let image_path = PathBuf::from(arguments.next().ok_or("image path is required")?);
    let output_path = arguments.next().map(PathBuf::from);
    if arguments.next().is_some() {
        return Err("too many arguments".into());
    }

    let image = image::load_from_memory(&std::fs::read(image_path)?)?.to_rgb8();
    let onnx_path = model_dir.join("inference.onnx");
    let config_path = model_dir.join("inference.yml");

    match module.as_str() {
        "document-orientation" => {
            let mut classifier = DocumentOrientationClassifier::from_files(
                &model_id,
                &onnx_path,
                &config_path,
                DocumentOrientationClassifierOptions::default(),
            )?;
            println!(
                "{}",
                serde_json::to_string_pretty(&classifier.classify(&image)?)?
            );
        }
        "textline-orientation" => {
            let mut classifier = TextlineOrientationClassifier::from_files(
                &model_id,
                &onnx_path,
                &config_path,
                TextlineOrientationClassifierOptions::default(),
            )?;
            println!(
                "{}",
                serde_json::to_string_pretty(&classifier.classify(&image)?)?
            );
        }
        "rectification" => {
            let mut rectifier = DocumentRectifier::from_files(
                &model_id,
                &onnx_path,
                &config_path,
                DocumentRectifierOptions::default(),
            )?;
            let output = rectifier.rectify(&image)?;
            if let Some(path) = output_path {
                output.save(path)?;
            }
            println!(
                "{}",
                serde_json::json!({
                    "model_id": rectifier.model_id(),
                    "width": output.width(),
                    "height": output.height(),
                })
            );
        }
        _ => return Err(format!("unknown module {module:?}").into()),
    }
    Ok(())
}
