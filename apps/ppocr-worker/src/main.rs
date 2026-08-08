mod engine;
mod server;

fn main() {
    ppocr_i18n::init().expect("failed to initialize translations");
    if let Err(error) = server::run_stdio(engine::OcrEngine::default()) {
        eprintln!(
            "{}",
            ppocr_i18n::tr!("OCR worker failed: {error}", error = error)
        );
        std::process::exit(1);
    }
}
