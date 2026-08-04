mod engine;
mod server;

fn main() {
    if let Err(error) = server::run_stdio(engine::OcrEngine::default()) {
        eprintln!("ppocr-worker failed: {error}");
        std::process::exit(1);
    }
}
