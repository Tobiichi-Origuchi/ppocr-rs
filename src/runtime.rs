use std::path::Path;

use ort::session::{Session, builder::GraphOptimizationLevel};

use crate::Result;

/// ONNX Runtime settings shared by all OCR modules.
#[derive(Debug, Clone)]
pub struct OrtSessionOptions {
    /// Intra-operator worker count. `None` keeps ONNX Runtime's default.
    pub intra_threads: Option<usize>,
    /// Inter-operator worker count. This matters only in parallel execution mode.
    pub inter_threads: Option<usize>,
    /// Allow independent graph branches to execute concurrently.
    pub parallel_execution: bool,
    /// ONNX Runtime graph optimization level.
    pub optimization_level: GraphOptimizationLevel,
}

impl Default for OrtSessionOptions {
    fn default() -> Self {
        Self {
            intra_threads: None,
            inter_threads: None,
            parallel_execution: false,
            optimization_level: GraphOptimizationLevel::All,
        }
    }
}

pub(crate) fn load_session(model_path: &Path, options: &OrtSessionOptions) -> Result<Session> {
    let mut builder = Session::builder()?
        .with_optimization_level(options.optimization_level)
        .map_err(ort::Error::from)?
        .with_parallel_execution(options.parallel_execution)
        .map_err(ort::Error::from)?
        // OCR inputs are dynamically resized; memory patterns assume stable shapes.
        .with_memory_pattern(false)
        .map_err(ort::Error::from)?;
    if let Some(threads) = options.intra_threads {
        builder = builder
            .with_intra_threads(threads)
            .map_err(ort::Error::from)?;
    }
    if let Some(threads) = options.inter_threads {
        builder = builder
            .with_inter_threads(threads)
            .map_err(ort::Error::from)?;
    }
    Ok(builder.commit_from_file(model_path)?)
}
