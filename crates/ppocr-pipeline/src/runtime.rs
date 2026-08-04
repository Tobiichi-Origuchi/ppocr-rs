use std::path::Path;

use ort::session::Session;

pub use ort::ep as execution_providers;
pub use ort::ep::ExecutionProviderDispatch;
pub use ort::session::builder::GraphOptimizationLevel;

use crate::Result;

/// ONNX Runtime settings shared by all OCR modules.
#[derive(Debug, Clone)]
pub struct OrtSessionOptions {
    /// Execution providers in descending priority order.
    ///
    /// An empty list uses ONNX Runtime's CPU provider. A dispatch created by
    /// `ort` falls back silently by default; call
    /// [`ExecutionProviderDispatch::error_on_failure`] when acceleration is
    /// required instead of optional.
    pub execution_providers: Vec<ExecutionProviderDispatch>,
    /// Intra-operator worker count. `None` keeps ONNX Runtime's default.
    pub intra_threads: Option<usize>,
    /// Inter-operator worker count. This matters only in parallel execution mode.
    pub inter_threads: Option<usize>,
    /// Allow independent graph branches to execute concurrently.
    ///
    /// Keep this `false` for DirectML, which rejects parallel graph execution.
    pub parallel_execution: bool,
    /// ONNX Runtime graph optimization level.
    pub optimization_level: GraphOptimizationLevel,
}

impl Default for OrtSessionOptions {
    fn default() -> Self {
        Self {
            execution_providers: Vec::new(),
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
    if !options.execution_providers.is_empty() {
        builder = builder
            .with_execution_providers(options.execution_providers.clone())
            .map_err(ort::Error::from)?;
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_is_the_default_and_explicit_provider_policy_is_preserved() {
        let default = OrtSessionOptions::default();
        assert!(default.execution_providers.is_empty());

        let cpu = execution_providers::CPU::default()
            .build()
            .error_on_failure();
        let configured = OrtSessionOptions {
            execution_providers: vec![cpu],
            ..OrtSessionOptions::default()
        };
        assert_eq!(configured.execution_providers.len(), 1);
        assert!(
            format!("{:?}", configured.execution_providers[0]).contains("CPUExecutionProvider")
        );
    }
}
