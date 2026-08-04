use std::collections::HashSet;
use std::sync::LazyLock;

use serde::Deserialize;

const EMBEDDED_MANIFEST: &str = include_str!("../../models/manifest.json");

static MANIFEST: LazyLock<ModelManifest> = LazyLock::new(|| {
    let manifest: ModelManifest = serde_json::from_str(EMBEDDED_MANIFEST)
        .expect("the embedded PaddleOCR model manifest must be valid JSON");
    manifest
        .validate()
        .expect("the embedded PaddleOCR model manifest must be internally consistent");
    manifest
});

/// Return the embedded, versioned model manifest.
pub fn manifest() -> &'static ModelManifest {
    &MANIFEST
}

/// Versioned inventory of every model supported by the general OCR pipeline.
#[derive(Debug, Deserialize)]
pub struct ModelManifest {
    pub schema_version: u32,
    pub generated_at: String,
    pub sources: ManifestSources,
    pub models: Vec<ModelSpec>,
}

impl ModelManifest {
    /// Find one model without initiating any filesystem or network activity.
    pub fn get(&self, model_id: &str) -> Option<&ModelSpec> {
        self.models.iter().find(|model| model.id == model_id)
    }

    fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1 {
            return Err(format!(
                "unsupported model manifest schema version {}",
                self.schema_version
            ));
        }

        let mut ids = HashSet::with_capacity(self.models.len());
        for model in &self.models {
            if !ids.insert(model.id.as_str()) {
                return Err(format!("duplicate model id {:?}", model.id));
            }

            if model.artifact.status == ArtifactStatus::Ready {
                let revision = model
                    .artifact
                    .revision
                    .as_deref()
                    .ok_or_else(|| format!("ready model {:?} has no revision", model.id))?;
                if revision.len() != 40 || !revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                    return Err(format!(
                        "ready model {:?} does not use an immutable commit revision",
                        model.id
                    ));
                }

                if model.artifact.required_files.len() != 2 {
                    return Err(format!(
                        "ready model {:?} must declare exactly two runtime files",
                        model.id
                    ));
                }

                for required in ["inference.onnx", "inference.yml"] {
                    if !model
                        .artifact
                        .required_files
                        .iter()
                        .any(|file| file.path == required)
                    {
                        return Err(format!(
                            "ready model {:?} is missing required file {:?}",
                            model.id, required
                        ));
                    }
                }

                for file in &model.artifact.required_files {
                    if file.size == 0 {
                        return Err(format!(
                            "ready model {:?} declares an empty file {:?}",
                            model.id, file.path
                        ));
                    }
                    if file.sha256.is_none() && file.git_oid.is_none() {
                        return Err(format!(
                            "ready model {:?} has no integrity hash for {:?}",
                            model.id, file.path
                        ));
                    }
                    if let Some(hash) = &file.sha256
                        && !is_hex_digest(hash, 64)
                    {
                        return Err(format!(
                            "ready model {:?} has an invalid SHA-256 for {:?}",
                            model.id, file.path
                        ));
                    }
                    if let Some(hash) = &file.git_oid
                        && !is_hex_digest(hash, 40)
                    {
                        return Err(format!(
                            "ready model {:?} has an invalid Git OID for {:?}",
                            model.id, file.path
                        ));
                    }
                }

                if model.artifact.origin == ArtifactOrigin::LocallyConverted
                    && model.artifact.conversion.is_none()
                {
                    return Err(format!(
                        "converted model {:?} has no conversion provenance",
                        model.id
                    ));
                }
            }
        }
        Ok(())
    }
}

fn is_hex_digest(value: &str, length: usize) -> bool {
    value.len() == length && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[derive(Debug, Deserialize)]
pub struct ManifestSources {
    pub paddleocr_commit: String,
    pub paddlex_commit: String,
    pub paddleocr_pipeline_url: String,
    pub paddlex_onnx_registry_url: String,
}

/// One model selectable by a pipeline module.
#[derive(Debug, Deserialize)]
pub struct ModelSpec {
    pub id: String,
    pub kind: ModelKind,
    pub runtime_profile: String,
    pub variant: Option<String>,
    pub paddle_inference_url: String,
    pub artifact: ModelArtifact,
}

/// The five module roles in PaddleOCR's general OCR pipeline.
#[derive(Debug, Clone, Copy, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    DocumentOrientation,
    DocumentRectification,
    TextlineOrientation,
    TextDetection,
    TextRecognition,
}

/// Immutable Hugging Face artifact used by the Rust runtime.
#[derive(Debug, Deserialize)]
pub struct ModelArtifact {
    pub origin: ArtifactOrigin,
    pub status: ArtifactStatus,
    pub repo_id: String,
    pub revision: Option<String>,
    pub license: String,
    pub required_files: Vec<ModelFile>,
    pub conversion: Option<ModelConversion>,
}

/// Reproducibility and numerical validation metadata for a local conversion.
#[derive(Debug, Deserialize)]
pub struct ModelConversion {
    pub source_archive_size: u64,
    pub source_archive_sha256: String,
    pub paddle_version: String,
    pub paddlex_version: String,
    pub paddle2onnx_version: String,
    pub onnx_version: String,
    pub onnxruntime_version: String,
    pub opset: u32,
    pub validation: ConversionValidation,
}

#[derive(Debug, Deserialize)]
pub struct ConversionValidation {
    pub onnx_checker: bool,
    pub ort_cpu_execution: bool,
    pub paddle_output_parity: bool,
    pub input_shape: [usize; 4],
    pub max_abs_error: f64,
}

#[derive(Debug, Clone, Copy, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactOrigin {
    PaddlepaddleOfficial,
    LocallyConverted,
}

#[derive(Debug, Clone, Copy, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactStatus {
    Ready,
    PendingConversion,
}

/// Integrity metadata for one file that must be downloaded for a selected model.
#[derive(Debug, Deserialize)]
pub struct ModelFile {
    pub path: String,
    pub size: u64,
    pub sha256: Option<String>,
    pub git_oid: Option<String>,
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn manifest_contains_the_complete_pipeline_inventory() {
        let manifest = manifest();
        assert_eq!(manifest.models.len(), 45);

        let ready = manifest
            .models
            .iter()
            .filter(|model| model.artifact.status == ArtifactStatus::Ready)
            .count();
        let pending = manifest.models.len() - ready;
        assert_eq!((ready, pending), (45, 0));

        let counts = manifest
            .models
            .iter()
            .fold([0_usize; 5], |mut counts, model| {
                let index = match model.kind {
                    ModelKind::DocumentOrientation => 0,
                    ModelKind::DocumentRectification => 1,
                    ModelKind::TextlineOrientation => 2,
                    ModelKind::TextDetection => 3,
                    ModelKind::TextRecognition => 4,
                };
                counts[index] += 1;
                counts
            });
        assert_eq!(counts, [1, 1, 2, 7, 34]);
    }

    #[test]
    fn model_ids_and_repositories_are_well_formed() {
        let manifest = manifest();
        let ids: HashSet<_> = manifest.models.iter().map(|model| &model.id).collect();
        assert_eq!(ids.len(), manifest.models.len());

        for model in &manifest.models {
            assert!(model.artifact.repo_id.contains('/'));
            assert!(model.paddle_inference_url.starts_with("https://"));
            if model.artifact.status == ArtifactStatus::Ready {
                assert_eq!(model.artifact.required_files.len(), 2);
            }
        }

        assert_eq!(
            manifest
                .models
                .iter()
                .filter(|model| model.artifact.conversion.is_some())
                .count(),
            20
        );
    }
}
