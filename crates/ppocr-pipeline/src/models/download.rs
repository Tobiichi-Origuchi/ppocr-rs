use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use hf_hub::HFClient;
use hf_hub::progress::{DownloadEvent, ProgressEvent, ProgressHandler};
use sha1::Sha1;
use sha2::{Digest as _, Sha256};
use thiserror::Error;

use super::manifest::{ArtifactStatus, ModelFile, manifest};

/// Controls lazy model retrieval.
#[derive(Debug, Clone, Default)]
pub struct ModelDownloaderOptions {
    /// Exact Hugging Face Hub cache directory. When omitted, environment and
    /// platform cache conventions are resolved by [`resolve_hf_hub_cache_dir`].
    pub cache_dir: Option<PathBuf>,
    /// Never access the network. A selected model must already exist in cache.
    pub local_files_only: bool,
    /// Re-download the selected model's required files even when cached.
    pub force_download: bool,
}

/// Downloads only files belonging to the explicitly selected model.
pub struct ModelDownloader {
    client: hf_hub::HFClientSync,
    options: ModelDownloaderOptions,
    cache_dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDownloadProgress {
    pub model_id: String,
    pub bytes_completed: u64,
    pub total_bytes: u64,
}

pub type ModelDownloadProgressCallback = Arc<dyn Fn(ModelDownloadProgress) + Send + Sync>;

struct DownloadProgressHandler {
    model_id: String,
    completed_before_file: u64,
    file_size: u64,
    total_bytes: u64,
    callback: ModelDownloadProgressCallback,
}

impl ProgressHandler for DownloadProgressHandler {
    fn on_progress(&self, event: &ProgressEvent) {
        let file_completed = match event {
            ProgressEvent::Download(DownloadEvent::Progress { files }) => {
                files.iter().map(|file| file.bytes_completed).max()
            }
            ProgressEvent::Download(DownloadEvent::AggregateProgress {
                bytes_completed, ..
            }) => Some(*bytes_completed),
            ProgressEvent::Download(DownloadEvent::Complete) => Some(self.file_size),
            _ => None,
        };
        if let Some(file_completed) = file_completed {
            (self.callback)(ModelDownloadProgress {
                model_id: self.model_id.clone(),
                bytes_completed: self
                    .completed_before_file
                    .saturating_add(file_completed.min(self.file_size)),
                total_bytes: self.total_bytes,
            });
        }
    }
}

impl std::fmt::Debug for ModelDownloader {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ModelDownloader")
            .field("cache_dir", &self.cache_dir)
            .field("local_files_only", &self.options.local_files_only)
            .field("force_download", &self.options.force_download)
            .finish_non_exhaustive()
    }
}

impl ModelDownloader {
    /// Build a downloader without creating the cache directory or contacting HF.
    pub fn new(options: ModelDownloaderOptions) -> Result<Self, ModelDownloadError> {
        let cache_dir = resolve_hf_hub_cache_dir(options.cache_dir.as_deref())?;
        let client = HFClient::builder()
            .cache_dir(cache_dir.clone())
            .build_sync()?;
        Ok(Self {
            client,
            options,
            cache_dir,
        })
    }

    /// The exact `hf-hub` cache directory selected for model files.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Lazily download and validate the two runtime files for one model ID.
    ///
    /// This calls `download_file` for each manifest-declared file. It never uses
    /// `snapshot_download` and therefore cannot fetch unrelated models or files.
    pub fn download(&self, model_id: &str) -> Result<DownloadedModel, ModelDownloadError> {
        self.download_with_progress(model_id, None)
    }

    pub fn download_with_progress(
        &self,
        model_id: &str,
        progress: Option<ModelDownloadProgressCallback>,
    ) -> Result<DownloadedModel, ModelDownloadError> {
        let spec = manifest()
            .get(model_id)
            .ok_or_else(|| ModelDownloadError::UnknownModel(model_id.to_owned()))?;

        if spec.artifact.status != ArtifactStatus::Ready {
            return Err(ModelDownloadError::ArtifactUnavailable {
                model_id: model_id.to_owned(),
                status: spec.artifact.status,
            });
        }

        let revision = spec.artifact.revision.as_deref().ok_or_else(|| {
            ModelDownloadError::InvalidManifest {
                model_id: model_id.to_owned(),
                message: "ready artifact has no immutable revision".to_owned(),
            }
        })?;
        let (owner, repository_name) = spec.artifact.repo_id.split_once('/').ok_or_else(|| {
            ModelDownloadError::InvalidManifest {
                model_id: model_id.to_owned(),
                message: format!(
                    "invalid Hugging Face repository id {:?}",
                    spec.artifact.repo_id
                ),
            }
        })?;
        let repository = self.client.model(owner, repository_name);

        let total_bytes = spec
            .artifact
            .required_files
            .iter()
            .map(|file| file.size)
            .sum();
        let mut completed_bytes = 0_u64;
        let mut files = BTreeMap::new();
        for expected in &spec.artifact.required_files {
            if let Some(callback) = progress.as_ref() {
                callback(ModelDownloadProgress {
                    model_id: model_id.to_owned(),
                    bytes_completed: completed_bytes,
                    total_bytes,
                });
            }
            let request = repository
                .download_file()
                .filename(expected.path.clone())
                .revision(revision.to_owned())
                .force_download(self.options.force_download)
                .local_files_only(self.options.local_files_only);
            let path = if let Some(callback) = progress.as_ref() {
                request
                    .progress(DownloadProgressHandler {
                        model_id: model_id.to_owned(),
                        completed_before_file: completed_bytes,
                        file_size: expected.size,
                        total_bytes,
                        callback: callback.clone(),
                    })
                    .send()?
            } else {
                request.send()?
            };
            verify_file(model_id, expected, &path)?;
            files.insert(expected.path.clone(), path);
            completed_bytes = completed_bytes.saturating_add(expected.size);
            if let Some(callback) = progress.as_ref() {
                callback(ModelDownloadProgress {
                    model_id: model_id.to_owned(),
                    bytes_completed: completed_bytes,
                    total_bytes,
                });
            }
        }

        let onnx_path = files.get("inference.onnx").cloned().ok_or_else(|| {
            ModelDownloadError::InvalidManifest {
                model_id: model_id.to_owned(),
                message: "artifact does not declare inference.onnx".to_owned(),
            }
        })?;
        let config_path = files.get("inference.yml").cloned().ok_or_else(|| {
            ModelDownloadError::InvalidManifest {
                model_id: model_id.to_owned(),
                message: "artifact does not declare inference.yml".to_owned(),
            }
        })?;

        Ok(DownloadedModel {
            model_id: model_id.to_owned(),
            repo_id: spec.artifact.repo_id.clone(),
            revision: revision.to_owned(),
            onnx_path,
            config_path,
            files,
        })
    }
}

/// Validated local paths for one selected model.
#[derive(Debug, Clone)]
pub struct DownloadedModel {
    pub model_id: String,
    pub repo_id: String,
    pub revision: String,
    pub onnx_path: PathBuf,
    pub config_path: PathBuf,
    pub files: BTreeMap<String, PathBuf>,
}

/// Resolve the exact Hub cache directory while explicitly honoring XDG.
///
/// Precedence: explicit path, `HF_HUB_CACHE`, legacy
/// `HUGGINGFACE_HUB_CACHE`, `HF_HOME/hub`,
/// `XDG_CACHE_HOME/huggingface/hub`, then the platform cache directory.
/// The final platform fallback is conventionally
/// `~/.cache/huggingface/hub` on Linux,
/// `~/Library/Caches/huggingface/hub` on macOS, and
/// `%LOCALAPPDATA%\huggingface\hub` on Windows.
pub fn resolve_hf_hub_cache_dir(explicit: Option<&Path>) -> Result<PathBuf, ModelDownloadError> {
    resolve_cache_dir_from(explicit, CacheEnvironment::read(), dirs::cache_dir())
}

#[derive(Debug, Default)]
struct CacheEnvironment {
    hf_hub_cache: Option<OsString>,
    legacy_hf_hub_cache: Option<OsString>,
    hf_home: Option<OsString>,
    xdg_cache_home: Option<OsString>,
}

impl CacheEnvironment {
    fn read() -> Self {
        Self {
            hf_hub_cache: std::env::var_os("HF_HUB_CACHE"),
            legacy_hf_hub_cache: std::env::var_os("HUGGINGFACE_HUB_CACHE"),
            hf_home: std::env::var_os("HF_HOME"),
            xdg_cache_home: std::env::var_os("XDG_CACHE_HOME"),
        }
    }
}

fn resolve_cache_dir_from(
    explicit: Option<&Path>,
    environment: CacheEnvironment,
    platform_cache: Option<PathBuf>,
) -> Result<PathBuf, ModelDownloadError> {
    if let Some(path) = explicit {
        return nonempty_path(path.as_os_str())
            .ok_or(ModelDownloadError::CacheDirectoryUnavailable);
    }
    if let Some(path) = environment.hf_hub_cache.and_then(nonempty_path) {
        return Ok(path);
    }
    if let Some(path) = environment.legacy_hf_hub_cache.and_then(nonempty_path) {
        return Ok(path);
    }
    if let Some(path) = environment.hf_home.and_then(nonempty_path) {
        return Ok(path.join("hub"));
    }
    if let Some(path) = environment.xdg_cache_home.and_then(nonempty_path) {
        return Ok(path.join("huggingface").join("hub"));
    }
    platform_cache
        .map(|path| path.join("huggingface").join("hub"))
        .ok_or(ModelDownloadError::CacheDirectoryUnavailable)
}

fn nonempty_path(value: impl AsRef<std::ffi::OsStr>) -> Option<PathBuf> {
    let value = value.as_ref();
    (!value.is_empty()).then(|| PathBuf::from(value))
}

fn verify_file(
    model_id: &str,
    expected: &ModelFile,
    path: &Path,
) -> Result<(), ModelDownloadError> {
    let actual_size = path.metadata()?.len();
    if actual_size != expected.size {
        return Err(ModelDownloadError::SizeMismatch {
            model_id: model_id.to_owned(),
            file: expected.path.clone(),
            expected: expected.size,
            actual: actual_size,
        });
    }

    if let Some(expected_sha256) = expected.sha256.as_deref() {
        let actual = sha256_file(path)?;
        if actual != expected_sha256 {
            return Err(ModelDownloadError::HashMismatch {
                model_id: model_id.to_owned(),
                file: expected.path.clone(),
                algorithm: "sha256",
                expected: expected_sha256.to_owned(),
                actual,
            });
        }
    } else if let Some(expected_git_oid) = expected.git_oid.as_deref() {
        let actual = git_blob_oid(path, actual_size)?;
        if actual != expected_git_oid {
            return Err(ModelDownloadError::HashMismatch {
                model_id: model_id.to_owned(),
                file: expected.path.clone(),
                algorithm: "git-sha1",
                expected: expected_git_oid.to_owned(),
                actual,
            });
        }
    }

    Ok(())
}

fn sha256_file(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    copy_into_digest(&mut file, &mut hasher)?;
    Ok(hex_encode(hasher.finalize().as_ref()))
}

fn git_blob_oid(path: &Path, size: u64) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha1::new();
    hasher.update(format!("blob {size}\0").as_bytes());
    copy_into_digest(&mut file, &mut hasher)?;
    Ok(hex_encode(hasher.finalize().as_ref()))
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

fn copy_into_digest<D: sha2::Digest>(reader: &mut impl Read, digest: &mut D) -> io::Result<()> {
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            return Ok(());
        }
        digest.update(&buffer[..read]);
    }
}

#[derive(Debug, Error)]
pub enum ModelDownloadError {
    #[error("unknown PaddleOCR model {0:?}")]
    UnknownModel(String),
    #[error("model {model_id:?} has no validated ONNX artifact yet ({status:?})")]
    ArtifactUnavailable {
        model_id: String,
        status: ArtifactStatus,
    },
    #[error("invalid manifest entry for model {model_id:?}: {message}")]
    InvalidManifest { model_id: String, message: String },
    #[error("no usable Hugging Face cache directory is available")]
    CacheDirectoryUnavailable,
    #[error(
        "downloaded file {file:?} for model {model_id:?} has size {actual}, expected {expected}"
    )]
    SizeMismatch {
        model_id: String,
        file: String,
        expected: u64,
        actual: u64,
    },
    #[error(
        "downloaded file {file:?} for model {model_id:?} failed {algorithm}: expected {expected}, got {actual}"
    )]
    HashMismatch {
        model_id: String,
        file: String,
        algorithm: &'static str,
        expected: String,
        actual: String,
    },
    #[error(transparent)]
    Hub(#[from] hf_hub::HFError),
    #[error(transparent)]
    Io(#[from] io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn download_progress_combines_completed_files_and_current_file() {
        let received = Arc::new(std::sync::Mutex::new(Vec::new()));
        let received_for_callback = received.clone();
        let handler = DownloadProgressHandler {
            model_id: "model".to_owned(),
            completed_before_file: 100,
            file_size: 400,
            total_bytes: 500,
            callback: Arc::new(move |progress| {
                received_for_callback.lock().unwrap().push(progress);
            }),
        };
        handler.on_progress(&ProgressEvent::Download(DownloadEvent::AggregateProgress {
            bytes_completed: 150,
            total_bytes: 400,
            bytes_per_sec: None,
        }));
        handler.on_progress(&ProgressEvent::Download(DownloadEvent::Complete));
        let received = received.lock().unwrap();
        assert_eq!(received[0].bytes_completed, 250);
        assert_eq!(received[0].total_bytes, 500);
        assert_eq!(received[1].bytes_completed, 500);
    }

    #[test]
    fn xdg_cache_is_resolved_explicitly() {
        let environment = CacheEnvironment {
            xdg_cache_home: Some(OsString::from("/var/tmp/xdg")),
            ..CacheEnvironment::default()
        };
        assert_eq!(
            resolve_cache_dir_from(None, environment, None).unwrap(),
            Path::new("/var/tmp/xdg/huggingface/hub")
        );
    }

    #[test]
    fn cache_precedence_is_deterministic() {
        let environment = CacheEnvironment {
            hf_hub_cache: Some(OsString::from("/cache/hub")),
            legacy_hf_hub_cache: Some(OsString::from("/legacy")),
            hf_home: Some(OsString::from("/hf-home")),
            xdg_cache_home: Some(OsString::from("/xdg")),
        };
        assert_eq!(
            resolve_cache_dir_from(Some(Path::new("/explicit")), environment, None).unwrap(),
            Path::new("/explicit")
        );
    }

    #[test]
    fn hf_home_and_platform_cache_roots_receive_the_hub_suffix() {
        let environment = CacheEnvironment {
            hf_home: Some(OsString::from("/portable/huggingface")),
            ..CacheEnvironment::default()
        };
        assert_eq!(
            resolve_cache_dir_from(None, environment, Some(PathBuf::from("/platform"))).unwrap(),
            Path::new("/portable/huggingface/hub")
        );

        assert_eq!(
            resolve_cache_dir_from(
                None,
                CacheEnvironment::default(),
                Some(PathBuf::from("/platform/cache")),
            )
            .unwrap(),
            Path::new("/platform/cache/huggingface/hub")
        );
    }

    #[test]
    fn constructing_downloader_is_lazy() {
        let temporary = tempfile::tempdir().unwrap();
        let cache = temporary.path().join("does-not-exist-yet");
        let downloader = ModelDownloader::new(ModelDownloaderOptions {
            cache_dir: Some(cache.clone()),
            local_files_only: true,
            force_download: false,
        })
        .unwrap();
        assert_eq!(downloader.cache_dir(), cache);
        assert!(!cache.exists());
    }

    #[test]
    fn unknown_model_fails_before_cache_or_network_access() {
        let temporary = tempfile::tempdir().unwrap();
        let cache = temporary.path().join("cache");
        let downloader = ModelDownloader::new(ModelDownloaderOptions {
            cache_dir: Some(cache.clone()),
            local_files_only: true,
            force_download: false,
        })
        .unwrap();
        let error = downloader.download("does-not-exist").unwrap_err();
        assert!(matches!(error, ModelDownloadError::UnknownModel(_)));
        assert!(!cache.exists());
    }

    #[test]
    fn integrity_verification_supports_sha256_and_git_blob_oid() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("small-file");
        std::fs::write(&path, b"abc").unwrap();

        let sha_file = ModelFile {
            path: "small-file".to_owned(),
            size: 3,
            sha256: Some(
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".to_owned(),
            ),
            git_oid: None,
        };
        verify_file("test", &sha_file, &path).unwrap();

        let git_file = ModelFile {
            path: "small-file".to_owned(),
            size: 3,
            sha256: None,
            git_oid: Some("f2ba8f84ab5c1bce84a7b441cb1959cfc7093b7f".to_owned()),
        };
        verify_file("test", &git_file, &path).unwrap();
    }

    #[test]
    #[ignore = "requires Hugging Face network access"]
    fn downloads_only_the_selected_official_model() {
        let temporary = tempfile::tempdir().unwrap();
        let downloader = ModelDownloader::new(ModelDownloaderOptions {
            cache_dir: Some(temporary.path().join("hub")),
            local_files_only: false,
            force_download: false,
        })
        .unwrap();

        let downloaded = downloader.download("PP-LCNet_x0_25_textline_ori").unwrap();

        assert_eq!(downloaded.files.len(), 2);
        assert!(downloaded.onnx_path.is_file());
        assert!(downloaded.config_path.is_file());
        assert!(
            downloaded
                .onnx_path
                .to_string_lossy()
                .contains("PaddlePaddle--PP-LCNet_x0_25_textline_ori_onnx")
        );
    }

    #[test]
    #[ignore = "requires Hugging Face network access"]
    fn downloads_a_selected_converted_model() {
        let temporary = tempfile::tempdir().unwrap();
        let downloader = ModelDownloader::new(ModelDownloaderOptions {
            cache_dir: Some(temporary.path().join("hub")),
            local_files_only: false,
            force_download: false,
        })
        .unwrap();

        let downloaded = downloader.download("PP-OCRv4_mobile_det").unwrap();

        assert_eq!(downloaded.files.len(), 2);
        assert!(downloaded.onnx_path.is_file());
        assert!(downloaded.config_path.is_file());
        assert!(
            downloaded
                .onnx_path
                .to_string_lossy()
                .contains("tobiichioriguchi--PP-OCRv4_mobile_det_onnx")
        );
    }
}
