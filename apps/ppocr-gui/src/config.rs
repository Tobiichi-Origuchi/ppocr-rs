use std::{
    fs, io,
    path::{Path, PathBuf},
};

use ppocr_app_protocol::PipelineConfig;
use serde::{Deserialize, Serialize};
use thiserror::Error;

const CONFIG_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppConfig {
    pub version: u32,
    pub pipeline: PipelineConfig,
    pub keep_worker_warm: bool,
    pub idle_timeout_minutes: u32,
    pub theme: ThemePreference,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            version: CONFIG_VERSION,
            pipeline: PipelineConfig::default(),
            keep_worker_warm: true,
            idle_timeout_minutes: 10,
            theme: ThemePreference::System,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThemePreference {
    #[default]
    System,
    Light,
    Dark,
}

impl AppConfig {
    pub fn load() -> Result<(Self, PathBuf), ConfigError> {
        let path = config_path()?;
        if !path.exists() {
            return Ok((Self::default(), path));
        }
        let bytes = fs::read(&path).map_err(|source| ConfigError::Read {
            path: path.clone(),
            source,
        })?;
        let config: Self = serde_json::from_slice(&bytes).map_err(|source| ConfigError::Parse {
            path: path.clone(),
            source,
        })?;
        if config.version != CONFIG_VERSION {
            return Err(ConfigError::UnsupportedVersion {
                path,
                version: config.version,
            });
        }
        Ok((config, path))
    }

    pub fn save(&self, path: &Path) -> Result<(), ConfigError> {
        let parent = path.parent().ok_or_else(|| ConfigError::NoParent {
            path: path.to_owned(),
        })?;
        fs::create_dir_all(parent).map_err(|source| ConfigError::CreateDirectory {
            path: parent.to_owned(),
            source,
        })?;
        let bytes = serde_json::to_vec_pretty(self).map_err(ConfigError::Serialize)?;
        fs::write(path, bytes).map_err(|source| ConfigError::Write {
            path: path.to_owned(),
            source,
        })
    }
}

pub fn config_path() -> Result<PathBuf, ConfigError> {
    config_path_from(std::env::var_os("XDG_CONFIG_HOME"), dirs::config_dir())
}

fn config_path_from(
    xdg_config_home: Option<std::ffi::OsString>,
    platform_config_dir: Option<PathBuf>,
) -> Result<PathBuf, ConfigError> {
    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    let base = xdg_config_home
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or(platform_config_dir);
    #[cfg(not(any(target_os = "linux", target_os = "freebsd")))]
    let base = {
        let _ = xdg_config_home;
        platform_config_dir
    };
    base.map(|path| path.join("ppocr-rs").join("config.json"))
        .ok_or(ConfigError::DirectoryUnavailable)
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("platform configuration directory is unavailable")]
    DirectoryUnavailable,
    #[error("configuration path {path:?} has no parent directory")]
    NoParent { path: PathBuf },
    #[error("failed to create configuration directory {path:?}: {source}")]
    CreateDirectory { path: PathBuf, source: io::Error },
    #[error("failed to read configuration {path:?}: {source}")]
    Read { path: PathBuf, source: io::Error },
    #[error("failed to parse configuration {path:?}: {source}")]
    Parse {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("configuration {path:?} uses unsupported version {version}")]
    UnsupportedVersion { path: PathBuf, version: u32 },
    #[error("failed to serialize configuration: {0}")]
    Serialize(serde_json::Error),
    #[error("failed to write configuration {path:?}: {source}")]
    Write { path: PathBuf, source: io::Error },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linux_xdg_path_wins_without_dconf() {
        let path = config_path_from(
            Some("/tmp/explicit-xdg".into()),
            Some(PathBuf::from("/tmp/platform-config")),
        )
        .unwrap();
        #[cfg(any(target_os = "linux", target_os = "freebsd"))]
        assert_eq!(path, Path::new("/tmp/explicit-xdg/ppocr-rs/config.json"));
        #[cfg(not(any(target_os = "linux", target_os = "freebsd")))]
        assert_eq!(path, Path::new("/tmp/platform-config/ppocr-rs/config.json"));
    }

    #[test]
    fn configuration_round_trips_as_a_normal_json_file() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("nested").join("config.json");
        let mut expected = AppConfig::default();
        expected.pipeline.text_detection_model = "PP-OCRv6_tiny_det".to_owned();
        expected.theme = ThemePreference::Dark;
        expected.save(&path).unwrap();
        let bytes = fs::read(&path).unwrap();
        let actual: AppConfig = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(actual, expected);
        assert!(
            String::from_utf8(bytes)
                .unwrap()
                .contains("PP-OCRv6_tiny_det")
        );
    }
}
