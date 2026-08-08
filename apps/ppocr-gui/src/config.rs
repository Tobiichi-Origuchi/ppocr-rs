use std::{
    fs, io,
    path::{Path, PathBuf},
};

use ppocr_app_protocol::PipelineConfig;
use ppocr_i18n::tr;
use serde::{Deserialize, Serialize};

const CONFIG_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppConfig {
    pub version: u32,
    pub pipeline: PipelineConfig,
    pub keep_worker_warm: bool,
    pub idle_timeout_minutes: u32,
    pub theme: ThemePreference,
    #[serde(default)]
    pub immediate_processing: bool,
    #[serde(default = "default_screenshot_delay_ms")]
    pub screenshot_delay_ms: u32,
    #[serde(default)]
    pub shortcuts: ShortcutConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            version: CONFIG_VERSION,
            pipeline: PipelineConfig::default(),
            keep_worker_warm: true,
            idle_timeout_minutes: 10,
            theme: ThemePreference::System,
            immediate_processing: false,
            screenshot_delay_ms: default_screenshot_delay_ms(),
            shortcuts: ShortcutConfig::default(),
        }
    }
}

const fn default_screenshot_delay_ms() -> u32 {
    500
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ShortcutConfig {
    pub open_image: String,
    pub paste_image: String,
    pub capture_region: String,
    pub capture_screen: String,
    pub capture_window: String,
    pub start_processing: String,
    pub copy_all: String,
    pub toggle_results: String,
    pub preferences: String,
}

impl Default for ShortcutConfig {
    fn default() -> Self {
        Self {
            open_image: "<Primary>O".to_owned(),
            paste_image: "<Primary>V".to_owned(),
            capture_region: "<Primary><Shift>O".to_owned(),
            capture_screen: String::new(),
            capture_window: String::new(),
            start_processing: "<Primary>Return".to_owned(),
            copy_all: "<Primary><Shift>C".to_owned(),
            toggle_results: String::new(),
            preferences: "<Primary>comma".to_owned(),
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

#[derive(Debug)]
pub enum ConfigError {
    DirectoryUnavailable,
    NoParent {
        path: PathBuf,
    },
    CreateDirectory {
        path: PathBuf,
        source: io::Error,
    },
    Read {
        path: PathBuf,
        source: io::Error,
    },
    Parse {
        path: PathBuf,
        source: serde_json::Error,
    },
    UnsupportedVersion {
        path: PathBuf,
        version: u32,
    },
    Serialize(serde_json::Error),
    Write {
        path: PathBuf,
        source: io::Error,
    },
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::DirectoryUnavailable => {
                tr!("The platform configuration directory is unavailable")
            }
            Self::NoParent { path } => tr!(
                "Configuration path {path} has no parent directory",
                path = path.display()
            ),
            Self::CreateDirectory { path, source } => tr!(
                "Failed to create configuration directory {path}: {error}",
                path = path.display(),
                error = source
            ),
            Self::Read { path, source } => tr!(
                "Failed to read configuration {path}: {error}",
                path = path.display(),
                error = source
            ),
            Self::Parse { path, source } => tr!(
                "Failed to parse configuration {path}: {error}",
                path = path.display(),
                error = source
            ),
            Self::UnsupportedVersion { path, version } => tr!(
                "Configuration {path} uses unsupported version {version}",
                path = path.display(),
                version = version
            ),
            Self::Serialize(error) => {
                tr!(
                    "Failed to serialize the configuration: {error}",
                    error = error
                )
            }
            Self::Write { path, source } => tr!(
                "Failed to write configuration {path}: {error}",
                path = path.display(),
                error = source
            ),
        };
        formatter.write_str(&message)
    }
}

impl std::error::Error for ConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::CreateDirectory { source, .. }
            | Self::Read { source, .. }
            | Self::Write { source, .. } => Some(source),
            Self::Parse { source, .. } | Self::Serialize(source) => Some(source),
            Self::DirectoryUnavailable
            | Self::NoParent { .. }
            | Self::UnsupportedVersion { .. } => None,
        }
    }
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

    #[test]
    fn new_interaction_settings_have_safe_defaults_in_existing_config_files() {
        let legacy = serde_json::json!({
            "version": 1,
            "pipeline": PipelineConfig::default(),
            "keep_worker_warm": true,
            "idle_timeout_minutes": 10,
            "theme": "system"
        });
        let config: AppConfig = serde_json::from_value(legacy).unwrap();
        assert!(!config.immediate_processing);
        assert_eq!(config.screenshot_delay_ms, 500);
        assert_eq!(config.shortcuts.paste_image, "<Primary>V");
        assert_eq!(config.shortcuts.start_processing, "<Primary>Return");
    }
}
