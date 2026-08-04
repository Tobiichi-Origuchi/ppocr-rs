use std::{cell::Cell, path::PathBuf, rc::Rc};

use adw::prelude::*;

use crate::config::{AppConfig, ThemePreference};
use crate::data::{
    DETECTION_MODELS, DOCUMENT_ORIENTATION_MODELS, DOCUMENT_RECTIFICATION_MODELS,
    RECOGNITION_MODELS, RUNTIME_PROVIDERS, TEXTLINE_ORIENTATION_MODELS,
};
use ppocr_app_protocol::{PipelineConfig, ProviderConfig};

#[derive(Debug)]
pub struct SettingsState {
    pub detector: Cell<u32>,
    pub recognizer: Cell<u32>,
    pub document_orientation_enabled: Cell<bool>,
    pub document_orientation_model: Cell<u32>,
    pub document_rectification_enabled: Cell<bool>,
    pub document_rectification_model: Cell<u32>,
    pub textline_orientation_enabled: Cell<bool>,
    pub textline_orientation_model: Cell<u32>,
    pub return_word_boxes: Cell<bool>,
    pub provider: Cell<u32>,
    pub confidence_percent: Cell<f64>,
    pub keep_worker_warm: Cell<bool>,
    pub idle_timeout_minutes: Cell<f64>,
    pub theme: Cell<u32>,
    base_pipeline: PipelineConfig,
    config_path: Option<PathBuf>,
}

impl Default for SettingsState {
    fn default() -> Self {
        Self::from_config(AppConfig::default(), None)
    }
}

impl SettingsState {
    pub fn load() -> (Self, Option<String>) {
        match AppConfig::load() {
            Ok((config, path)) => (Self::from_config(config, Some(path)), None),
            Err(error) => (
                Self::from_config(AppConfig::default(), crate::config::config_path().ok()),
                Some(format!("配置文件读取失败，已使用默认值：{error}")),
            ),
        }
    }

    fn from_config(config: AppConfig, config_path: Option<PathBuf>) -> Self {
        let pipeline = config.pipeline;
        let document_orientation_enabled = pipeline.document_orientation_model.is_some();
        let document_rectification_enabled = pipeline.document_rectification_model.is_some();
        let textline_orientation_enabled = pipeline.textline_orientation_model.is_some();
        let detector = model_index(DETECTION_MODELS, Some(&pipeline.text_detection_model));
        let recognizer = model_index(RECOGNITION_MODELS, Some(&pipeline.text_recognition_model));
        let document_orientation_model = model_index(
            DOCUMENT_ORIENTATION_MODELS,
            pipeline.document_orientation_model.as_deref(),
        );
        let document_rectification_model = model_index(
            DOCUMENT_RECTIFICATION_MODELS,
            pipeline.document_rectification_model.as_deref(),
        );
        let textline_orientation_model = model_index(
            TEXTLINE_ORIENTATION_MODELS,
            pipeline.textline_orientation_model.as_deref(),
        );
        let provider = match pipeline.provider {
            ProviderConfig::Cpu => 0,
            ProviderConfig::Cuda { .. } => 1,
            ProviderConfig::DirectMl { .. } => 2,
            ProviderConfig::CoreMl => 3,
            ProviderConfig::WebGpu { .. } => 4,
            ProviderConfig::MiGraphX { .. } => 5,
        };
        let theme = match config.theme {
            ThemePreference::System => 0,
            ThemePreference::Light => 1,
            ThemePreference::Dark => 2,
        };
        Self {
            detector: Cell::new(detector),
            recognizer: Cell::new(recognizer),
            document_orientation_enabled: Cell::new(document_orientation_enabled),
            document_orientation_model: Cell::new(document_orientation_model),
            document_rectification_enabled: Cell::new(document_rectification_enabled),
            document_rectification_model: Cell::new(document_rectification_model),
            textline_orientation_enabled: Cell::new(textline_orientation_enabled),
            textline_orientation_model: Cell::new(textline_orientation_model),
            return_word_boxes: Cell::new(pipeline.return_word_boxes),
            provider: Cell::new(provider),
            confidence_percent: Cell::new(f64::from(
                pipeline.text_recognition_score_threshold * 100.0,
            )),
            keep_worker_warm: Cell::new(config.keep_worker_warm),
            idle_timeout_minutes: Cell::new(f64::from(config.idle_timeout_minutes)),
            theme: Cell::new(theme),
            base_pipeline: pipeline,
            config_path,
        }
    }

    pub fn pipeline_config(&self) -> PipelineConfig {
        let mut pipeline = self.base_pipeline.clone();
        pipeline.text_detection_model = selected_model(DETECTION_MODELS, self.detector.get());
        pipeline.text_recognition_model = selected_model(RECOGNITION_MODELS, self.recognizer.get());
        pipeline.document_orientation_model = self.document_orientation_enabled.get().then(|| {
            selected_model(
                DOCUMENT_ORIENTATION_MODELS,
                self.document_orientation_model.get(),
            )
        });
        pipeline.document_rectification_model =
            self.document_rectification_enabled.get().then(|| {
                selected_model(
                    DOCUMENT_RECTIFICATION_MODELS,
                    self.document_rectification_model.get(),
                )
            });
        pipeline.textline_orientation_model = self.textline_orientation_enabled.get().then(|| {
            selected_model(
                TEXTLINE_ORIENTATION_MODELS,
                self.textline_orientation_model.get(),
            )
        });
        pipeline.provider = match self.provider.get() {
            1 => ProviderConfig::Cuda { device_id: 0 },
            2 => ProviderConfig::DirectMl { device_id: 0 },
            3 => ProviderConfig::CoreMl,
            4 => ProviderConfig::WebGpu { device_id: 0 },
            5 => ProviderConfig::MiGraphX { device_id: 0 },
            _ => ProviderConfig::Cpu,
        };
        pipeline.text_recognition_score_threshold = (self.confidence_percent.get() / 100.0) as f32;
        pipeline.return_word_boxes = self.return_word_boxes.get();
        pipeline
    }

    pub fn persist(&self) {
        let Some(path) = self.config_path.as_deref() else {
            eprintln!("configuration directory is unavailable; settings were not saved");
            return;
        };
        let theme = match self.theme.get() {
            1 => ThemePreference::Light,
            2 => ThemePreference::Dark,
            _ => ThemePreference::System,
        };
        let config = AppConfig {
            version: 1,
            pipeline: self.pipeline_config(),
            keep_worker_warm: self.keep_worker_warm.get(),
            idle_timeout_minutes: self.idle_timeout_minutes.get().round() as u32,
            theme,
        };
        if let Err(error) = config.save(path) {
            eprintln!("failed to persist configuration: {error}");
        }
    }
}

fn model_index(models: &[&str], selected: Option<&str>) -> u32 {
    selected
        .and_then(|selected| models.iter().position(|model| *model == selected))
        .unwrap_or(0) as u32
}

fn selected_model(models: &[&str], selected: u32) -> String {
    models
        .get(selected as usize)
        .copied()
        .unwrap_or(models[0])
        .to_owned()
}

pub fn present_settings(
    parent: &adw::ApplicationWindow,
    state: Rc<SettingsState>,
    runtime_summary: &gtk::Label,
) {
    let dialog = adw::PreferencesDialog::builder()
        .title("设置")
        .search_enabled(true)
        .content_width(680)
        .content_height(720)
        .build();

    dialog.add(&models_page(state.clone(), runtime_summary));
    dialog.add(&runtime_page(state.clone(), runtime_summary));
    dialog.add(&interface_page(state));
    dialog.present(Some(parent));
}

fn models_page(state: Rc<SettingsState>, runtime_summary: &gtk::Label) -> adw::PreferencesPage {
    let page = adw::PreferencesPage::builder()
        .name("models")
        .title("模型与产线")
        .icon_name("system-run-symbolic")
        .description("选择必要 OCR 模型，并按需启用三个前处理模块")
        .build();

    let required = adw::PreferencesGroup::builder()
        .title("必要模型")
        .description("文本检测和文本识别始终启用")
        .build();
    let detector = combo_row(
        "文本检测模型",
        "定位图像中的文本区域",
        DETECTION_MODELS,
        state.detector.get(),
        true,
    );
    let recognizer = combo_row(
        "文本识别模型",
        "将裁剪后的文本行转换为字符",
        RECOGNITION_MODELS,
        state.recognizer.get(),
        true,
    );
    required.add(&detector);
    required.add(&recognizer);
    page.add(&required);

    let state_for_detector = state.clone();
    let summary_for_detector = runtime_summary.clone();
    detector.connect_selected_notify(move |row| {
        state_for_detector.detector.set(row.selected());
        update_runtime_summary(&state_for_detector, &summary_for_detector);
        state_for_detector.persist();
    });
    let state_for_recognizer = state.clone();
    recognizer.connect_selected_notify(move |row| {
        state_for_recognizer.recognizer.set(row.selected());
        state_for_recognizer.persist();
    });

    let optional = adw::PreferencesGroup::builder()
        .title("可选模块")
        .description("每个模块都可以独立开关和选择对应模型")
        .build();

    add_optional_module(
        &optional,
        "文档图像方向分类",
        "检测整页 0°、90°、180° 或 270° 旋转",
        DOCUMENT_ORIENTATION_MODELS,
        state.document_orientation_enabled.get(),
        state.document_orientation_model.get(),
        {
            let state = state.clone();
            move |active| {
                state.document_orientation_enabled.set(active);
                state.persist();
            }
        },
        {
            let state = state.clone();
            move |selected| {
                state.document_orientation_model.set(selected);
                state.persist();
            }
        },
    );
    add_optional_module(
        &optional,
        "文本图像矫正",
        "使用 UVDoc 修正文档透视与弯曲形变",
        DOCUMENT_RECTIFICATION_MODELS,
        state.document_rectification_enabled.get(),
        state.document_rectification_model.get(),
        {
            let state = state.clone();
            move |active| {
                state.document_rectification_enabled.set(active);
                state.persist();
            }
        },
        {
            let state = state.clone();
            move |selected| {
                state.document_rectification_model.set(selected);
                state.persist();
            }
        },
    );
    add_optional_module(
        &optional,
        "文本行方向分类",
        "在识别前校正单行文本方向",
        TEXTLINE_ORIENTATION_MODELS,
        state.textline_orientation_enabled.get(),
        state.textline_orientation_model.get(),
        {
            let state = state.clone();
            move |active| {
                state.textline_orientation_enabled.set(active);
                state.persist();
            }
        },
        {
            let state = state.clone();
            move |selected| {
                state.textline_orientation_model.set(selected);
                state.persist();
            }
        },
    );
    page.add(&optional);

    page
}

fn runtime_page(state: Rc<SettingsState>, runtime_summary: &gtk::Label) -> adw::PreferencesPage {
    let page = adw::PreferencesPage::builder()
        .name("runtime")
        .title("运行与输出")
        .icon_name("preferences-system-symbolic")
        .description("设备可用性将在 IPC 后端连接后验证")
        .build();

    let inference = adw::PreferencesGroup::builder().title("推理").build();
    let provider = combo_row(
        "推理设备",
        "不可用的 Provider 不会静默回退",
        RUNTIME_PROVIDERS,
        state.provider.get(),
        false,
    );
    inference.add(&provider);

    let confidence = adw::SpinRow::with_range(0.0, 100.0, 1.0);
    confidence.set_title("最低置信度");
    confidence.set_subtitle("低于该分数的识别文本将被过滤");
    confidence.set_value(state.confidence_percent.get());
    confidence.set_digits(0);
    inference.add(&confidence);
    page.add(&inference);

    let state_for_provider = state.clone();
    let summary_for_provider = runtime_summary.clone();
    provider.connect_selected_notify(move |row| {
        state_for_provider.provider.set(row.selected());
        update_runtime_summary(&state_for_provider, &summary_for_provider);
        state_for_provider.persist();
    });
    let state_for_confidence = state.clone();
    confidence.connect_value_notify(move |row| {
        state_for_confidence.confidence_percent.set(row.value());
        state_for_confidence.persist();
    });

    let lifecycle = adw::PreferencesGroup::builder().title("识别进程").build();
    let keep_warm = adw::SwitchRow::builder()
        .title("保持识别进程常驻")
        .subtitle("减少连续识别时的模型加载等待")
        .active(state.keep_worker_warm.get())
        .build();
    let idle = adw::SpinRow::with_range(1.0, 120.0, 1.0);
    idle.set_title("空闲退出时间");
    idle.set_subtitle("分钟");
    idle.set_value(state.idle_timeout_minutes.get());
    idle.set_digits(0);
    idle.set_sensitive(keep_warm.is_active());
    lifecycle.add(&keep_warm);
    lifecycle.add(&idle);
    page.add(&lifecycle);

    let state_for_keep = state.clone();
    let idle_for_switch = idle.clone();
    keep_warm.connect_active_notify(move |row| {
        state_for_keep.keep_worker_warm.set(row.is_active());
        idle_for_switch.set_sensitive(row.is_active());
        state_for_keep.persist();
    });
    let state_for_idle = state.clone();
    idle.connect_value_notify(move |row| {
        state_for_idle.idle_timeout_minutes.set(row.value());
        state_for_idle.persist();
    });

    let output = adw::PreferencesGroup::builder().title("结果").build();
    let word_boxes = adw::SwitchRow::builder()
        .title("返回单词级文本框")
        .subtitle("仅在支持字符对齐的识别模型上生成")
        .active(state.return_word_boxes.get())
        .build();
    output.add(&word_boxes);
    page.add(&output);
    word_boxes.connect_active_notify(move |row| {
        state.return_word_boxes.set(row.is_active());
        state.persist();
    });

    page
}

fn interface_page(state: Rc<SettingsState>) -> adw::PreferencesPage {
    let page = adw::PreferencesPage::builder()
        .name("interface")
        .title("界面")
        .icon_name("applications-graphics-symbolic")
        .build();

    let appearance = adw::PreferencesGroup::builder().title("外观").build();
    let theme = combo_row(
        "主题",
        "GTK 会跟随桌面配色和辅助功能设置",
        &["跟随系统", "浅色", "深色"],
        state.theme.get(),
        false,
    );
    appearance.add(&theme);
    page.add(&appearance);

    theme.connect_selected_notify(move |row| {
        state.theme.set(row.selected());
        let color_scheme = match row.selected() {
            1 => adw::ColorScheme::ForceLight,
            2 => adw::ColorScheme::ForceDark,
            _ => adw::ColorScheme::Default,
        };
        adw::StyleManager::default().set_color_scheme(color_scheme);
        state.persist();
    });

    let shortcuts = adw::PreferencesGroup::builder().title("快捷键").build();
    shortcuts.add(&shortcut_row("截图识别", "<Ctrl><Shift>O"));
    shortcuts.add(&shortcut_row("打开图片", "<Ctrl>O"));
    shortcuts.add(&shortcut_row("打开设置", "<Ctrl>comma"));
    shortcuts.add(&shortcut_row("复制全部文本", "<Ctrl><Shift>C"));
    page.add(&shortcuts);

    page
}

fn combo_row(
    title: &str,
    subtitle: &str,
    items: &[&str],
    selected: u32,
    enable_search: bool,
) -> adw::ComboRow {
    let model = gtk::StringList::new(items);
    adw::ComboRow::builder()
        .title(title)
        .subtitle(subtitle)
        .model(&model)
        .selected(selected.min(items.len().saturating_sub(1) as u32))
        .enable_search(enable_search)
        .build()
}

#[allow(clippy::too_many_arguments)]
fn add_optional_module(
    group: &adw::PreferencesGroup,
    title: &str,
    subtitle: &str,
    models: &[&str],
    enabled: bool,
    selected: u32,
    on_enabled: impl Fn(bool) + 'static,
    on_selected: impl Fn(u32) + 'static,
) {
    let switch = adw::SwitchRow::builder()
        .title(title)
        .subtitle(subtitle)
        .active(enabled)
        .build();
    let model = combo_row("模型", "选择该模块使用的模型", models, selected, false);
    model.set_sensitive(enabled);
    group.add(&switch);
    group.add(&model);

    let model_for_switch = model.clone();
    switch.connect_active_notify(move |row| {
        model_for_switch.set_sensitive(row.is_active());
        on_enabled(row.is_active());
    });
    model.connect_selected_notify(move |row| on_selected(row.selected()));
}

fn shortcut_row(title: &str, accelerator: &str) -> adw::ActionRow {
    let row = adw::ActionRow::builder().title(title).build();
    let label = gtk::ShortcutLabel::new(accelerator);
    row.add_suffix(&label);
    row
}

pub fn update_runtime_summary(state: &SettingsState, label: &gtk::Label) {
    let provider = RUNTIME_PROVIDERS
        .get(state.provider.get() as usize)
        .copied()
        .unwrap_or(RUNTIME_PROVIDERS[0]);
    let detector = DETECTION_MODELS
        .get(state.detector.get() as usize)
        .copied()
        .unwrap_or(DETECTION_MODELS[0]);
    label.set_label(&format!("{provider}  ·  {detector}"));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn changed_settings_persist_the_selected_pipeline_as_json() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("ppocr-rs").join("config.json");
        let state = SettingsState::from_config(AppConfig::default(), Some(path.clone()));
        state.detector.set(2);
        state.document_rectification_enabled.set(false);
        state.return_word_boxes.set(false);
        state.keep_worker_warm.set(false);
        state.theme.set(2);

        state.persist();

        let persisted: AppConfig = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(persisted.pipeline.text_detection_model, "PP-OCRv6_tiny_det");
        assert_eq!(persisted.pipeline.document_rectification_model, None);
        assert!(!persisted.pipeline.return_word_boxes);
        assert!(!persisted.keep_worker_warm);
        assert_eq!(persisted.theme, ThemePreference::Dark);
    }
}
