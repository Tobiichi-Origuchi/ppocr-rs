use std::{
    cell::{Cell, RefCell},
    path::PathBuf,
    rc::Rc,
};

use adw::prelude::*;

use crate::config::{AppConfig, ShortcutConfig, ThemePreference};
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
    pub immediate_processing: Cell<bool>,
    pub screenshot_delay_ms: Cell<f64>,
    pub shortcuts: RefCell<ShortcutConfig>,
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
            immediate_processing: Cell::new(config.immediate_processing),
            screenshot_delay_ms: Cell::new(f64::from(config.screenshot_delay_ms.min(5_000))),
            shortcuts: RefCell::new(config.shortcuts),
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
            immediate_processing: self.immediate_processing.get(),
            screenshot_delay_ms: self.screenshot_delay_ms.get().round() as u32,
            shortcuts: self.shortcuts.borrow().clone(),
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
    app: &adw::Application,
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
    dialog.add(&interface_page(app, state));
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
    let immediate = adw::SwitchRow::builder()
        .title("立即处理")
        .subtitle("打开、粘贴或截取图片后自动开始识别；默认关闭")
        .active(state.immediate_processing.get())
        .build();
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
    lifecycle.add(&immediate);
    lifecycle.add(&keep_warm);
    lifecycle.add(&idle);
    page.add(&lifecycle);

    let state_for_immediate = state.clone();
    immediate.connect_active_notify(move |row| {
        state_for_immediate
            .immediate_processing
            .set(row.is_active());
        state_for_immediate.persist();
    });

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

fn interface_page(app: &adw::Application, state: Rc<SettingsState>) -> adw::PreferencesPage {
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

    let state_for_theme = state.clone();
    theme.connect_selected_notify(move |row| {
        state_for_theme.theme.set(row.selected());
        let color_scheme = match row.selected() {
            1 => adw::ColorScheme::ForceLight,
            2 => adw::ColorScheme::ForceDark,
            _ => adw::ColorScheme::Default,
        };
        adw::StyleManager::default().set_color_scheme(color_scheme);
        state_for_theme.persist();
    });

    let capture = adw::PreferencesGroup::builder()
        .title("截图")
        .description("区域和屏幕截图会在隐藏主窗口后等待桌面动画稳定，再获取物理画面")
        .build();
    let delay = adw::SpinRow::with_range(0.0, 5000.0, 50.0);
    delay.set_title("截图稳定延迟");
    delay.set_subtitle("毫秒；仅影响区域和屏幕截图，窗口截图不会隐藏主窗口");
    delay.set_value(state.screenshot_delay_ms.get());
    delay.set_digits(0);
    capture.add(&delay);
    page.add(&capture);
    let state_for_delay = state.clone();
    delay.connect_value_notify(move |row| {
        state_for_delay.screenshot_delay_ms.set(row.value());
        state_for_delay.persist();
    });

    let shortcuts = adw::PreferencesGroup::builder()
        .title("快捷键")
        .description("使用 GTK 加速键格式；留空可停用快捷键")
        .build();
    let configured = state.shortcuts.borrow().clone();
    let open_image = shortcut_entry("打开图片", &configured.open_image);
    let paste_image = shortcut_entry("粘贴图片", &configured.paste_image);
    let capture_region = shortcut_entry("区域截图", &configured.capture_region);
    let capture_screen = shortcut_entry("屏幕截图", &configured.capture_screen);
    let capture_window = shortcut_entry("窗口截图", &configured.capture_window);
    let start_processing = shortcut_entry("开始或重新处理", &configured.start_processing);
    let copy_all = shortcut_entry("复制全部文本", &configured.copy_all);
    let toggle_results = shortcut_entry("显示或隐藏结果", &configured.toggle_results);
    let preferences = shortcut_entry("打开设置", &configured.preferences);
    for row in [
        &open_image,
        &paste_image,
        &capture_region,
        &capture_screen,
        &capture_window,
        &start_processing,
        &copy_all,
        &toggle_results,
        &preferences,
    ] {
        shortcuts.add(row);
    }
    let apply_row = adw::ActionRow::builder()
        .title("应用快捷键")
        .subtitle("例如 &lt;Primary&gt;V、&lt;Alt&gt;Print；相同组合不能重复")
        .build();
    let shortcut_status = gtk::Label::new(None);
    shortcut_status.add_css_class("caption");
    shortcut_status.add_css_class("dim-label");
    let apply = gtk::Button::with_label("应用");
    apply.set_valign(gtk::Align::Center);
    apply.add_css_class("suggested-action");
    apply_row.add_suffix(&shortcut_status);
    apply_row.add_suffix(&apply);
    shortcuts.add(&apply_row);
    page.add(&shortcuts);

    let app = app.clone();
    apply.connect_clicked(move |_| {
        let updated = ShortcutConfig {
            open_image: open_image.text().to_string(),
            paste_image: paste_image.text().to_string(),
            capture_region: capture_region.text().to_string(),
            capture_screen: capture_screen.text().to_string(),
            capture_window: capture_window.text().to_string(),
            start_processing: start_processing.text().to_string(),
            copy_all: copy_all.text().to_string(),
            toggle_results: toggle_results.text().to_string(),
            preferences: preferences.text().to_string(),
        };
        match validate_shortcuts(&updated) {
            Ok(()) => {
                state.shortcuts.replace(updated.clone());
                state.persist();
                apply_shortcuts(&app, &updated);
                shortcut_status.remove_css_class("error");
                shortcut_status.add_css_class("success");
                shortcut_status.set_label("已应用");
            }
            Err(error) => {
                shortcut_status.remove_css_class("success");
                shortcut_status.add_css_class("error");
                shortcut_status.set_label(&error);
            }
        }
    });

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

fn shortcut_entry(title: &str, accelerator: &str) -> adw::EntryRow {
    adw::EntryRow::builder()
        .title(title)
        .text(accelerator)
        .build()
}

pub fn apply_shortcuts(app: &adw::Application, shortcuts: &ShortcutConfig) {
    for (action, accelerator) in shortcut_bindings(shortcuts) {
        if accelerator.trim().is_empty() || gtk::accelerator_parse(accelerator).is_none() {
            app.set_accels_for_action(action, &[]);
        } else {
            app.set_accels_for_action(action, &[accelerator]);
        }
    }
}

fn validate_shortcuts(shortcuts: &ShortcutConfig) -> Result<(), String> {
    let mut used = std::collections::HashSet::new();
    for (_, accelerator) in shortcut_bindings(shortcuts) {
        let accelerator = accelerator.trim();
        if accelerator.is_empty() {
            continue;
        }
        let Some((key, modifiers)) = gtk::accelerator_parse(accelerator) else {
            return Err(format!("无效：{accelerator}"));
        };
        let normalized = gtk::accelerator_name(key, modifiers).to_string();
        if !used.insert(normalized) {
            return Err(format!("快捷键重复：{accelerator}"));
        }
    }
    Ok(())
}

fn shortcut_bindings(shortcuts: &ShortcutConfig) -> [(&'static str, &str); 9] {
    [
        ("win.open", &shortcuts.open_image),
        ("win.paste", &shortcuts.paste_image),
        ("win.capture-region", &shortcuts.capture_region),
        ("win.capture-screen", &shortcuts.capture_screen),
        ("win.capture-window", &shortcuts.capture_window),
        ("win.process", &shortcuts.start_processing),
        ("win.copy", &shortcuts.copy_all),
        ("win.toggle-results", &shortcuts.toggle_results),
        ("win.preferences", &shortcuts.preferences),
    ]
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
