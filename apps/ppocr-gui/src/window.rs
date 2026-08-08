use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

use adw::prelude::*;
use gtk::{gdk, gio, glib};
use ppocr_app_protocol::{Limits, OcrMetadata};
use ppocr_i18n::{ntr, tr};

use crate::{
    backend::{Backend, BackendEvent, ImageJob, detect_image_format},
    data::export_formats,
    preview::{self, PreviewPane},
    settings::{self, SettingsState},
};
use ppocr_screenshot::{CaptureKind, CaptureOutcome};

thread_local! {
    static ACTIVE_TOAST: RefCell<Option<adw::Toast>> = const { RefCell::new(None) };
}

#[derive(Clone)]
struct LoadedDocument {
    source_name: String,
    bytes: Vec<u8>,
    format: ppocr_app_protocol::EncodedImageFormat,
}

struct DocumentState {
    loaded: RefCell<Option<LoadedDocument>>,
    processing: Cell<bool>,
    has_result: Cell<bool>,
    process_action: gio::SimpleAction,
    process_button: RefCell<Option<gtk::Button>>,
    clear_results: RefCell<Option<Box<dyn Fn()>>>,
}

impl DocumentState {
    fn new() -> Rc<Self> {
        let process_action = gio::SimpleAction::new("process", None);
        process_action.set_enabled(false);
        Rc::new(Self {
            loaded: RefCell::new(None),
            processing: Cell::new(false),
            has_result: Cell::new(false),
            process_action,
            process_button: RefCell::new(None),
            clear_results: RefCell::new(None),
        })
    }

    fn set_processing(&self, processing: bool) {
        self.processing.set(processing);
        self.process_action
            .set_enabled(!processing && self.loaded.borrow().is_some());
        if let Some(button) = self.process_button.borrow().as_ref() {
            button.set_label(&if processing {
                tr!("Processing")
            } else if self.has_result.get() {
                tr!("Process again")
            } else {
                tr!("Start processing")
            });
        }
    }

    fn clear_results(&self) {
        if let Some(clear) = self.clear_results.borrow().as_ref() {
            clear();
        }
    }
}

pub fn build_ui(app: &adw::Application) {
    install_css();

    let window = adw::ApplicationWindow::builder()
        .application(app)
        .title("Paddle OCR")
        .default_width(1280)
        .default_height(820)
        .width_request(820)
        .height_request(620)
        .build();

    let (loaded_settings, config_warning) = SettingsState::load();
    let settings_state = Rc::new(loaded_settings);
    let document_state = DocumentState::new();
    window.add_action(&document_state.process_action);
    apply_theme(settings_state.theme.get());

    let toolbar_view = adw::ToolbarView::new();
    let toast_overlay = adw::ToastOverlay::new();
    let status_text = gtk::Label::new(Some(&tr!("Choose an image to start recognition")));
    let runtime_summary = gtk::Label::new(None);
    settings::update_runtime_summary(&settings_state, &runtime_summary);

    let results = build_results(&toast_overlay, &status_text);
    let results_to_clear = results.clone();
    document_state
        .clear_results
        .replace(Some(Box::new(move || results_to_clear.clear())));
    let list_for_preview = results.line_list.clone();
    let scroller_for_preview = results.line_scroller.clone();
    let stack_for_preview = results.stack.clone();
    let status_for_preview = status_text.clone();
    let toasts_for_toggle = toast_overlay.clone();
    let status_for_toggle = status_text.clone();
    let preview = preview::build_preview(
        move |selected| {
            if let Some(index) = selected {
                if let Some(row) = list_for_preview.row_at_index(index as i32) {
                    list_for_preview.select_row(Some(&row));
                    stack_for_preview.set_visible_child_name("lines");
                    row.grab_focus();
                    let row_for_scroll = row.clone();
                    let list_for_scroll = list_for_preview.clone();
                    let scroller_for_scroll = scroller_for_preview.clone();
                    glib::idle_add_local_once(move || {
                        if let Some(bounds) = row_for_scroll.compute_bounds(&list_for_scroll) {
                            scroller_for_scroll.vadjustment().clamp_page(
                                f64::from(bounds.y()),
                                f64::from(bounds.y() + bounds.height()),
                            );
                        }
                    });
                    status_for_preview
                        .set_label(&tr!("Selected text line {index}", index = index + 1));
                }
            } else {
                list_for_preview.unselect_all();
            }
        },
        move |showing_processed| {
            let message = if showing_processed {
                tr!("Switched to the processed image")
            } else {
                tr!("Switched to the original image")
            };
            status_for_toggle.set_label(&message);
            add_toast(&toasts_for_toggle, &message);
        },
    );

    let selected_for_results = preview.selected_line.clone();
    let drawing_for_results = preview.drawing_area.clone();
    let status_for_results = status_text.clone();
    results.line_list.connect_row_selected(move |_, row| {
        let selected = row.map(|row| row.index() as usize);
        selected_for_results.set(selected);
        drawing_for_results.queue_draw();
        if let Some(index) = selected {
            status_for_results.set_label(&tr!("Selected text line {index}", index = index + 1));
        }
    });

    let backend = build_backend(
        &preview,
        &results,
        &toast_overlay,
        &status_text,
        &runtime_summary,
        document_state.clone(),
    );

    preview.root.set_margin_end(12);
    let split_view = adw::OverlaySplitView::builder()
        .content(&preview.root)
        .sidebar(&results.root)
        .sidebar_position(gtk::PackType::End)
        .sidebar_width_fraction(0.35)
        .min_sidebar_width(380.0)
        .max_sidebar_width(460.0)
        .pin_sidebar(true)
        .show_sidebar(true)
        .enable_hide_gesture(true)
        .enable_show_gesture(true)
        .hexpand(true)
        .vexpand(true)
        .build();
    split_view.add_css_class("ocr-split-view");

    let breakpoint_bin = adw::BreakpointBin::builder()
        .child(&split_view)
        .width_request(400)
        .height_request(400)
        .build();
    let compact_condition = adw::BreakpointCondition::parse("max-width: 950px")
        .expect("hard-coded breakpoint condition must be valid");
    let compact_breakpoint = adw::Breakpoint::new(compact_condition);
    compact_breakpoint.add_setters(&[
        (&split_view, "collapsed", true),
        (&split_view, "show-sidebar", false),
    ]);
    compact_breakpoint.add_setter(&preview.root, "margin-end", Some(&0_i32.to_value()));
    breakpoint_bin.add_breakpoint(compact_breakpoint);

    let work_area = gtk::Box::new(gtk::Orientation::Vertical, 0);
    work_area.set_margin_top(18);
    work_area.set_margin_bottom(14);
    work_area.set_margin_start(18);
    work_area.set_margin_end(18);
    work_area.append(&breakpoint_bin);
    breakpoint_bin.set_vexpand(true);
    breakpoint_bin.set_hexpand(true);
    toast_overlay.set_child(Some(&work_area));
    toolbar_view.set_content(Some(&toast_overlay));

    let header = build_header(
        app,
        &window,
        &toast_overlay,
        &status_text,
        &runtime_summary,
        &results,
        &preview,
        settings_state.clone(),
        &split_view,
        backend.clone(),
        document_state.clone(),
    );
    toolbar_view.add_top_bar(&header);

    let status_bar = build_status_bar(&status_text, &runtime_summary);
    toolbar_view.add_bottom_bar(&status_bar);
    window.set_content(Some(&toolbar_view));

    let backend_for_close = backend.clone();
    window.connect_close_request(move |_| {
        backend_for_close.shutdown();
        glib::Propagation::Proceed
    });
    window.present();
    if let Some(warning) = config_warning {
        add_toast(&toast_overlay, &warning);
    }
    #[cfg(debug_assertions)]
    if let Some(path) = std::env::var_os("PPOCR_GUI_TEST_IMAGE") {
        submit_image_file(
            gio::File::for_path(path),
            backend,
            settings_state,
            preview,
            document_state,
            &toast_overlay,
            &status_text,
        );
    }
}

fn build_backend(
    preview: &PreviewPane,
    results: &ResultsPane,
    toasts: &adw::ToastOverlay,
    status: &gtk::Label,
    runtime_summary: &gtk::Label,
    document: Rc<DocumentState>,
) -> Rc<Backend> {
    let preview = preview.clone();
    let results = results.clone();
    let toasts = toasts.clone();
    let status = status.clone();
    let runtime_summary = runtime_summary.clone();
    let document_for_events = document;
    Backend::new(move |event| match event {
        BackendEvent::Status(message) => {
            status.set_label(&message);
            if document_for_events.processing.get() {
                preview.set_progress(&message, None);
            }
        }
        BackendEvent::Progress(progress) => {
            let fraction = progress
                .bytes_completed
                .zip(progress.total_bytes)
                .filter(|(_, total)| *total > 0)
                .map(|(done, total)| done as f64 / total as f64);
            preview.set_progress(&progress.message, fraction);
            status.set_label(&progress.message);
        }
        BackendEvent::Capabilities(hello) => {
            let available = hello
                .providers
                .iter()
                .filter(|provider| provider.compiled && provider.ort_reported_available)
                .count();
            runtime_summary.set_tooltip_text(Some(&tr!(
                "IPC connected: {available} of {total} execution providers available",
                available = available,
                total = hello.providers.len()
            )));
        }
        BackendEvent::Result {
            source_name,
            result,
            processed_png,
        } => {
            document_for_events.has_result.set(true);
            document_for_events.set_processing(false);
            results.set_metadata(&result.metadata);
            let model = result.metadata.model_ids.text_recognition.clone();
            let line_count = result.metadata.text_lines.len();
            match preview.set_result(processed_png, result.metadata, &source_name) {
                Ok(()) => {
                    status.set_label(&ntr!(
                        "Recognition complete · 1 line · {elapsed} ms",
                        "Recognition complete · {count} lines · {elapsed} ms",
                        line_count,
                        count = line_count,
                        elapsed = result.elapsed_ms
                    ));
                    runtime_summary.set_label(&format!("{model}  ·  {} ms", result.elapsed_ms));
                }
                Err(error) => {
                    let message = tr!(
                        "Recognition succeeded, but the preview image cannot be displayed: {error}",
                        error = error
                    );
                    status.set_label(&message);
                    add_toast(&toasts, &message);
                }
            }
        }
        BackendEvent::Error(message) => {
            document_for_events.set_processing(false);
            preview.set_failed();
            status.set_label(&tr!("Recognition failed"));
            add_toast(&toasts, &message);
        }
    })
}

#[derive(Clone)]
struct ResultsPane {
    root: gtk::Box,
    stack: adw::ViewStack,
    line_list: gtk::ListBox,
    line_scroller: gtk::ScrolledWindow,
    buffer: gtk::TextBuffer,
    count: gtk::Label,
    rows: Rc<RefCell<Vec<(adw::ActionRow, String)>>>,
    original_text: Rc<RefCell<String>>,
}

impl ResultsPane {
    fn set_metadata(&self, metadata: &OcrMetadata) {
        self.line_list.unselect_all();
        while let Some(child) = self.line_list.first_child() {
            self.line_list.remove(&child);
        }
        let mut rows = self.rows.borrow_mut();
        rows.clear();
        for (index, line) in metadata.text_lines.iter().enumerate() {
            let row = adw::ActionRow::builder()
                .title(&line.text)
                .subtitle(tr!(
                    "Text line {index:02}  ·  Confidence {confidence:.1}%",
                    index = index + 1,
                    confidence = line.recognition_score * 100.0
                ))
                .activatable(true)
                .build();
            self.line_list.append(&row);
            rows.push((row, line.text.to_lowercase()));
        }
        self.count
            .set_label(&line_count_label(metadata.text_lines.len()));
        let text = metadata
            .text_lines
            .iter()
            .map(|line| line.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        self.original_text.replace(text.clone());
        self.buffer.set_text(&text);
    }

    fn clear(&self) {
        self.line_list.unselect_all();
        while let Some(child) = self.line_list.first_child() {
            self.line_list.remove(&child);
        }
        self.rows.borrow_mut().clear();
        self.count.set_label(&line_count_label(0));
        self.original_text.borrow_mut().clear();
        self.buffer.set_text("");
    }
}

fn build_results(toasts: &adw::ToastOverlay, status: &gtk::Label) -> ResultsPane {
    let root = gtk::Box::new(gtk::Orientation::Vertical, 0);
    root.add_css_class("card");
    root.set_overflow(gtk::Overflow::Hidden);
    root.set_size_request(390, -1);
    root.set_vexpand(true);

    let top = gtk::Box::new(gtk::Orientation::Horizontal, 10);
    top.set_margin_top(14);
    top.set_margin_bottom(10);
    top.set_margin_start(16);
    top.set_margin_end(16);
    let title = gtk::Label::new(Some(&tr!("Recognition results")));
    title.add_css_class("heading");
    title.set_xalign(0.0);
    top.append(&title);
    let spacer = gtk::Box::new(gtk::Orientation::Horizontal, 0);
    spacer.set_hexpand(true);
    top.append(&spacer);
    let count = gtk::Label::new(Some(&line_count_label(0)));
    count.add_css_class("dim-label");
    count.add_css_class("pill");
    top.append(&count);
    root.append(&top);

    let stack = adw::ViewStack::builder()
        .hexpand(true)
        .vexpand(true)
        .vhomogeneous(true)
        .build();
    let switcher = adw::ViewSwitcher::builder()
        .stack(&stack)
        .policy(adw::ViewSwitcherPolicy::Wide)
        .halign(gtk::Align::Center)
        .build();
    switcher.set_margin_bottom(10);
    root.append(&switcher);

    let buffer = gtk::TextBuffer::new(None);
    let text_view = gtk::TextView::builder()
        .buffer(&buffer)
        .wrap_mode(gtk::WrapMode::WordChar)
        .top_margin(14)
        .bottom_margin(14)
        .left_margin(14)
        .right_margin(14)
        .hexpand(true)
        .vexpand(true)
        .build();
    text_view.add_css_class("ocr-text");
    let text_scroller = gtk::ScrolledWindow::builder()
        .hscrollbar_policy(gtk::PolicyType::Never)
        .vscrollbar_policy(gtk::PolicyType::Automatic)
        .hexpand(true)
        .vexpand(true)
        .child(&text_view)
        .build();
    stack.add_titled(&text_scroller, Some("text"), &tr!("Text"));

    let lines_page = gtk::Box::new(gtk::Orientation::Vertical, 10);
    lines_page.set_margin_start(12);
    lines_page.set_margin_end(12);
    lines_page.set_margin_bottom(12);
    let search = gtk::SearchEntry::builder()
        .placeholder_text(tr!("Search text lines"))
        .build();
    lines_page.append(&search);
    let line_list = gtk::ListBox::new();
    line_list.set_selection_mode(gtk::SelectionMode::Single);
    line_list.add_css_class("boxed-list");
    let toasts_for_line_copy = toasts.clone();
    let status_for_line_copy = status.clone();
    line_list.connect_row_activated(move |_, row| {
        let Ok(row) = row.clone().downcast::<adw::ActionRow>() else {
            return;
        };
        let text = row.title();
        row.display().clipboard().set_text(&text);
        status_for_line_copy.set_label(&tr!("Copied the selected text line"));
        add_toast(
            &toasts_for_line_copy,
            &tr!("Text line copied to the clipboard"),
        );
    });

    let rows = Rc::new(RefCell::new(Vec::<(adw::ActionRow, String)>::new()));
    let rows_for_search = rows.clone();
    search.connect_search_changed(move |entry| {
        let query = entry.text().trim().to_lowercase();
        for (row, text) in rows_for_search.borrow().iter() {
            row.set_visible(query.is_empty() || text.contains(&query));
        }
    });

    let line_scroller = gtk::ScrolledWindow::builder()
        .hscrollbar_policy(gtk::PolicyType::Never)
        .vscrollbar_policy(gtk::PolicyType::Automatic)
        .hexpand(true)
        .vexpand(true)
        .child(&line_list)
        .build();
    lines_page.append(&line_scroller);
    stack.add_titled(&lines_page, Some("lines"), &tr!("Text lines"));

    root.append(&stack);
    root.append(&gtk::Separator::new(gtk::Orientation::Horizontal));

    let footer = gtk::Box::new(gtk::Orientation::Horizontal, 10);
    footer.set_margin_top(10);
    footer.set_margin_bottom(12);
    footer.set_margin_start(14);
    footer.set_margin_end(14);
    let char_count = gtk::Label::new(Some(&character_count_label(0)));
    char_count.add_css_class("dim-label");
    footer.append(&char_count);
    let footer_spacer = gtk::Box::new(gtk::Orientation::Horizontal, 0);
    footer_spacer.set_hexpand(true);
    footer.append(&footer_spacer);
    let restore = gtk::Button::with_label(&tr!("Restore result"));
    restore.add_css_class("flat");
    footer.append(&restore);
    let copy = gtk::Button::builder()
        .label(tr!("Copy all"))
        .icon_name("edit-copy-symbolic")
        .build();
    footer.append(&copy);
    root.append(&footer);

    let count_for_buffer = char_count.clone();
    buffer.connect_changed(move |buffer| {
        let text = buffer.text(&buffer.start_iter(), &buffer.end_iter(), true);
        count_for_buffer.set_label(&character_count_label(text.chars().count()));
    });

    let buffer_for_copy = buffer.clone();
    let toasts_for_copy = toasts.clone();
    copy.connect_clicked(move |_| {
        copy_buffer(&buffer_for_copy);
        add_toast(
            &toasts_for_copy,
            &tr!("Recognized text copied to the clipboard"),
        );
    });

    let original_text = Rc::new(RefCell::new(String::new()));
    let original_for_restore = original_text.clone();
    let buffer_for_restore = buffer.clone();
    let toasts_for_restore = toasts.clone();
    let status_for_restore = status.clone();
    restore.connect_clicked(move |_| {
        buffer_for_restore.set_text(&original_for_restore.borrow());
        let message = tr!("Restored the original recognized text");
        status_for_restore.set_label(&message);
        add_toast(&toasts_for_restore, &message);
    });

    ResultsPane {
        root,
        stack,
        line_list,
        line_scroller,
        buffer,
        count,
        rows,
        original_text,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_header(
    app: &adw::Application,
    window: &adw::ApplicationWindow,
    toasts: &adw::ToastOverlay,
    status: &gtk::Label,
    runtime_summary: &gtk::Label,
    results_pane: &ResultsPane,
    preview: &PreviewPane,
    settings_state: Rc<SettingsState>,
    split_view: &adw::OverlaySplitView,
    backend: Rc<Backend>,
    document: Rc<DocumentState>,
) -> adw::HeaderBar {
    let header = adw::HeaderBar::new();
    let title = adw::WindowTitle::builder()
        .title("Paddle OCR")
        .subtitle(tr!("Native GTK4 frontend"))
        .build();
    header.set_title_widget(Some(&title));

    let capture_menu = gio::Menu::new();
    capture_menu.append(Some(&tr!("Capture region")), Some("win.capture-region"));
    capture_menu.append(Some(&tr!("Capture screen")), Some("win.capture-screen"));
    capture_menu.append(Some(&tr!("Capture window")), Some("win.capture-window"));
    let capture = adw::SplitButton::builder()
        .label(tr!("Capture and recognize"))
        .icon_name("edit-select-all-symbolic")
        .menu_model(&capture_menu)
        .dropdown_tooltip(tr!("Choose a capture target"))
        .build();
    capture.add_css_class("suggested-action");
    capture.set_action_name(Some("win.capture-region"));
    capture.set_tooltip_text(Some(&tr!(
        "Select a screen region and recognize it  Ctrl+Shift+O"
    )));
    header.pack_start(&capture);

    let open = button_content(&tr!("Open image"), "document-open-symbolic");
    open.set_action_name(Some("win.open"));
    open.set_tooltip_text(Some(&tr!("Choose an image file  Ctrl+O")));
    header.pack_start(&open);

    let process = button_content(&tr!("Start processing"), "media-playback-start-symbolic");
    process.set_action_name(Some("win.process"));
    process.add_css_class("suggested-action");
    process.set_tooltip_text(Some(&tr!(
        "Start recognition with the current pipeline settings"
    )));
    document.process_button.replace(Some(process.clone()));
    header.pack_start(&process);

    let preferences = gtk::Button::from_icon_name("emblem-system-symbolic");
    preferences.add_css_class("flat");
    preferences.set_action_name(Some("win.preferences"));
    preferences.set_tooltip_text(Some(&tr!(
        "Model, pipeline, and interface settings  Ctrl+,"
    )));
    header.pack_end(&preferences);

    let more = build_more_menu(window);
    header.pack_end(&more);
    let export = build_export_menu(toasts, status);
    header.pack_end(&export);

    let copy = button_content(&tr!("Copy"), "edit-copy-symbolic");
    copy.set_action_name(Some("win.copy"));
    copy.set_tooltip_text(Some(&tr!("Copy all recognized text  Ctrl+Shift+C")));
    header.pack_end(&copy);

    let results = button_content(&tr!("Results"), "sidebar-show-symbolic");
    results.set_action_name(Some("win.toggle-results"));
    results.set_tooltip_text(Some(&tr!("Show or hide recognition results")));
    header.pack_end(&results);

    install_window_actions(
        app,
        window,
        toasts,
        status,
        runtime_summary,
        &results_pane.buffer,
        preview,
        settings_state,
        split_view,
        backend,
        document,
    );
    header
}

#[allow(clippy::too_many_arguments)]
fn install_window_actions(
    app: &adw::Application,
    window: &adw::ApplicationWindow,
    toasts: &adw::ToastOverlay,
    status: &gtk::Label,
    runtime_summary: &gtk::Label,
    buffer: &gtk::TextBuffer,
    preview: &PreviewPane,
    settings_state: Rc<SettingsState>,
    split_view: &adw::OverlaySplitView,
    backend: Rc<Backend>,
    document: Rc<DocumentState>,
) {
    let open = gio::SimpleAction::new("open", None);
    let window_for_open = window.clone();
    let backend_for_open = backend.clone();
    let settings_for_open = settings_state.clone();
    let preview_for_open = preview.clone();
    let toasts_for_open = toasts.clone();
    let status_for_open = status.clone();
    let document_for_open = document.clone();
    open.connect_activate(move |_, _| {
        open_image_dialog(
            &window_for_open,
            backend_for_open.clone(),
            settings_for_open.clone(),
            preview_for_open.clone(),
            document_for_open.clone(),
            &toasts_for_open,
            &status_for_open,
        );
    });
    window.add_action(&open);

    let paste = gio::SimpleAction::new("paste", None);
    paste.set_enabled(preview.drawing_area.has_focus());
    let paste_for_focus = paste.clone();
    preview.drawing_area.connect_has_focus_notify(move |area| {
        paste_for_focus.set_enabled(area.has_focus());
    });
    let window_for_paste = window.clone();
    let backend_for_paste = backend.clone();
    let settings_for_paste = settings_state.clone();
    let preview_for_paste = preview.clone();
    let document_for_paste = document.clone();
    let toasts_for_paste = toasts.clone();
    let status_for_paste = status.clone();
    paste.connect_activate(move |_, _| {
        paste_clipboard_image(
            &window_for_paste,
            backend_for_paste.clone(),
            settings_for_paste.clone(),
            preview_for_paste.clone(),
            document_for_paste.clone(),
            &toasts_for_paste,
            &status_for_paste,
        );
    });
    window.add_action(&paste);

    for (action_name, kind, prompt) in [
        (
            "capture-region",
            CaptureKind::Region,
            tr!("Preparing region capture"),
        ),
        (
            "capture-screen",
            CaptureKind::Screen,
            tr!("Choose the screen to capture"),
        ),
        (
            "capture-window",
            CaptureKind::Window,
            tr!("Choose the window to capture"),
        ),
    ] {
        let capture = gio::SimpleAction::new(action_name, None);
        let window_for_capture = window.clone();
        let backend = backend.clone();
        let settings = settings_state.clone();
        let preview = preview.clone();
        let toasts = toasts.clone();
        let status = status.clone();
        let document = document.clone();
        capture.connect_activate(move |_, _| {
            let window = window_for_capture.clone();
            let backend = backend.clone();
            let settings = settings.clone();
            let preview = preview.clone();
            let toasts = toasts.clone();
            let status = status.clone();
            let document = document.clone();
            status.set_label(&prompt);
            glib::MainContext::default().spawn_local(async move {
                let configured_delay = settings.screenshot_delay_ms.get().round() as u64;
                #[cfg(debug_assertions)]
                let configured_delay = std::env::var("PPOCR_GUI_TEST_SCREENSHOT_DELAY_MS")
                    .ok()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(configured_delay);
                let settle_delay = std::time::Duration::from_millis(configured_delay.min(30_000));
                match ppocr_screenshot::capture(
                    window.upcast_ref::<gtk::Window>(),
                    kind,
                    settle_delay,
                )
                .await
                {
                    Ok(CaptureOutcome::Captured(image)) => {
                        load_image_bytes(
                            image.bytes,
                            image.source_name,
                            backend,
                            settings,
                            preview,
                            document,
                            &toasts,
                            &status,
                        );
                    }
                    Ok(CaptureOutcome::Cancelled) => {
                        status.set_label(&tr!("Capture cancelled"));
                    }
                    Err(error) => {
                        eprintln!("screenshot capture failed: {error}");
                        status.set_label(&tr!("Capture failed"));
                        add_toast(&toasts, &error.to_string());
                    }
                }
            });
        });
        window.add_action(&capture);
    }

    let copy = gio::SimpleAction::new("copy", None);
    let buffer_for_copy = buffer.clone();
    let toasts_for_copy = toasts.clone();
    copy.connect_activate(move |_, _| {
        copy_buffer(&buffer_for_copy);
        add_toast(
            &toasts_for_copy,
            &tr!("Recognized text copied to the clipboard"),
        );
    });
    window.add_action(&copy);

    let backend_for_process = backend.clone();
    let settings_for_process = settings_state.clone();
    let preview_for_process = preview.clone();
    let status_for_process = status.clone();
    let document_for_process = document.clone();
    document.process_action.connect_activate(move |_, _| {
        process_loaded_document(
            backend_for_process.clone(),
            settings_for_process.clone(),
            preview_for_process.clone(),
            document_for_process.clone(),
            &status_for_process,
        );
    });

    let preferences = gio::SimpleAction::new("preferences", None);
    let app_for_preferences = app.clone();
    let window_for_preferences = window.clone();
    let runtime_for_preferences = runtime_summary.clone();
    let settings_for_preferences = settings_state.clone();
    preferences.connect_activate(move |_, _| {
        settings::present_settings(
            &app_for_preferences,
            &window_for_preferences,
            settings_for_preferences.clone(),
            &runtime_for_preferences,
        );
    });
    window.add_action(&preferences);

    let toggle_results = gio::SimpleAction::new("toggle-results", None);
    let split_for_action = split_view.clone();
    toggle_results.connect_activate(move |_, _| {
        split_for_action.set_show_sidebar(!split_for_action.shows_sidebar());
    });
    window.add_action(&toggle_results);

    settings::apply_shortcuts(app, &settings_state.shortcuts.borrow());
}

fn open_image_dialog(
    window: &adw::ApplicationWindow,
    backend: Rc<Backend>,
    settings: Rc<SettingsState>,
    preview: PreviewPane,
    document: Rc<DocumentState>,
    toasts: &adw::ToastOverlay,
    status: &gtk::Label,
) {
    let image_filter = gtk::FileFilter::new();
    image_filter.set_name(Some(&tr!("Supported images")));
    for mime_type in [
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/tiff",
        "image/bmp",
        "image/gif",
    ] {
        image_filter.add_mime_type(mime_type);
    }
    let filters = gio::ListStore::new::<gtk::FileFilter>();
    filters.append(&image_filter);
    let dialog = gtk::FileDialog::builder()
        .title(tr!("Choose an image to recognize"))
        .accept_label(tr!("Open"))
        .modal(true)
        .filters(&filters)
        .default_filter(&image_filter)
        .build();
    let window = window.clone();
    let toasts = toasts.clone();
    let status = status.clone();
    glib::MainContext::default().spawn_local(async move {
        let file = match dialog.open_future(Some(&window)).await {
            Ok(file) => file,
            Err(error) if error.matches::<gio::IOErrorEnum>(gio::IOErrorEnum::Cancelled) => return,
            Err(error) => {
                add_toast(
                    &toasts,
                    &tr!("Failed to open the file chooser: {error}", error = error),
                );
                return;
            }
        };
        submit_image_file(file, backend, settings, preview, document, &toasts, &status);
    });
}

#[allow(clippy::too_many_arguments)]
fn paste_clipboard_image(
    window: &adw::ApplicationWindow,
    backend: Rc<Backend>,
    settings: Rc<SettingsState>,
    preview: PreviewPane,
    document: Rc<DocumentState>,
    toasts: &adw::ToastOverlay,
    status: &gtk::Label,
) {
    let clipboard = gtk::prelude::WidgetExt::display(window).clipboard();
    let toasts = toasts.clone();
    let status = status.clone();
    glib::MainContext::default().spawn_local(async move {
        status.set_label(&tr!("Reading an image from the clipboard"));
        match clipboard.read_texture_future().await {
            Ok(Some(texture)) => {
                let png = texture.save_to_png_bytes();
                load_image_bytes(
                    png.as_ref().to_vec(),
                    tr!("Clipboard image.png"),
                    backend,
                    settings,
                    preview,
                    document,
                    &toasts,
                    &status,
                );
            }
            Ok(None) => {
                status.set_label(&tr!("The clipboard does not contain an image"));
                add_toast(
                    &toasts,
                    &tr!("No pasteable image was found on the clipboard"),
                );
            }
            Err(error) => {
                status.set_label(&tr!("Failed to read the clipboard"));
                add_toast(
                    &toasts,
                    &tr!("Failed to read the clipboard image: {error}", error = error),
                );
            }
        }
    });
}

fn submit_image_file(
    file: gio::File,
    backend: Rc<Backend>,
    settings: Rc<SettingsState>,
    preview: PreviewPane,
    document: Rc<DocumentState>,
    toasts: &adw::ToastOverlay,
    status: &gtk::Label,
) {
    let toasts = toasts.clone();
    let status = status.clone();
    glib::MainContext::default().spawn_local(async move {
        status.set_label(&tr!("Reading the image"));
        let (bytes, _) = match file.load_bytes_future().await {
            Ok(value) => value,
            Err(error) => {
                status.set_label(&tr!("Failed to read the image"));
                add_toast(
                    &toasts,
                    &tr!("Failed to read the selected image: {error}", error = error),
                );
                return;
            }
        };
        let bytes = bytes.as_ref().to_vec();
        let source_name = file
            .basename()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| file.parse_name().to_string());
        load_image_bytes(
            bytes,
            source_name,
            backend,
            settings,
            preview,
            document,
            &toasts,
            &status,
        );
    });
}

#[allow(clippy::too_many_arguments)]
fn load_image_bytes(
    bytes: Vec<u8>,
    source_name: String,
    backend: Rc<Backend>,
    settings: Rc<SettingsState>,
    preview: PreviewPane,
    document: Rc<DocumentState>,
    toasts: &adw::ToastOverlay,
    status: &gtk::Label,
) {
    if bytes.len() as u64 > Limits::DEFAULT.max_attachment_bytes {
        status.set_label(&tr!("The image file is too large"));
        add_toast(toasts, &tr!("The image file exceeds the 256 MiB IPC limit"));
        return;
    }
    let Some(format) = detect_image_format(&bytes) else {
        status.set_label(&tr!("Unsupported image format"));
        add_toast(
            toasts,
            &tr!("The file is not a supported PNG, JPEG, WebP, TIFF, BMP, or GIF image"),
        );
        return;
    };
    if let Err(error) = preview.set_source(&bytes, &source_name) {
        status.set_label(&tr!("Image preview failed"));
        add_toast(
            toasts,
            &tr!(
                "Failed to decode the selected image: {error}",
                error = error
            ),
        );
        return;
    }
    backend.invalidate();
    document.clear_results();
    document.loaded.replace(Some(LoadedDocument {
        source_name,
        bytes,
        format,
    }));
    document.has_result.set(false);
    document.set_processing(false);
    if settings.immediate_processing.get() {
        process_loaded_document(backend, settings, preview, document, status);
    } else {
        status.set_label(&tr!(
            "The image is ready; click “Start processing” to recognize it"
        ));
    }
}

fn process_loaded_document(
    backend: Rc<Backend>,
    settings: Rc<SettingsState>,
    preview: PreviewPane,
    document: Rc<DocumentState>,
    status: &gtk::Label,
) {
    if document.processing.get() {
        return;
    }
    let Some(loaded) = document.loaded.borrow().clone() else {
        status.set_label(&tr!("Open, paste, or capture an image first"));
        return;
    };
    document.set_processing(true);
    preview.set_processing();
    status.set_label(&tr!("Connecting to the recognition backend"));
    glib::MainContext::default().spawn_local(async move {
        #[cfg(debug_assertions)]
        if let Ok(delay_ms) = std::env::var("PPOCR_GUI_TEST_DELAY_MS")
            && let Ok(delay_ms) = delay_ms.parse::<u64>()
        {
            glib::timeout_future(std::time::Duration::from_millis(delay_ms)).await;
        }
        backend.submit(ImageJob {
            source_name: loaded.source_name,
            bytes: loaded.bytes,
            format: loaded.format,
            config: settings.pipeline_config(),
            keep_worker_warm: settings.keep_worker_warm.get(),
            idle_timeout_minutes: settings.idle_timeout_minutes.get().round() as u32,
        });
    });
}

fn build_export_menu(toasts: &adw::ToastOverlay, status: &gtk::Label) -> gtk::MenuButton {
    let button = gtk::MenuButton::new();
    let content = adw::ButtonContent::builder()
        .icon_name("document-save-symbolic")
        .label(tr!("Export"))
        .can_shrink(true)
        .build();
    button.set_child(Some(&content));
    button.set_tooltip_text(Some(&tr!("Export recognition results")));

    let popover = gtk::Popover::new();
    let box_ = gtk::Box::new(gtk::Orientation::Vertical, 4);
    box_.set_margin_top(10);
    box_.set_margin_bottom(10);
    box_.set_margin_start(10);
    box_.set_margin_end(10);
    box_.set_size_request(240, -1);
    let heading = gtk::Label::new(Some(&tr!("Export recognition results")));
    heading.set_xalign(0.0);
    heading.add_css_class("heading");
    heading.set_margin_bottom(4);
    box_.append(&heading);
    for format in export_formats() {
        let item = gtk::Button::with_label(&format);
        item.set_halign(gtk::Align::Fill);
        item.add_css_class("flat");
        let toasts = toasts.clone();
        let status = status.clone();
        let popover = popover.clone();
        item.connect_clicked(move |_| {
            status.set_label(&tr!("Selected {format}", format = format));
            add_toast(
                &toasts,
                &tr!(
                    "Exporting {format} will be implemented in a later stage",
                    format = format
                ),
            );
            popover.popdown();
        });
        box_.append(&item);
    }
    box_.append(&gtk::Separator::new(gtk::Orientation::Horizontal));
    let pdf = gtk::Button::with_label(&tr!("Searchable PDF (later stage)"));
    pdf.set_sensitive(false);
    pdf.set_tooltip_text(Some(&tr!(
        "Requires PDF page rendering and text-layer coordinate mapping"
    )));
    box_.append(&pdf);
    popover.set_child(Some(&box_));
    button.set_popover(Some(&popover));
    button
}

fn build_more_menu(window: &adw::ApplicationWindow) -> gtk::MenuButton {
    let button = gtk::MenuButton::new();
    button.set_icon_name("open-menu-symbolic");
    button.add_css_class("flat");
    button.set_tooltip_text(Some(&tr!("More options")));

    let popover = gtk::Popover::new();
    let box_ = gtk::Box::new(gtk::Orientation::Vertical, 4);
    box_.set_margin_top(8);
    box_.set_margin_bottom(8);
    box_.set_margin_start(8);
    box_.set_margin_end(8);
    box_.set_size_request(210, -1);

    let about = gtk::Button::with_label(&tr!("About Paddle OCR"));
    about.add_css_class("flat");
    let window_for_about = window.clone();
    about.connect_clicked(move |_| {
        let dialog = adw::AboutDialog::builder()
            .application_name("Paddle OCR")
            .application_icon("image-text-symbolic")
            .developer_name("ppocr-rs contributors")
            .version("0.1.0")
            .comments(tr!(
                "A PaddleOCR desktop frontend built with Rust, GTK4, and libadwaita"
            ))
            .website("https://github.com/PaddlePaddle/PaddleOCR")
            .license_type(gtk::License::Apache20)
            .build();
        dialog.present(Some(&window_for_about));
    });
    box_.append(&about);
    popover.set_child(Some(&box_));
    button.set_popover(Some(&popover));
    button
}

fn build_status_bar(status: &gtk::Label, runtime_summary: &gtk::Label) -> gtk::Box {
    let bar = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    bar.add_css_class("status-bar");
    bar.set_margin_top(7);
    bar.set_margin_bottom(9);
    bar.set_margin_start(18);
    bar.set_margin_end(18);

    let icon = gtk::Image::from_icon_name("emblem-ok-symbolic");
    icon.add_css_class("success");
    bar.append(&icon);
    status.set_xalign(0.0);
    status.add_css_class("dim-label");
    status.set_ellipsize(gtk::pango::EllipsizeMode::End);
    bar.append(status);

    let spacer = gtk::Box::new(gtk::Orientation::Horizontal, 0);
    spacer.set_hexpand(true);
    bar.append(&spacer);
    runtime_summary.set_xalign(1.0);
    runtime_summary.add_css_class("dim-label");
    runtime_summary.set_ellipsize(gtk::pango::EllipsizeMode::Middle);
    runtime_summary.set_max_width_chars(52);
    bar.append(runtime_summary);
    bar
}

fn button_content(label: &str, icon_name: &str) -> gtk::Button {
    let button = gtk::Button::new();
    let content = adw::ButtonContent::builder()
        .label(label)
        .icon_name(icon_name)
        .can_shrink(true)
        .build();
    button.set_child(Some(&content));
    button
}

fn line_count_label(count: usize) -> String {
    ntr!("1 line", "{count} lines", count, count = count)
}

fn character_count_label(count: usize) -> String {
    ntr!("1 character", "{count} characters", count, count = count)
}

fn copy_buffer(buffer: &gtk::TextBuffer) {
    let text = buffer.text(&buffer.start_iter(), &buffer.end_iter(), true);
    if let Some(display) = gdk::Display::default() {
        display.clipboard().set_text(&text);
    }
}

fn add_toast(overlay: &adw::ToastOverlay, message: &str) {
    let toast = adw::Toast::builder().title(message).timeout(3).build();
    ACTIVE_TOAST.with(|active| {
        if let Some(previous) = active.replace(Some(toast.clone())) {
            previous.dismiss();
        }
    });
    overlay.add_toast(toast);
}

fn apply_theme(theme: u32) {
    let color_scheme = match theme {
        1 => adw::ColorScheme::ForceLight,
        2 => adw::ColorScheme::ForceDark,
        _ => adw::ColorScheme::Default,
    };
    adw::StyleManager::default().set_color_scheme(color_scheme);
}

fn install_css() {
    let provider = gtk::CssProvider::new();
    provider.load_from_string(
        r#"
        .card {
            border-radius: 12px;
            border: 1px solid alpha(@window_fg_color, 0.12);
            background-color: @view_bg_color;
        }
        overlay-split-view.ocr-split-view > widget.sidebar-pane,
        overlay-split-view.ocr-split-view > widget.background {
            background-color: transparent;
            background-image: none;
            border: none;
            box-shadow: none;
        }
        .preview-canvas {
            background-color: @headerbar_bg_color;
        }
        .processing-overlay {
            background-color: rgba(0, 0, 0, 0.44);
        }
        .processing-label {
            color: white;
            font-weight: 600;
        }
        .ocr-text {
            font-size: 1.02em;
            line-height: 1.25;
        }
        .status-bar {
            min-height: 22px;
        }
        .pill {
            border-radius: 999px;
            padding: 4px 10px;
        }
        "#,
    );
    if let Some(display) = gdk::Display::default() {
        gtk::style_context_add_provider_for_display(
            &display,
            &provider,
            gtk::STYLE_PROVIDER_PRIORITY_APPLICATION,
        );
    }
}
