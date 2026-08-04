use std::rc::Rc;

use adw::prelude::*;
use gtk::{gdk, gio};

use crate::{
    data::{DEMO_LINES, EXPORT_FORMATS, demo_text},
    preview,
    settings::{self, SettingsState},
};

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

    let settings_state = Rc::new(SettingsState::default());
    let toolbar_view = adw::ToolbarView::new();
    let toast_overlay = adw::ToastOverlay::new();
    let status_text = gtk::Label::new(Some("界面演示 · 尚未连接识别后端"));
    let runtime_summary = gtk::Label::new(None);
    settings::update_runtime_summary(&settings_state, &runtime_summary);

    let results = build_results(&toast_overlay, &status_text);
    let list_for_preview = results.line_list.clone();
    let stack_for_preview = results.stack.clone();
    let status_for_preview = status_text.clone();
    let preview = preview::build_preview(move |selected| {
        if let Some(index) = selected {
            if let Some(row) = list_for_preview.row_at_index(index as i32) {
                list_for_preview.select_row(Some(&row));
                stack_for_preview.set_visible_child_name("lines");
                status_for_preview.set_label(&format!("已选择文本行 {}", index + 1));
            }
        } else {
            list_for_preview.unselect_all();
        }
    });

    let selected_for_results = preview.selected_line.clone();
    let drawing_for_results = preview.drawing_area.clone();
    let status_for_results = status_text.clone();
    results.line_list.connect_row_selected(move |_, row| {
        let selected = row.map(|row| row.index() as usize);
        selected_for_results.set(selected);
        drawing_for_results.queue_draw();
        if let Some(index) = selected {
            status_for_results.set_label(&format!("已选择文本行 {}", index + 1));
        }
    });

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
        &results.buffer,
        settings_state,
        &split_view,
    );
    toolbar_view.add_top_bar(&header);

    let status_bar = build_status_bar(&status_text, &runtime_summary);
    toolbar_view.add_bottom_bar(&status_bar);
    window.set_content(Some(&toolbar_view));
    window.present();
}

struct ResultsPane {
    root: gtk::Box,
    stack: adw::ViewStack,
    line_list: gtk::ListBox,
    buffer: gtk::TextBuffer,
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
    let title = gtk::Label::new(Some("识别结果"));
    title.add_css_class("heading");
    title.set_xalign(0.0);
    top.append(&title);
    let spacer = gtk::Box::new(gtk::Orientation::Horizontal, 0);
    spacer.set_hexpand(true);
    top.append(&spacer);
    let count = gtk::Label::new(Some("8 行"));
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
    buffer.set_text(&demo_text());
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
    stack.add_titled(&text_scroller, Some("text"), "文本");

    let lines_page = gtk::Box::new(gtk::Orientation::Vertical, 10);
    lines_page.set_margin_start(12);
    lines_page.set_margin_end(12);
    lines_page.set_margin_bottom(12);
    let search = gtk::SearchEntry::builder()
        .placeholder_text("搜索文本行")
        .build();
    lines_page.append(&search);
    let line_list = gtk::ListBox::new();
    line_list.set_selection_mode(gtk::SelectionMode::Single);
    line_list.add_css_class("boxed-list");

    let mut rows = Vec::with_capacity(DEMO_LINES.len());
    for (index, line) in DEMO_LINES.iter().enumerate() {
        let row = adw::ActionRow::builder()
            .title(line.text)
            .subtitle(format!(
                "文本行 {:02}  ·  置信度 {:.1}%",
                index + 1,
                line.confidence * 100.0
            ))
            .activatable(true)
            .build();
        line_list.append(&row);
        rows.push((row, line.text.to_lowercase()));
    }
    let rows = Rc::new(rows);
    let rows_for_search = rows.clone();
    search.connect_search_changed(move |entry| {
        let query = entry.text().trim().to_lowercase();
        for (row, text) in rows_for_search.iter() {
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
    stack.add_titled(&lines_page, Some("lines"), "文本行");

    root.append(&stack);
    root.append(&gtk::Separator::new(gtk::Orientation::Horizontal));

    let footer = gtk::Box::new(gtk::Orientation::Horizontal, 10);
    footer.set_margin_top(10);
    footer.set_margin_bottom(12);
    footer.set_margin_start(14);
    footer.set_margin_end(14);
    let char_count = gtk::Label::new(Some(&format!("{} 个字符", demo_text().chars().count())));
    char_count.add_css_class("dim-label");
    footer.append(&char_count);
    let footer_spacer = gtk::Box::new(gtk::Orientation::Horizontal, 0);
    footer_spacer.set_hexpand(true);
    footer.append(&footer_spacer);
    let restore = gtk::Button::with_label("恢复结果");
    restore.add_css_class("flat");
    footer.append(&restore);
    let copy = gtk::Button::builder()
        .label("复制全部")
        .icon_name("edit-copy-symbolic")
        .build();
    footer.append(&copy);
    root.append(&footer);

    let count_for_buffer = char_count.clone();
    buffer.connect_changed(move |buffer| {
        let text = buffer.text(&buffer.start_iter(), &buffer.end_iter(), true);
        count_for_buffer.set_label(&format!("{} 个字符", text.chars().count()));
    });

    let buffer_for_copy = buffer.clone();
    let toasts_for_copy = toasts.clone();
    copy.connect_clicked(move |_| {
        copy_buffer(&buffer_for_copy);
        add_toast(&toasts_for_copy, "识别文本已复制到剪贴板");
    });

    let buffer_for_restore = buffer.clone();
    let toasts_for_restore = toasts.clone();
    let status_for_restore = status.clone();
    restore.connect_clicked(move |_| {
        buffer_for_restore.set_text(&demo_text());
        status_for_restore.set_label("已恢复界面演示结果");
        add_toast(&toasts_for_restore, "已恢复原始识别文本");
    });

    ResultsPane {
        root,
        stack,
        line_list,
        buffer,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_header(
    app: &adw::Application,
    window: &adw::ApplicationWindow,
    toasts: &adw::ToastOverlay,
    status: &gtk::Label,
    runtime_summary: &gtk::Label,
    buffer: &gtk::TextBuffer,
    settings_state: Rc<SettingsState>,
    split_view: &adw::OverlaySplitView,
) -> adw::HeaderBar {
    let header = adw::HeaderBar::new();
    let title = adw::WindowTitle::builder()
        .title("Paddle OCR")
        .subtitle("原生 GTK4 前端")
        .build();
    header.set_title_widget(Some(&title));

    let capture = button_content("截图识别", "edit-select-all-symbolic");
    capture.add_css_class("suggested-action");
    capture.set_action_name(Some("win.capture"));
    capture.set_tooltip_text(Some("框选屏幕区域并识别  Ctrl+Shift+O"));
    header.pack_start(&capture);

    let open = button_content("打开图片", "document-open-symbolic");
    open.set_action_name(Some("win.open"));
    open.set_tooltip_text(Some("选择图片文件  Ctrl+O"));
    header.pack_start(&open);

    let preferences = gtk::Button::from_icon_name("emblem-system-symbolic");
    preferences.add_css_class("flat");
    preferences.set_action_name(Some("win.preferences"));
    preferences.set_tooltip_text(Some("模型、产线与界面设置  Ctrl+,"));
    header.pack_end(&preferences);

    let more = build_more_menu(window, toasts, status, buffer);
    header.pack_end(&more);
    let export = build_export_menu(toasts, status);
    header.pack_end(&export);

    let copy = button_content("复制", "edit-copy-symbolic");
    copy.set_action_name(Some("win.copy"));
    copy.set_tooltip_text(Some("复制全部识别文本  Ctrl+Shift+C"));
    header.pack_end(&copy);

    let results = button_content("结果", "sidebar-show-symbolic");
    results.set_action_name(Some("win.toggle-results"));
    results.set_tooltip_text(Some("显示或隐藏识别结果"));
    header.pack_end(&results);

    install_window_actions(
        app,
        window,
        toasts,
        status,
        runtime_summary,
        buffer,
        settings_state,
        split_view,
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
    settings_state: Rc<SettingsState>,
    split_view: &adw::OverlaySplitView,
) {
    let open = gio::SimpleAction::new("open", None);
    let buffer_for_open = buffer.clone();
    let toasts_for_open = toasts.clone();
    let status_for_open = status.clone();
    open.connect_activate(move |_, _| {
        buffer_for_open.set_text(&demo_text());
        status_for_open.set_label("打开图片 · 等待 IPC 后端接入");
        add_toast(&toasts_for_open, "文件选择将在下一阶段接入；已载入演示文档");
    });
    window.add_action(&open);

    let capture = gio::SimpleAction::new("capture", None);
    let buffer_for_capture = buffer.clone();
    let toasts_for_capture = toasts.clone();
    let status_for_capture = status.clone();
    capture.connect_activate(move |_, _| {
        buffer_for_capture.set_text(&demo_text());
        status_for_capture.set_label("截图识别 · 等待 IPC 后端接入");
        add_toast(
            &toasts_for_capture,
            "截图框选将在下一阶段接入；已载入演示结果",
        );
    });
    window.add_action(&capture);

    let copy = gio::SimpleAction::new("copy", None);
    let buffer_for_copy = buffer.clone();
    let toasts_for_copy = toasts.clone();
    copy.connect_activate(move |_, _| {
        copy_buffer(&buffer_for_copy);
        add_toast(&toasts_for_copy, "识别文本已复制到剪贴板");
    });
    window.add_action(&copy);

    let preferences = gio::SimpleAction::new("preferences", None);
    let window_for_preferences = window.clone();
    let runtime_for_preferences = runtime_summary.clone();
    preferences.connect_activate(move |_, _| {
        settings::present_settings(
            &window_for_preferences,
            settings_state.clone(),
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

    app.set_accels_for_action("win.open", &["<Primary>O"]);
    app.set_accels_for_action("win.capture", &["<Primary><Shift>O"]);
    app.set_accels_for_action("win.copy", &["<Primary><Shift>C"]);
    app.set_accels_for_action("win.preferences", &["<Primary>comma"]);
}

fn build_export_menu(toasts: &adw::ToastOverlay, status: &gtk::Label) -> gtk::MenuButton {
    let button = gtk::MenuButton::new();
    let content = adw::ButtonContent::builder()
        .icon_name("document-save-symbolic")
        .label("导出")
        .can_shrink(true)
        .build();
    button.set_child(Some(&content));
    button.set_tooltip_text(Some("导出识别结果"));

    let popover = gtk::Popover::new();
    let box_ = gtk::Box::new(gtk::Orientation::Vertical, 4);
    box_.set_margin_top(10);
    box_.set_margin_bottom(10);
    box_.set_margin_start(10);
    box_.set_margin_end(10);
    box_.set_size_request(240, -1);
    let heading = gtk::Label::new(Some("导出识别结果"));
    heading.set_xalign(0.0);
    heading.add_css_class("heading");
    heading.set_margin_bottom(4);
    box_.append(&heading);
    for format in EXPORT_FORMATS {
        let item = gtk::Button::with_label(format);
        item.set_halign(gtk::Align::Fill);
        item.add_css_class("flat");
        let format = (*format).to_owned();
        let toasts = toasts.clone();
        let status = status.clone();
        let popover = popover.clone();
        item.connect_clicked(move |_| {
            status.set_label(&format!("已选择 {format} · 等待 IPC 后端接入"));
            add_toast(&toasts, &format!("{format} 导出将在后端接入后启用"));
            popover.popdown();
        });
        box_.append(&item);
    }
    box_.append(&gtk::Separator::new(gtk::Orientation::Horizontal));
    let pdf = gtk::Button::with_label("可搜索 PDF（后续阶段）");
    pdf.set_sensitive(false);
    pdf.set_tooltip_text(Some("需要 PDF 页面渲染与文本层坐标映射"));
    box_.append(&pdf);
    popover.set_child(Some(&box_));
    button.set_popover(Some(&popover));
    button
}

fn build_more_menu(
    window: &adw::ApplicationWindow,
    toasts: &adw::ToastOverlay,
    status: &gtk::Label,
    buffer: &gtk::TextBuffer,
) -> gtk::MenuButton {
    let button = gtk::MenuButton::new();
    button.set_icon_name("open-menu-symbolic");
    button.add_css_class("flat");
    button.set_tooltip_text(Some("更多选项"));

    let popover = gtk::Popover::new();
    let box_ = gtk::Box::new(gtk::Orientation::Vertical, 4);
    box_.set_margin_top(8);
    box_.set_margin_bottom(8);
    box_.set_margin_start(8);
    box_.set_margin_end(8);
    box_.set_size_request(210, -1);

    let reload = gtk::Button::with_label("重新载入界面演示");
    reload.add_css_class("flat");
    let buffer_for_reload = buffer.clone();
    let toasts_for_reload = toasts.clone();
    let status_for_reload = status.clone();
    let popover_for_reload = popover.clone();
    reload.connect_clicked(move |_| {
        buffer_for_reload.set_text(&demo_text());
        status_for_reload.set_label("界面演示 · 尚未连接识别后端");
        add_toast(&toasts_for_reload, "已重新载入演示数据");
        popover_for_reload.popdown();
    });
    box_.append(&reload);
    box_.append(&gtk::Separator::new(gtk::Orientation::Horizontal));

    let about = gtk::Button::with_label("关于 Paddle OCR");
    about.add_css_class("flat");
    let window_for_about = window.clone();
    about.connect_clicked(move |_| {
        let dialog = adw::AboutDialog::builder()
            .application_name("Paddle OCR")
            .application_icon("image-text-symbolic")
            .developer_name("ppocr-rs contributors")
            .version("0.1.0")
            .comments("基于 Rust、GTK4 和 libadwaita 的 PaddleOCR 桌面前端")
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

fn copy_buffer(buffer: &gtk::TextBuffer) {
    let text = buffer.text(&buffer.start_iter(), &buffer.end_iter(), true);
    if let Some(display) = gdk::Display::default() {
        display.clipboard().set_text(&text);
    }
}

fn add_toast(overlay: &adw::ToastOverlay, message: &str) {
    overlay.add_toast(adw::Toast::builder().title(message).timeout(3).build());
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
