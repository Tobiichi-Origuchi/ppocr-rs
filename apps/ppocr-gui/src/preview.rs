use std::{
    cell::{Cell, RefCell},
    f64::consts::PI,
    io::Cursor,
    rc::Rc,
};

use gdk_pixbuf::Pixbuf;
use gtk::{cairo, gdk, prelude::*};
use ppocr_app_protocol::OcrMetadata;
use ppocr_i18n::tr;

use crate::data::RectF;

#[derive(Clone)]
pub struct PreviewPane {
    pub root: gtk::Box,
    pub drawing_area: gtk::DrawingArea,
    pub selected_line: Rc<Cell<Option<usize>>>,
    subtitle: gtk::Label,
    badge: gtk::Label,
    state: Rc<RefCell<PreviewState>>,
    processing_overlay: gtk::CenterBox,
    spinner: gtk::Spinner,
    processing_label: gtk::Label,
    progress_bar: gtk::ProgressBar,
}

#[derive(Default)]
struct PreviewState {
    original_image: Option<Pixbuf>,
    processed_image: Option<Pixbuf>,
    showing_processed: bool,
    source_name: String,
    metadata: Option<OcrMetadata>,
}

impl PreviewPane {
    pub fn set_source(
        &self,
        encoded_image: &[u8],
        source_name: &str,
    ) -> Result<(), gtk::glib::Error> {
        let loader = gdk_pixbuf::PixbufLoader::new();
        loader.write(encoded_image)?;
        loader.close()?;
        let image = loader.pixbuf().ok_or_else(|| {
            gtk::glib::Error::new(
                gdk_pixbuf::PixbufError::Failed,
                "image decoder returned no preview",
            )
        })?;
        self.subtitle.set_label(&format!(
            "{}  ·  {} × {}",
            source_name,
            image.width(),
            image.height()
        ));
        self.badge.set_label(&tr!("Ready to process"));
        self.badge.remove_css_class("accent");
        self.badge.remove_css_class("error");
        self.badge.add_css_class("dim-label");
        self.selected_line.set(None);
        self.state.replace(PreviewState {
            original_image: Some(image),
            processed_image: None,
            showing_processed: false,
            source_name: source_name.to_owned(),
            metadata: None,
        });
        self.spinner.stop();
        self.processing_overlay.set_visible(false);
        self.progress_bar.set_visible(false);
        self.drawing_area.queue_draw();
        Ok(())
    }

    pub fn set_processing(&self) {
        self.badge.set_label(&tr!("Processing"));
        self.badge.remove_css_class("accent");
        self.badge.remove_css_class("error");
        self.badge.add_css_class("dim-label");
        self.processing_label
            .set_label(&tr!("Preparing recognition…"));
        self.progress_bar.set_visible(false);
        self.spinner.start();
        self.processing_overlay.set_visible(true);
    }

    pub fn set_progress(&self, message: &str, fraction: Option<f64>) {
        self.processing_label.set_label(message);
        if let Some(fraction) = fraction {
            self.progress_bar.set_fraction(fraction.clamp(0.0, 1.0));
            self.progress_bar.set_visible(true);
        } else {
            self.progress_bar.set_visible(false);
        }
    }

    pub fn set_result(
        &self,
        processed_png: Vec<u8>,
        metadata: OcrMetadata,
        source_name: &str,
    ) -> Result<(), gtk::glib::Error> {
        let image = Pixbuf::from_read(Cursor::new(processed_png))?;
        self.subtitle.set_label(&format!(
            "{}  ·  {} × {}",
            source_name, metadata.processed_size.width, metadata.processed_size.height
        ));
        self.badge.set_label(&tr!("Recognition complete"));
        self.badge.remove_css_class("dim-label");
        self.badge.remove_css_class("error");
        self.badge.add_css_class("accent");
        self.spinner.stop();
        self.processing_overlay.set_visible(false);
        self.selected_line.set(None);
        let mut state = self.state.borrow_mut();
        state.processed_image = Some(image);
        state.showing_processed = true;
        state.source_name = source_name.to_owned();
        state.metadata = Some(metadata);
        drop(state);
        self.drawing_area.queue_draw();
        Ok(())
    }

    pub fn set_failed(&self) {
        self.badge.set_label(&tr!("Processing failed"));
        self.badge.remove_css_class("accent");
        self.badge.add_css_class("error");
        self.spinner.stop();
        self.processing_overlay.set_visible(false);
        self.progress_bar.set_visible(false);
    }
}

pub fn build_preview(
    on_line_selected: impl Fn(Option<usize>) + 'static,
    on_image_toggled: impl Fn(bool) + 'static,
) -> PreviewPane {
    let root = gtk::Box::new(gtk::Orientation::Vertical, 0);
    root.add_css_class("card");
    root.set_overflow(gtk::Overflow::Hidden);
    root.set_hexpand(true);
    root.set_vexpand(true);

    let heading = gtk::Box::new(gtk::Orientation::Horizontal, 10);
    heading.set_margin_top(14);
    heading.set_margin_bottom(12);
    heading.set_margin_start(16);
    heading.set_margin_end(16);

    let title_box = gtk::Box::new(gtk::Orientation::Vertical, 2);
    let title = gtk::Label::new(Some(&tr!("Document preview")));
    title.set_xalign(0.0);
    title.add_css_class("heading");
    let subtitle = gtk::Label::new(Some(&tr!("No image opened")));
    subtitle.set_xalign(0.0);
    subtitle.set_ellipsize(gtk::pango::EllipsizeMode::Middle);
    subtitle.add_css_class("dim-label");
    subtitle.add_css_class("caption");
    title_box.append(&title);
    title_box.append(&subtitle);
    heading.append(&title_box);

    let spacer = gtk::Box::new(gtk::Orientation::Horizontal, 0);
    spacer.set_hexpand(true);
    heading.append(&spacer);
    let badge = gtk::Label::new(Some(&tr!("Waiting for an image")));
    badge.add_css_class("dim-label");
    badge.add_css_class("pill");
    heading.append(&badge);
    root.append(&heading);

    let selected_line = Rc::new(Cell::new(None));
    let show_boxes = Rc::new(Cell::new(true));
    let zoom = Rc::new(Cell::new(1.0_f64));
    let page_rect = Rc::new(Cell::new(RectF::default()));
    let state = Rc::new(RefCell::new(PreviewState::default()));

    let drawing_area = gtk::DrawingArea::builder()
        .content_width(660)
        .content_height(760)
        .hexpand(true)
        .vexpand(true)
        .focusable(true)
        .build();
    drawing_area.add_css_class("preview-canvas");
    drawing_area.set_accessible_role(gtk::AccessibleRole::Img);
    let accessible_label = tr!("OCR document preview");
    drawing_area.update_property(&[gtk::accessible::Property::Label(&accessible_label)]);

    let selected_for_draw = selected_line.clone();
    let boxes_for_draw = show_boxes.clone();
    let zoom_for_draw = zoom.clone();
    let page_for_draw = page_rect.clone();
    let state_for_draw = state.clone();
    drawing_area.set_draw_func(move |_area, cr, width, height| {
        draw_preview(
            cr,
            width,
            height,
            zoom_for_draw.get(),
            boxes_for_draw.get(),
            selected_for_draw.get(),
            &page_for_draw,
            &state_for_draw.borrow(),
        );
    });

    let selected_for_click = selected_line.clone();
    let page_for_click = page_rect.clone();
    let drawing_for_click = drawing_area.clone();
    let state_for_click = state.clone();
    let badge_for_click = badge.clone();
    let subtitle_for_click = subtitle.clone();
    let gesture = gtk::GestureClick::new();
    gesture.connect_pressed(move |_, presses, x, y| {
        drawing_for_click.grab_focus();
        if presses == 2 {
            let (showing_processed, source_name, width, height) = {
                let mut state = state_for_click.borrow_mut();
                if state.processed_image.is_none() {
                    return;
                }
                state.showing_processed = !state.showing_processed;
                let image = if state.showing_processed {
                    state.processed_image.as_ref()
                } else {
                    state.original_image.as_ref()
                }
                .expect("an available preview image was checked above");
                (
                    state.showing_processed,
                    state.source_name.clone(),
                    image.width(),
                    image.height(),
                )
            };
            subtitle_for_click.set_label(&format!("{source_name}  ·  {width} × {height}"));
            badge_for_click.set_label(&if showing_processed {
                tr!("Processed image")
            } else {
                tr!("Original image")
            });
            selected_for_click.set(None);
            on_line_selected(None);
            on_image_toggled(showing_processed);
            drawing_for_click.queue_draw();
            return;
        }
        let page = page_for_click.get();
        let state = state_for_click.borrow();
        let selected = state
            .showing_processed
            .then_some(state.metadata.as_ref())
            .flatten()
            .and_then(|metadata| line_at(metadata, page, x, y));
        selected_for_click.set(selected);
        on_line_selected(selected);
        drawing_for_click.queue_draw();
    });
    drawing_area.add_controller(gesture);

    let scroller = gtk::ScrolledWindow::builder()
        .hscrollbar_policy(gtk::PolicyType::Automatic)
        .vscrollbar_policy(gtk::PolicyType::Automatic)
        .min_content_height(320)
        .hexpand(true)
        .vexpand(true)
        .child(&drawing_area)
        .build();
    let image_overlay = gtk::Overlay::new();
    image_overlay.set_hexpand(true);
    image_overlay.set_vexpand(true);
    image_overlay.set_child(Some(&scroller));

    let spinner = gtk::Spinner::new();
    spinner.set_size_request(34, 34);
    let processing_label = gtk::Label::new(Some(&tr!("Processing image…")));
    processing_label.add_css_class("processing-label");
    processing_label.set_ellipsize(gtk::pango::EllipsizeMode::Middle);
    processing_label.set_max_width_chars(64);
    let progress_bar = gtk::ProgressBar::new();
    progress_bar.set_size_request(360, -1);
    progress_bar.set_visible(false);
    let processing_content = gtk::Box::new(gtk::Orientation::Vertical, 10);
    processing_content.set_halign(gtk::Align::Center);
    processing_content.set_valign(gtk::Align::Center);
    processing_content.append(&spinner);
    processing_content.append(&processing_label);
    processing_content.append(&progress_bar);

    let processing_overlay = gtk::CenterBox::new();
    processing_overlay.add_css_class("processing-overlay");
    processing_overlay.set_halign(gtk::Align::Fill);
    processing_overlay.set_valign(gtk::Align::Fill);
    processing_overlay.set_center_widget(Some(&processing_content));
    processing_overlay.set_visible(false);
    image_overlay.add_overlay(&processing_overlay);
    root.append(&image_overlay);

    let controls = gtk::Box::new(gtk::Orientation::Horizontal, 10);
    controls.set_margin_top(10);
    controls.set_margin_bottom(12);
    controls.set_margin_start(14);
    controls.set_margin_end(14);

    let overlay_switch = gtk::CheckButton::with_label(&tr!("Show text boxes"));
    overlay_switch.set_active(true);
    let boxes_for_toggle = show_boxes.clone();
    let drawing_for_toggle = drawing_area.clone();
    overlay_switch.connect_toggled(move |button| {
        boxes_for_toggle.set(button.is_active());
        drawing_for_toggle.queue_draw();
    });
    controls.append(&overlay_switch);
    controls.append(&gtk::Separator::new(gtk::Orientation::Vertical));

    let zoom_out = gtk::Button::from_icon_name("zoom-out-symbolic");
    zoom_out.add_css_class("flat");
    zoom_out.set_tooltip_text(Some(&tr!("Zoom out")));
    controls.append(&zoom_out);

    let adjustment = gtk::Adjustment::new(100.0, 50.0, 180.0, 10.0, 20.0, 0.0);
    let scale = gtk::Scale::new(gtk::Orientation::Horizontal, Some(&adjustment));
    scale.set_draw_value(false);
    scale.set_size_request(150, -1);
    scale.set_tooltip_text(Some(&tr!("Preview zoom")));
    controls.append(&scale);

    let zoom_in = gtk::Button::from_icon_name("zoom-in-symbolic");
    zoom_in.add_css_class("flat");
    zoom_in.set_tooltip_text(Some(&tr!("Zoom in")));
    controls.append(&zoom_in);

    let fit = gtk::Button::with_label(&tr!("Fit to window"));
    fit.add_css_class("flat");
    controls.append(&fit);

    let zoom_label = gtk::Label::new(Some("100%"));
    zoom_label.add_css_class("dim-label");
    zoom_label.set_width_chars(5);
    controls.append(&zoom_label);

    let page_label = gtk::Label::new(Some(&tr!("Page 1 of 1")));
    page_label.set_hexpand(true);
    page_label.set_halign(gtk::Align::End);
    page_label.add_css_class("dim-label");
    controls.append(&page_label);

    let update_zoom = {
        let zoom = zoom.clone();
        let zoom_label = zoom_label.clone();
        let drawing_area = drawing_area.clone();
        move |value: f64| {
            zoom.set(value / 100.0);
            zoom_label.set_label(&format!("{value:.0}%"));
            drawing_area.set_content_width((660.0 * value / 100.0).round() as i32);
            drawing_area.set_content_height((760.0 * value / 100.0).round() as i32);
            drawing_area.queue_draw();
        }
    };
    let update_zoom = Rc::new(update_zoom);

    let update_from_scale = update_zoom.clone();
    scale.connect_value_changed(move |scale| update_from_scale(scale.value()));
    let adjustment_for_out = adjustment.clone();
    zoom_out.connect_clicked(move |_| {
        adjustment_for_out.set_value((adjustment_for_out.value() - 10.0).max(50.0));
    });
    let adjustment_for_in = adjustment.clone();
    zoom_in.connect_clicked(move |_| {
        adjustment_for_in.set_value((adjustment_for_in.value() + 10.0).min(180.0));
    });
    fit.connect_clicked(move |_| adjustment.set_value(100.0));

    root.append(&controls);

    PreviewPane {
        root,
        drawing_area,
        selected_line,
        subtitle,
        badge,
        state,
        processing_overlay,
        spinner,
        processing_label,
        progress_bar,
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_preview(
    cr: &cairo::Context,
    width: i32,
    height: i32,
    zoom: f64,
    show_boxes: bool,
    selected: Option<usize>,
    page_out: &Cell<RectF>,
    state: &PreviewState,
) {
    let canvas = if adw::StyleManager::default().is_dark() {
        gdk::RGBA::new(0.12, 0.12, 0.12, 1.0)
    } else {
        gdk::RGBA::new(0.91, 0.91, 0.90, 1.0)
    };
    set_source_rgba(cr, canvas);
    let _ = cr.paint();

    let image = if state.showing_processed {
        state.processed_image.as_ref()
    } else {
        state.original_image.as_ref()
    };
    let Some(image) = image else {
        page_out.set(RectF::default());
        draw_empty_state(cr, width, height);
        return;
    };

    let aspect = f64::from(image.width()) / f64::from(image.height());
    let available_width = (f64::from(width) / zoom - 54.0).max(240.0);
    let available_height = (f64::from(height) / zoom - 42.0).max(300.0);
    let base_height = available_height.min(720.0);
    let base_width = (base_height * aspect).min(available_width);
    let base_height = base_width / aspect;
    let page_width = base_width * zoom;
    let page_height = base_height * zoom;
    let page = RectF {
        x: (f64::from(width) - page_width) / 2.0,
        y: (f64::from(height) - page_height) / 2.0,
        width: page_width,
        height: page_height,
    };
    page_out.set(page);

    rounded_rectangle(cr, page.x + 2.0, page.y + 5.0, page.width, page.height, 8.0);
    cr.set_source_rgba(0.0, 0.0, 0.0, 0.20);
    let _ = cr.fill();

    let _ = cr.save();
    rounded_rectangle(cr, page.x, page.y, page.width, page.height, 8.0);
    cr.clip();
    cr.translate(page.x, page.y);
    cr.scale(
        page.width / f64::from(image.width()),
        page.height / f64::from(image.height()),
    );
    gdk::prelude::GdkCairoContextExt::set_source_pixbuf(cr, image, 0.0, 0.0);
    let _ = cr.paint();
    let _ = cr.restore();

    rounded_rectangle(cr, page.x, page.y, page.width, page.height, 8.0);
    cr.set_source_rgba(0.4, 0.4, 0.4, 0.28);
    cr.set_line_width(1.0);
    let _ = cr.stroke();

    if state.showing_processed
        && show_boxes
        && let Some(metadata) = state.metadata.as_ref()
    {
        draw_text_boxes(cr, page, metadata, selected);
    }
}

fn draw_empty_state(cr: &cairo::Context, width: i32, height: i32) {
    let layout = pangocairo::functions::create_layout(cr);
    layout.set_text(&tr!("Open an image to start recognition"));
    let font = gtk::pango::FontDescription::from_string("Sans 13");
    layout.set_font_description(Some(&font));
    let (text_width, text_height) = layout.pixel_size();
    cr.move_to(
        (f64::from(width - text_width) / 2.0).max(16.0),
        (f64::from(height - text_height) / 2.0).max(16.0),
    );
    cr.set_source_rgba(0.45, 0.45, 0.45, 0.95);
    pangocairo::functions::show_layout(cr, &layout);
}

fn draw_text_boxes(
    cr: &cairo::Context,
    page: RectF,
    metadata: &OcrMetadata,
    selected: Option<usize>,
) {
    let image_width = f64::from(metadata.processed_size.width.max(1));
    let image_height = f64::from(metadata.processed_size.height.max(1));
    for (index, line) in metadata.text_lines.iter().enumerate() {
        cr.new_path();
        for (point_index, point) in line.polygon.iter().enumerate() {
            let x = page.x + f64::from(point.x) / image_width * page.width;
            let y = page.y + f64::from(point.y) / image_height * page.height;
            if point_index == 0 {
                cr.move_to(x, y);
            } else {
                cr.line_to(x, y);
            }
        }
        cr.close_path();
        if selected == Some(index) {
            cr.set_source_rgba(0.21, 0.52, 0.90, 0.24);
            let _ = cr.fill_preserve();
            cr.set_line_width(2.0);
        } else {
            cr.set_source_rgba(0.21, 0.52, 0.90, 0.08);
            let _ = cr.fill_preserve();
            cr.set_line_width(1.0);
        }
        cr.set_source_rgba(0.21, 0.52, 0.90, 0.92);
        let _ = cr.stroke();
    }
}

fn line_at(metadata: &OcrMetadata, page: RectF, x: f64, y: f64) -> Option<usize> {
    if !page.contains(x, y) {
        return None;
    }
    let image_x = ((x - page.x) / page.width) * f64::from(metadata.processed_size.width);
    let image_y = ((y - page.y) / page.height) * f64::from(metadata.processed_size.height);
    metadata.text_lines.iter().position(|line| {
        let [left, top, right, bottom] = line.bounding_box;
        image_x >= f64::from(left)
            && image_x <= f64::from(right)
            && image_y >= f64::from(top)
            && image_y <= f64::from(bottom)
    })
}

fn set_source_rgba(cr: &cairo::Context, color: gdk::RGBA) {
    cr.set_source_rgba(
        color.red().into(),
        color.green().into(),
        color.blue().into(),
        color.alpha().into(),
    );
}

fn rounded_rectangle(cr: &cairo::Context, x: f64, y: f64, width: f64, height: f64, radius: f64) {
    let radius = radius.min(width / 2.0).min(height / 2.0);
    cr.new_sub_path();
    cr.arc(x + width - radius, y + radius, radius, -PI / 2.0, 0.0);
    cr.arc(
        x + width - radius,
        y + height - radius,
        radius,
        0.0,
        PI / 2.0,
    );
    cr.arc(x + radius, y + height - radius, radius, PI / 2.0, PI);
    cr.arc(x + radius, y + radius, radius, PI, 3.0 * PI / 2.0);
    cr.close_path();
}
