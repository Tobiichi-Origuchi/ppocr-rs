use std::{
    cell::{Cell, RefCell},
    io::Cursor,
    rc::Rc,
};

use gdk_pixbuf::{Pixbuf, PixbufLoader};
use gtk::{cairo, gdk, glib, prelude::*};
use ppocr_i18n::tr;

const HANDLE_RADIUS: f64 = 5.0;
const HANDLE_HIT_RADIUS: f64 = 12.0;
const MIN_SELECTION_SIZE: f64 = 4.0;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Rect {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

impl Rect {
    fn from_points(a: Point, b: Point) -> Self {
        let x = a.x.min(b.x);
        let y = a.y.min(b.y);
        Self {
            x,
            y,
            width: (a.x - b.x).abs(),
            height: (a.y - b.y).abs(),
        }
    }

    fn right(self) -> f64 {
        self.x + self.width
    }

    fn bottom(self) -> f64 {
        self.y + self.height
    }

    fn contains(self, point: Point) -> bool {
        point.x >= self.x
            && point.x <= self.right()
            && point.y >= self.y
            && point.y <= self.bottom()
    }

    fn clamp(self, width: f64, height: f64) -> Self {
        let x = self.x.clamp(0.0, width);
        let y = self.y.clamp(0.0, height);
        let right = self.right().clamp(x, width);
        let bottom = self.bottom().clamp(y, height);
        Self {
            x,
            y,
            width: right - x,
            height: bottom - y,
        }
    }

    fn is_usable(self) -> bool {
        self.width >= MIN_SELECTION_SIZE && self.height >= MIN_SELECTION_SIZE
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Point {
    x: f64,
    y: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct Placement {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    image_width: f64,
    image_height: f64,
}

impl Placement {
    fn for_size(canvas_width: f64, canvas_height: f64, image: &Pixbuf) -> Self {
        Self::for_dimensions(
            canvas_width,
            canvas_height,
            f64::from(image.width()),
            f64::from(image.height()),
        )
    }

    fn for_dimensions(
        canvas_width: f64,
        canvas_height: f64,
        image_width: f64,
        image_height: f64,
    ) -> Self {
        // Fractional display scaling can make the logical allocation and the
        // physical output frame differ by one rounded pixel. Treat nearly
        // identical ratios as the same output; retain letterboxing for a
        // genuinely different source aspect ratio.
        let canvas_aspect = canvas_width / canvas_height;
        let image_aspect = image_width / image_height;
        if ((canvas_aspect - image_aspect) / image_aspect).abs() <= 0.005 {
            return Self {
                x: 0.0,
                y: 0.0,
                width: canvas_width,
                height: canvas_height,
                image_width,
                image_height,
            };
        }
        let scale = (canvas_width / image_width)
            .min(canvas_height / image_height)
            .max(f64::EPSILON);
        let width = image_width * scale;
        let height = image_height * scale;
        Self {
            x: (canvas_width - width) / 2.0,
            y: (canvas_height - height) / 2.0,
            width,
            height,
            image_width,
            image_height,
        }
    }

    fn to_image(self, point: Point) -> Option<Point> {
        if point.x < self.x
            || point.x > self.x + self.width
            || point.y < self.y
            || point.y > self.y + self.height
        {
            return None;
        }
        Some(Point {
            x: ((point.x - self.x) / self.width * self.image_width).clamp(0.0, self.image_width),
            y: ((point.y - self.y) / self.height * self.image_height).clamp(0.0, self.image_height),
        })
    }

    fn to_canvas(self, rect: Rect) -> Rect {
        Rect {
            x: self.x + rect.x / self.image_width * self.width,
            y: self.y + rect.y / self.image_height * self.height,
            width: rect.width / self.image_width * self.width,
            height: rect.height / self.image_height * self.height,
        }
    }

    fn image_units_per_canvas_pixel(self) -> f64 {
        self.image_width / self.width
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Handle {
    NorthWest,
    North,
    NorthEast,
    East,
    SouthEast,
    South,
    SouthWest,
    West,
}

#[derive(Clone, Copy, Debug)]
enum Interaction {
    New {
        anchor: Point,
    },
    Move {
        anchor: Point,
        original: Rect,
    },
    Resize {
        anchor: Point,
        original: Rect,
        handle: Handle,
    },
}

#[derive(Default)]
struct SelectorState {
    selection: Option<Rect>,
    placement: Placement,
    interaction: Option<Interaction>,
}

pub async fn select_region(
    parent: &gtk::Window,
    monitor: Option<&gdk::Monitor>,
    encoded_image: Vec<u8>,
) -> Result<Option<Vec<u8>>, String> {
    let image = decode(&encoded_image)?;
    let selector = gtk::Window::builder()
        .title(tr!("Select capture region"))
        .decorated(false)
        .build();
    if let Some(application) = parent.application() {
        selector.set_application(Some(&application));
    }
    install_css(&gtk::prelude::WidgetExt::display(&selector));

    let state = Rc::new(RefCell::new(SelectorState::default()));
    let (sender, receiver) = async_channel::bounded::<Option<Rect>>(1);
    let sender = Rc::new(RefCell::new(Some(sender)));
    let drawing = build_drawing_area(&image, &state);
    let root = gtk::Overlay::new();
    root.set_child(Some(&drawing));

    let hint = gtk::Label::new(Some(&tr!(
        "Drag to select · Move or drag handles to adjust · Enter to confirm · Esc to cancel"
    )));
    hint.add_css_class("ppocr-screenshot-hint");
    hint.set_halign(gtk::Align::Center);
    hint.set_valign(gtk::Align::Start);
    hint.set_margin_top(24);
    root.add_overlay(&hint);

    let size_label = gtk::Label::new(Some(&tr!("Drag to select a capture region")));
    size_label.add_css_class("dim-label");
    let cancel = gtk::Button::with_label(&tr!("Cancel"));
    let full = gtk::Button::with_label(&tr!("Select entire desktop"));
    let accept = gtk::Button::builder()
        .label(tr!("Confirm selected region"))
        .icon_name("object-select-symbolic")
        .sensitive(false)
        .build();
    accept.add_css_class("suggested-action");

    let controls = gtk::Box::new(gtk::Orientation::Horizontal, 10);
    controls.add_css_class("ppocr-screenshot-controls");
    controls.set_halign(gtk::Align::Center);
    controls.set_valign(gtk::Align::End);
    controls.set_margin_bottom(28);
    controls.append(&size_label);
    controls.append(&gtk::Separator::new(gtk::Orientation::Vertical));
    controls.append(&cancel);
    controls.append(&full);
    controls.append(&accept);
    root.add_overlay(&controls);

    connect_selection_gestures(&drawing, &state, &size_label, &accept);

    let selector_for_cancel = selector.clone();
    let sender_for_cancel = sender.clone();
    cancel.connect_clicked(move |_| {
        finish(&selector_for_cancel, &sender_for_cancel, None);
    });

    let state_for_full = state.clone();
    let drawing_for_full = drawing.clone();
    let size_for_full = size_label.clone();
    let accept_for_full = accept.clone();
    full.connect_clicked(move |_| {
        let placement = state_for_full.borrow().placement;
        if placement.image_width <= 0.0 || placement.image_height <= 0.0 {
            return;
        }
        let selection = Rect {
            x: 0.0,
            y: 0.0,
            width: placement.image_width,
            height: placement.image_height,
        };
        state_for_full.borrow_mut().selection = Some(selection);
        update_selection_ui(selection, &size_for_full, &accept_for_full);
        drawing_for_full.queue_draw();
    });

    let selector_for_accept = selector.clone();
    let sender_for_accept = sender.clone();
    let state_for_accept = state.clone();
    accept.connect_clicked(move |_| {
        let selection = state_for_accept.borrow().selection;
        finish(&selector_for_accept, &sender_for_accept, selection);
    });

    let key = gtk::EventControllerKey::new();
    let selector_for_key = selector.clone();
    let sender_for_key = sender.clone();
    let state_for_key = state.clone();
    key.connect_key_pressed(move |_, key, _, _| match key {
        gdk::Key::Escape => {
            finish(&selector_for_key, &sender_for_key, None);
            glib::Propagation::Stop
        }
        gdk::Key::Return | gdk::Key::KP_Enter => {
            if let Some(selection) = state_for_key.borrow().selection
                && selection.is_usable()
            {
                finish(&selector_for_key, &sender_for_key, Some(selection));
                return glib::Propagation::Stop;
            }
            glib::Propagation::Proceed
        }
        _ => glib::Propagation::Proceed,
    });
    selector.add_controller(key);

    let sender_for_close = sender.clone();
    selector.connect_close_request(move |_| {
        if let Some(sender) = sender_for_close.borrow_mut().take() {
            let _ = sender.try_send(None);
        }
        glib::Propagation::Proceed
    });

    selector.set_child(Some(&root));

    // A regular xdg_toplevel becomes another niri scrolling-layout column and
    // changes the very desktop arrangement represented by the frozen image.
    // An overlay layer surface is outside the tiling layout, covers panels and
    // compositor decorations, and therefore preserves screen coordinates.
    #[cfg(target_os = "linux")]
    super::layer_shell::configure_overlay(&selector, monitor)?;
    #[cfg(not(target_os = "linux"))]
    if let Some(monitor) = monitor {
        selector.fullscreen_on_monitor(monitor);
    } else {
        selector.fullscreen();
    }
    selector.present();
    selector.grab_focus();
    let selected = receiver.recv().await.map_err(|error| error.to_string())?;
    let Some(selected) = selected else {
        return Ok(None);
    };
    crop_png(&image, selected).map(Some)
}

fn install_css(display: &gdk::Display) {
    thread_local! {
        static INSTALLED: Cell<bool> = const { Cell::new(false) };
    }

    INSTALLED.with(|installed| {
        if installed.replace(true) {
            return;
        }
        let provider = gtk::CssProvider::new();
        provider.load_from_string(
            r#"
            .ppocr-screenshot-hint,
            .ppocr-screenshot-controls {
                color: white;
                background-color: rgba(24, 24, 27, 0.92);
                border: 1px solid rgba(255, 255, 255, 0.18);
                border-radius: 12px;
                padding: 10px 14px;
                box-shadow: 0 8px 28px rgba(0, 0, 0, 0.35);
            }
            .ppocr-screenshot-hint {
                font-weight: 600;
            }
            .ppocr-screenshot-controls .dim-label {
                color: rgba(255, 255, 255, 0.78);
            }
            "#,
        );
        gtk::style_context_add_provider_for_display(
            display,
            &provider,
            gtk::STYLE_PROVIDER_PRIORITY_APPLICATION,
        );
    });
}

fn decode(bytes: &[u8]) -> Result<Pixbuf, String> {
    let loader = PixbufLoader::new();
    loader.write(bytes).map_err(|error| error.to_string())?;
    loader.close().map_err(|error| error.to_string())?;
    loader
        .pixbuf()
        .ok_or_else(|| tr!("The capture decoder returned no image"))
}

fn crop_png(image: &Pixbuf, selection: Rect) -> Result<Vec<u8>, String> {
    let selection = selection.clamp(f64::from(image.width()), f64::from(image.height()));
    if !selection.is_usable() {
        return Err(tr!("The capture region is too small"));
    }
    let x = selection.x.floor() as i32;
    let y = selection.y.floor() as i32;
    let right = selection.right().ceil().min(f64::from(image.width())) as i32;
    let bottom = selection.bottom().ceil().min(f64::from(image.height())) as i32;
    encode_png(&image.new_subpixbuf(x, y, right - x, bottom - y))
}

fn encode_png(image: &Pixbuf) -> Result<Vec<u8>, String> {
    use image::{ExtendedColorType, ImageEncoder, codecs::png::PngEncoder};

    let width = image.width() as usize;
    let height = image.height() as usize;
    let channels = image.n_channels() as usize;
    let rowstride = image.rowstride() as usize;
    let pixels = image.read_pixel_bytes();
    let pixels = pixels.as_ref();
    let mut rgba = Vec::with_capacity(width * height * 4);
    for y in 0..height {
        let row = &pixels[y * rowstride..y * rowstride + width * channels];
        for pixel in row.chunks_exact(channels) {
            rgba.extend_from_slice(&pixel[..3]);
            rgba.push(if channels == 4 { pixel[3] } else { u8::MAX });
        }
    }
    let mut encoded = Cursor::new(Vec::new());
    PngEncoder::new(&mut encoded)
        .write_image(&rgba, width as u32, height as u32, ExtendedColorType::Rgba8)
        .map_err(|error| error.to_string())?;
    Ok(encoded.into_inner())
}

fn build_drawing_area(image: &Pixbuf, state: &Rc<RefCell<SelectorState>>) -> gtk::DrawingArea {
    let drawing = gtk::DrawingArea::builder()
        .hexpand(true)
        .vexpand(true)
        .focusable(true)
        .build();
    drawing.set_cursor_from_name(Some("crosshair"));
    let image = image.clone();
    let state = state.clone();
    drawing.set_draw_func(move |_, cr, width, height| {
        draw_selector(cr, width, height, &image, &state);
    });
    drawing
}

fn connect_selection_gestures(
    drawing: &gtk::DrawingArea,
    state: &Rc<RefCell<SelectorState>>,
    size_label: &gtk::Label,
    accept: &gtk::Button,
) {
    let drag = gtk::GestureDrag::new();
    drag.set_button(gdk::BUTTON_PRIMARY);

    let state_for_begin = state.clone();
    drag.connect_drag_begin(move |gesture, x, y| {
        let mut state = state_for_begin.borrow_mut();
        let Some(point) = state.placement.to_image(Point { x, y }) else {
            gesture.set_state(gtk::EventSequenceState::Denied);
            return;
        };
        let tolerance = HANDLE_HIT_RADIUS * state.placement.image_units_per_canvas_pixel();
        state.interaction = match state.selection {
            Some(selection) => match hit_handle(selection, point, tolerance) {
                Some(handle) => Some(Interaction::Resize {
                    anchor: point,
                    original: selection,
                    handle,
                }),
                None if selection.contains(point) => Some(Interaction::Move {
                    anchor: point,
                    original: selection,
                }),
                None => {
                    state.selection = None;
                    Some(Interaction::New { anchor: point })
                }
            },
            None => Some(Interaction::New { anchor: point }),
        };
    });

    let state_for_update = state.clone();
    let drawing_for_update = drawing.clone();
    let size_for_update = size_label.clone();
    let accept_for_update = accept.clone();
    drag.connect_drag_update(move |gesture, offset_x, offset_y| {
        let Some((start_x, start_y)) = gesture.start_point() else {
            return;
        };
        let mut state = state_for_update.borrow_mut();
        let Some(interaction) = state.interaction else {
            return;
        };
        let canvas_point = Point {
            x: start_x + offset_x,
            y: start_y + offset_y,
        };
        let point = state
            .placement
            .to_image(canvas_point)
            .unwrap_or_else(|| clamp_to_image(state.placement, canvas_point));
        let selection = update_interaction(
            interaction,
            point,
            state.placement.image_width,
            state.placement.image_height,
        );
        state.selection = Some(selection);
        drop(state);
        update_selection_ui(selection, &size_for_update, &accept_for_update);
        drawing_for_update.queue_draw();
    });

    let state_for_end = state.clone();
    let drawing_for_end = drawing.clone();
    let size_for_end = size_label.clone();
    let accept_for_end = accept.clone();
    drag.connect_drag_end(move |_, _, _| {
        let mut state = state_for_end.borrow_mut();
        state.interaction = None;
        if state
            .selection
            .is_some_and(|selection| !selection.is_usable())
        {
            state.selection = None;
            size_for_end.set_label(&tr!(
                "The capture region is too small; please select it again"
            ));
            accept_for_end.set_sensitive(false);
        }
        drawing_for_end.queue_draw();
    });
    drawing.add_controller(drag);
}

fn update_interaction(interaction: Interaction, point: Point, width: f64, height: f64) -> Rect {
    match interaction {
        Interaction::New { anchor } => Rect::from_points(anchor, point).clamp(width, height),
        Interaction::Move { anchor, original } => {
            let mut x = original.x + point.x - anchor.x;
            let mut y = original.y + point.y - anchor.y;
            x = x.clamp(0.0, (width - original.width).max(0.0));
            y = y.clamp(0.0, (height - original.height).max(0.0));
            Rect { x, y, ..original }
        }
        Interaction::Resize {
            anchor,
            original,
            handle,
        } => resize(
            original,
            point.x - anchor.x,
            point.y - anchor.y,
            handle,
            width,
            height,
        ),
    }
}

fn resize(original: Rect, dx: f64, dy: f64, handle: Handle, width: f64, height: f64) -> Rect {
    let mut left = original.x;
    let mut top = original.y;
    let mut right = original.right();
    let mut bottom = original.bottom();
    if matches!(handle, Handle::NorthWest | Handle::West | Handle::SouthWest) {
        left = (left + dx).clamp(0.0, right - MIN_SELECTION_SIZE);
    }
    if matches!(handle, Handle::NorthEast | Handle::East | Handle::SouthEast) {
        right = (right + dx).clamp(left + MIN_SELECTION_SIZE, width);
    }
    if matches!(
        handle,
        Handle::NorthWest | Handle::North | Handle::NorthEast
    ) {
        top = (top + dy).clamp(0.0, bottom - MIN_SELECTION_SIZE);
    }
    if matches!(
        handle,
        Handle::SouthWest | Handle::South | Handle::SouthEast
    ) {
        bottom = (bottom + dy).clamp(top + MIN_SELECTION_SIZE, height);
    }
    Rect {
        x: left,
        y: top,
        width: right - left,
        height: bottom - top,
    }
}

fn hit_handle(selection: Rect, point: Point, tolerance: f64) -> Option<Handle> {
    handles(selection)
        .into_iter()
        .find(|(_, handle_point)| {
            (handle_point.x - point.x).abs() <= tolerance
                && (handle_point.y - point.y).abs() <= tolerance
        })
        .map(|(handle, _)| handle)
}

fn handles(rect: Rect) -> [(Handle, Point); 8] {
    let center_x = rect.x + rect.width / 2.0;
    let center_y = rect.y + rect.height / 2.0;
    [
        (
            Handle::NorthWest,
            Point {
                x: rect.x,
                y: rect.y,
            },
        ),
        (
            Handle::North,
            Point {
                x: center_x,
                y: rect.y,
            },
        ),
        (
            Handle::NorthEast,
            Point {
                x: rect.right(),
                y: rect.y,
            },
        ),
        (
            Handle::East,
            Point {
                x: rect.right(),
                y: center_y,
            },
        ),
        (
            Handle::SouthEast,
            Point {
                x: rect.right(),
                y: rect.bottom(),
            },
        ),
        (
            Handle::South,
            Point {
                x: center_x,
                y: rect.bottom(),
            },
        ),
        (
            Handle::SouthWest,
            Point {
                x: rect.x,
                y: rect.bottom(),
            },
        ),
        (
            Handle::West,
            Point {
                x: rect.x,
                y: center_y,
            },
        ),
    ]
}

fn clamp_to_image(placement: Placement, point: Point) -> Point {
    Point {
        x: ((point.x - placement.x) / placement.width * placement.image_width)
            .clamp(0.0, placement.image_width),
        y: ((point.y - placement.y) / placement.height * placement.image_height)
            .clamp(0.0, placement.image_height),
    }
}

fn update_selection_ui(selection: Rect, label: &gtk::Label, accept: &gtk::Button) {
    label.set_label(&tr!(
        "{width} × {height} pixels",
        width = selection.width.round() as u32,
        height = selection.height.round() as u32
    ));
    accept.set_sensitive(selection.is_usable());
}

fn finish(
    window: &gtk::Window,
    sender: &Rc<RefCell<Option<async_channel::Sender<Option<Rect>>>>>,
    selection: Option<Rect>,
) {
    if let Some(sender) = sender.borrow_mut().take() {
        let _ = sender.try_send(selection);
    }
    window.close();
}

fn draw_selector(
    cr: &cairo::Context,
    width: i32,
    height: i32,
    image: &Pixbuf,
    state: &Rc<RefCell<SelectorState>>,
) {
    cr.set_source_rgb(0.03, 0.03, 0.03);
    let _ = cr.paint();
    let placement = Placement::for_size(f64::from(width), f64::from(height), image);
    state.borrow_mut().placement = placement;

    let _ = cr.save();
    cr.translate(placement.x, placement.y);
    cr.scale(
        placement.width / f64::from(image.width()),
        placement.height / f64::from(image.height()),
    );
    gdk::prelude::GdkCairoContextExt::set_source_pixbuf(cr, image, 0.0, 0.0);
    let _ = cr.paint();
    let _ = cr.restore();

    let selection = state.borrow().selection;
    let canvas = selection.map(|selection| placement.to_canvas(selection));
    cr.rectangle(placement.x, placement.y, placement.width, placement.height);
    if let Some(rect) = canvas {
        cr.rectangle(rect.x, rect.y, rect.width, rect.height);
        cr.set_fill_rule(cairo::FillRule::EvenOdd);
    }
    cr.set_source_rgba(0.0, 0.0, 0.0, if canvas.is_some() { 0.58 } else { 0.30 });
    let _ = cr.fill();
    cr.set_fill_rule(cairo::FillRule::Winding);

    let Some(rect) = canvas else {
        return;
    };
    cr.rectangle(
        rect.x + 0.5,
        rect.y + 0.5,
        rect.width - 1.0,
        rect.height - 1.0,
    );
    cr.set_source_rgba(0.20, 0.58, 0.98, 1.0);
    cr.set_line_width(2.0);
    let _ = cr.stroke();

    for (_, point) in handles(rect) {
        cr.arc(point.x, point.y, HANDLE_RADIUS, 0.0, std::f64::consts::TAU);
        cr.set_source_rgb(1.0, 1.0, 1.0);
        let _ = cr.fill_preserve();
        cr.set_source_rgba(0.20, 0.58, 0.98, 1.0);
        cr.set_line_width(2.0);
        let _ = cr.stroke();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reverse_drag_is_normalized() {
        assert_eq!(
            Rect::from_points(Point { x: 90.0, y: 70.0 }, Point { x: 10.0, y: 20.0 }),
            Rect {
                x: 10.0,
                y: 20.0,
                width: 80.0,
                height: 50.0,
            }
        );
    }

    #[test]
    fn moving_selection_stays_inside_image() {
        let original = Rect {
            x: 20.0,
            y: 30.0,
            width: 40.0,
            height: 50.0,
        };
        let moved = update_interaction(
            Interaction::Move {
                anchor: Point { x: 25.0, y: 35.0 },
                original,
            },
            Point { x: 500.0, y: 500.0 },
            100.0,
            120.0,
        );
        assert_eq!(moved.x, 60.0);
        assert_eq!(moved.y, 70.0);
    }

    #[test]
    fn resizing_enforces_minimum_and_bounds() {
        let original = Rect {
            x: 20.0,
            y: 20.0,
            width: 30.0,
            height: 30.0,
        };
        let resized = resize(original, 100.0, 100.0, Handle::NorthWest, 80.0, 80.0);
        assert_eq!(resized.x, 46.0);
        assert_eq!(resized.y, 46.0);
        assert_eq!(resized.width, MIN_SELECTION_SIZE);
        assert_eq!(resized.height, MIN_SELECTION_SIZE);

        let expanded = resize(original, 100.0, 100.0, Handle::SouthEast, 80.0, 80.0);
        assert_eq!(expanded.right(), 80.0);
        assert_eq!(expanded.bottom(), 80.0);
    }

    #[test]
    fn crop_rounding_never_exceeds_image() {
        let selection = Rect {
            x: -2.0,
            y: 8.2,
            width: 25.9,
            height: 30.0,
        }
        .clamp(20.0, 20.0);
        assert_eq!(selection.x, 0.0);
        assert_eq!(selection.right(), 20.0);
        assert_eq!(selection.bottom(), 20.0);
    }

    #[test]
    fn handle_hit_testing_prefers_exact_handle() {
        let rect = Rect {
            x: 10.0,
            y: 20.0,
            width: 100.0,
            height: 60.0,
        };
        assert_eq!(
            hit_handle(rect, Point { x: 110.0, y: 50.0 }, 2.0),
            Some(Handle::East)
        );
        assert_eq!(hit_handle(rect, Point { x: 55.0, y: 50.0 }, 2.0), None);
    }

    #[test]
    fn fractional_scale_rounding_fills_the_canvas_without_translation() {
        let placement = Placement::for_dimensions(1707.0, 1067.0, 2560.0, 1600.0);
        assert_eq!(placement.x, 0.0);
        assert_eq!(placement.y, 0.0);
        assert_eq!(placement.width, 1707.0);
        assert_eq!(placement.height, 1067.0);
        assert_eq!(
            placement.to_image(Point {
                x: 1707.0,
                y: 1067.0,
            }),
            Some(Point {
                x: 2560.0,
                y: 1600.0,
            })
        );
    }

    #[test]
    fn genuinely_different_aspect_ratios_remain_letterboxed() {
        let placement = Placement::for_dimensions(1600.0, 900.0, 1000.0, 1000.0);
        assert_eq!(placement.x, 350.0);
        assert_eq!(placement.y, 0.0);
        assert_eq!(placement.width, 900.0);
        assert_eq!(placement.height, 900.0);
    }

    #[test]
    fn pixbuf_crop_encodes_png() {
        let image = Pixbuf::new(gdk_pixbuf::Colorspace::Rgb, true, 8, 32, 32)
            .expect("test pixbuf allocation must succeed");
        image.fill(0x4a90_e2ff);
        let crop = crop_png(
            &image,
            Rect {
                x: 1.0,
                y: 1.0,
                width: 20.0,
                height: 20.0,
            },
        )
        .expect("fixture crop must encode");
        assert!(crop.starts_with(b"\x89PNG\r\n\x1a\n"));
    }
}
