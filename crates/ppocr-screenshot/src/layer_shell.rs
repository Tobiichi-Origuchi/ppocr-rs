use gtk4_layer_shell::{Edge, KeyboardMode, Layer, LayerShell};
use ppocr_i18n::tr;

pub fn configure_overlay(
    window: &gtk::Window,
    monitor: Option<&gtk::gdk::Monitor>,
) -> Result<(), String> {
    if !gtk4_layer_shell::is_supported() {
        return Err(tr!(
            "The current Wayland compositor does not support wlr-layer-shell, so a screenshot selection overlay cannot be created"
        ));
    }

    // This must happen before GTK realizes the window. An overlay layer
    // surface does not enter niri's scrolling tiling layout, and an exclusive
    // zone of -1 lets it cover panels and other surfaces that reserve space.
    window.init_layer_shell();
    window.set_namespace(Some("ppocr-screenshot-selector"));
    window.set_layer(Layer::Overlay);
    window.set_exclusive_zone(-1);
    window.set_keyboard_mode(KeyboardMode::Exclusive);
    for edge in [Edge::Left, Edge::Right, Edge::Top, Edge::Bottom] {
        window.set_anchor(edge, true);
    }
    window.set_monitor(monitor);
    Ok(())
}
