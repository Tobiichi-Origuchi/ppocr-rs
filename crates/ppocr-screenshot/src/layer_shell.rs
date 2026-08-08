use gtk4_layer_shell::{Edge, KeyboardMode, Layer, LayerShell};

pub fn configure_overlay(
    window: &gtk::Window,
    monitor: Option<&gtk::gdk::Monitor>,
) -> Result<(), String> {
    if !gtk4_layer_shell::is_supported() {
        return Err(
            "当前 Wayland compositor 不支持 wlr-layer-shell；无法创建不参与平铺布局的截图选择层"
                .to_owned(),
        );
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
