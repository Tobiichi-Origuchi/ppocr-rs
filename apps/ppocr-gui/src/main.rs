#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod backend;
mod config;
mod data;
mod preview;
mod settings;
mod window;

use adw::prelude::*;
use gtk::glib;

fn main() -> glib::ExitCode {
    ppocr_i18n::init().expect("failed to initialize translations");
    let application = adw::Application::builder()
        .application_id("io.github.ppocr_rs.PaddleOCR")
        .build();
    application.connect_activate(window::build_ui);
    application.run()
}
