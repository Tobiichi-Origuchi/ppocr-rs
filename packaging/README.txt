ppocr-rs portable archive
=========================

Run the application from the extracted directory:

    ./ppocr-gui

This is an ORT CPU package. ONNX Runtime is included in the worker (statically
on the current target, or under lib/ on targets that require a shared
runtime). OCR models are still downloaded lazily into the platform's normal
cache directory when a selected pipeline first needs them.

The Linux desktop stack is intentionally supplied by the operating system,
not copied into this archive. The host needs GTK 4.14 or newer, libadwaita 1.5
or newer, GStreamer 1.28 with its GL/base/PipeWire plugins,
gtk4-layer-shell 0.8 or newer, and a working XDG desktop portal. The current
release is intended for recent x86_64 GNU/Linux distributions and Wayland.

Always use the top-level ppocr-gui launcher. It sets the private ONNX Runtime
library path and the bundled gettext catalog before starting the GUI. The
ppocr-worker and ppocr-screencopy files in bin/ are internal helper processes.

PACKAGE-METADATA records the exact source revision, Rust target, and inference
backend. SHA256SUMS verifies all regular files in the extracted bundle. The
separate .tar.gz.sha256 file verifies the archive before extraction.
