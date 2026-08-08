use std::{
    io::Cursor,
    os::fd::{AsRawFd, OwnedFd},
};

use gst_video::VideoFrameExt;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_gl as gst_gl;
use gstreamer_gl::prelude::*;
use gstreamer_video as gst_video;
use image::{ExtendedColorType, ImageEncoder, codecs::png::PngEncoder};
use ppocr_i18n::tr;

pub async fn capture_png(node_id: u32, fd: OwnedFd) -> Result<Vec<u8>, String> {
    let (sender, receiver) = async_channel::bounded(1);
    std::thread::Builder::new()
        .name("ppocr-pipewire-frame".to_owned())
        .spawn(move || {
            let result = capture_with_gstreamer(node_id, fd);
            let _ = sender.send_blocking(result);
        })
        .map_err(|error| {
            tr!(
                "Failed to start the PipeWire capture thread: {error}",
                error = error
            )
        })?;
    receiver.recv().await.map_err(|error| {
        tr!(
            "The PipeWire capture thread ended unexpectedly: {error}",
            error = error
        )
    })?
}

fn capture_with_gstreamer(node_id: u32, fd: OwnedFd) -> Result<Vec<u8>, String> {
    gst::init().map_err(|error| tr!("Failed to initialize GStreamer: {error}", error = error))?;

    let dma_buf_caps = dma_buf_import_caps()?;
    let source = gst::ElementFactory::make("pipewiresrc")
        .name("portal-source")
        .property("fd", fd.as_raw_fd())
        .property("path", node_id.to_string())
        .property("do-timestamp", true)
        .build()
        .map_err(|error| {
            tr!(
                "Failed to create the GStreamer PipeWire source: {error}",
                error = error
            )
        })?;
    let dma_buf_filter = gst::ElementFactory::make("capsfilter")
        .name("portal-dma-buf-format")
        .property("caps", &dma_buf_caps)
        .build()
        .map_err(|error| {
            tr!(
                "Failed to create the DMA-BUF format filter: {error}",
                error = error
            )
        })?;
    let upload = gst::ElementFactory::make("glupload")
        .build()
        .map_err(|error| {
            tr!(
                "Failed to create the GStreamer DMA-BUF importer: {error}",
                error = error
            )
        })?;
    let gl_convert = gst::ElementFactory::make("glcolorconvert")
        .build()
        .map_err(|error| {
            tr!(
                "Failed to create the GStreamer GL pixel converter: {error}",
                error = error
            )
        })?;
    let download = gst::ElementFactory::make("gldownload")
        .build()
        .map_err(|error| {
            tr!(
                "Failed to create the GStreamer GL downloader: {error}",
                error = error
            )
        })?;
    let convert = gst::ElementFactory::make("videoconvert")
        .build()
        .map_err(|error| {
            tr!(
                "Failed to create the GStreamer pixel converter: {error}",
                error = error
            )
        })?;
    let caps = gst::Caps::builder("video/x-raw")
        .field("format", "RGBA")
        .build();
    let sink = gst_app::AppSink::builder()
        .caps(&caps)
        .sync(false)
        .max_buffers(1)
        .drop(true)
        .build();

    let pipeline = gst::Pipeline::new();
    pipeline
        .add_many([
            &source,
            &dma_buf_filter,
            &upload,
            &gl_convert,
            &download,
            &convert,
            sink.upcast_ref(),
        ])
        .map_err(|error| {
            tr!(
                "Failed to assemble the GStreamer capture pipeline: {error}",
                error = error
            )
        })?;
    gst::Element::link_many([
        &source,
        &dma_buf_filter,
        &upload,
        &gl_convert,
        &download,
        &convert,
        sink.upcast_ref(),
    ])
    .map_err(|error| {
        tr!(
            "Failed to link the GStreamer capture pipeline: {error}",
            error = error
        )
    })?;

    if let Err(error) = pipeline.set_state(gst::State::Playing) {
        let detail = wait_for_bus_failure(&pipeline, 10).unwrap_or_else(|| {
            tr!(
                "GStreamer state change failed: {error}",
                error = format!("{error:?}")
            )
        });
        let _ = pipeline.set_state(gst::State::Null);
        return Err(tr!(
            "Failed to start the GStreamer capture pipeline: {detail}",
            detail = detail
        ));
    }
    let result = pull_first_frame(&pipeline, &sink);
    let stop_result = pipeline.set_state(gst::State::Null);
    if let Err(error) = stop_result
        && result.is_ok()
    {
        return Err(tr!(
            "Failed to stop the GStreamer capture pipeline: {error}",
            error = format!("{error:?}")
        ));
    }
    result
}

fn dma_buf_import_caps() -> Result<gst::Caps, String> {
    use gst::glib::{
        self,
        translate::{ToGlibPtr, ToGlibPtrMut},
        value::ToValue,
    };

    let display = gst_gl::GLDisplay::with_type(gst_gl::GLDisplayType::EGL_SURFACELESS)
        .ok_or_else(|| tr!("GStreamer could not create a surfaceless EGL display"))?;
    let context = gst_gl::GLContext::new(&display);
    context
        .create(None::<&gst_gl::GLContext>)
        .map_err(|error| {
            tr!(
                "Failed to create an EGL context for DMA-BUF negotiation: {error}",
                error = error
            )
        })?;

    let source_format = "RGBA".to_value();
    let mut drm_formats: glib::Value = unsafe { glib::translate::uninitialized() };
    let flags = gst_gl::ffi::GST_GL_DRM_FORMAT_DIRECT_IMPORT
        | gst_gl::ffi::GST_GL_DRM_FORMAT_INCLUDE_EXTERNAL;
    let transformed = unsafe {
        gst_gl::ffi::gst_gl_dma_buf_transform_gst_formats_to_drm_formats(
            context.to_glib_none().0,
            source_format.to_glib_none().0,
            flags,
            drm_formats.to_glib_none_mut().0,
        ) != glib::ffi::GFALSE
    };
    if !transformed {
        return Err(tr!(
            "The EGL driver reported no directly importable DMA-BUF formats"
        ));
    }

    let drm_formats = value_strings(&drm_formats)?;
    if drm_formats.is_empty() {
        return Err(tr!("The EGL driver returned an empty DMA-BUF format list"));
    }
    let drm_formats = gst::List::new(drm_formats);
    Ok(gst::Caps::builder("video/x-raw")
        .features(["memory:DMABuf"])
        .field("format", "DMA_DRM")
        .field("drm-format", drm_formats)
        .build())
}

fn value_strings(value: &gst::glib::Value) -> Result<Vec<String>, String> {
    if let Ok(value) = value.get::<String>() {
        return Ok(vec![value]);
    }
    let values = value
        .get::<gst::List>()
        .map_err(|_| tr!("The EGL driver returned an invalid DMA-BUF format list"))?;
    values
        .iter()
        .map(|value| {
            value
                .get::<String>()
                .map_err(|_| tr!("The EGL driver returned a non-string DMA-BUF format"))
        })
        .collect()
}

fn wait_for_bus_failure(pipeline: &gst::Pipeline, attempts: usize) -> Option<String> {
    let bus = pipeline.bus()?;
    for _ in 0..attempts {
        let Some(message) = bus.timed_pop_filtered(
            gst::ClockTime::from_mseconds(100),
            &[gst::MessageType::Error, gst::MessageType::Eos],
        ) else {
            continue;
        };
        return Some(match message.view() {
            gst::MessageView::Error(error) => tr!(
                "{component}: {error}{debug}",
                component = message
                    .src()
                    .map(|source| source.path_string().to_string())
                    .unwrap_or_else(|| tr!("Unknown component")),
                error = error.error(),
                debug = error
                    .debug()
                    .map(|debug| format!(" ({debug})"))
                    .unwrap_or_default()
            ),
            gst::MessageView::Eos(_) => tr!("The PipeWire stream ended during startup"),
            _ => unreachable!("the bus filter only returns error or EOS messages"),
        });
    }
    None
}

fn pull_first_frame(pipeline: &gst::Pipeline, sink: &gst_app::AppSink) -> Result<Vec<u8>, String> {
    let bus = pipeline
        .bus()
        .ok_or_else(|| tr!("The GStreamer capture pipeline has no message bus"))?;
    for _ in 0..120 {
        if let Some(sample) = sink.try_pull_sample(gst::ClockTime::from_mseconds(100)) {
            return encode_sample(&sample);
        }
        if let Some(message) = bus.pop_filtered(&[gst::MessageType::Error, gst::MessageType::Eos]) {
            return match message.view() {
                gst::MessageView::Error(error) => Err(tr!(
                    "GStreamer PipeWire stream error: {error}{debug}",
                    error = error.error(),
                    debug = error
                        .debug()
                        .map(|debug| format!(" ({debug})"))
                        .unwrap_or_default()
                )),
                gst::MessageView::Eos(_) => {
                    Err(tr!("The PipeWire stream ended before returning a frame"))
                }
                _ => unreachable!("the bus filter only returns error or EOS messages"),
            };
        }
    }
    Err(tr!("Timed out waiting for a PipeWire capture frame"))
}

fn encode_sample(sample: &gst::Sample) -> Result<Vec<u8>, String> {
    let caps = sample
        .caps()
        .ok_or_else(|| tr!("The PipeWire capture frame has no video format"))?;
    let info = gst_video::VideoInfo::from_caps(caps).map_err(|error| {
        tr!(
            "Failed to parse the PipeWire video format: {error}",
            error = error
        )
    })?;
    if info.format() != gst_video::VideoFormat::Rgba {
        return Err(tr!(
            "GStreamer returned the unexpected pixel format {format}",
            format = format!("{:?}", info.format())
        ));
    }
    let buffer = sample
        .buffer()
        .ok_or_else(|| tr!("The PipeWire capture sample has no pixel buffer"))?;
    let frame =
        gst_video::VideoFrameRef::from_buffer_ref_readable(buffer, &info).map_err(|error| {
            tr!(
                "Failed to map the PipeWire capture frame: {error}",
                error = error
            )
        })?;
    let stride = *frame
        .plane_stride()
        .first()
        .ok_or_else(|| tr!("The PipeWire capture frame has no color plane"))?;
    let pixels = frame.plane_data(0).map_err(|error| {
        tr!(
            "Failed to read PipeWire capture pixels: {error}",
            error = error
        )
    })?;
    encode_rgba_png(info.width(), info.height(), stride, pixels)
}

fn encode_rgba_png(width: u32, height: u32, stride: i32, pixels: &[u8]) -> Result<Vec<u8>, String> {
    let row_bytes = (width as usize)
        .checked_mul(4)
        .ok_or_else(|| tr!("Capture width overflow"))?;
    if stride < 0 || (stride as usize) < row_bytes {
        return Err(tr!("The PipeWire frame stride is smaller than an RGBA row"));
    }
    let stride = stride as usize;
    let required = stride
        .checked_mul(height.saturating_sub(1) as usize)
        .and_then(|offset| offset.checked_add(row_bytes))
        .ok_or_else(|| tr!("Capture dimensions overflow"))?;
    if pixels.len() < required {
        return Err(tr!("The PipeWire capture pixel buffer is incomplete"));
    }
    let mut rgba = Vec::with_capacity(
        row_bytes
            .checked_mul(height as usize)
            .ok_or_else(|| tr!("Capture dimensions overflow"))?,
    );
    for y in 0..height as usize {
        let start = y * stride;
        rgba.extend_from_slice(&pixels[start..start + row_bytes]);
    }
    let mut encoded = Cursor::new(Vec::new());
    PngEncoder::new(&mut encoded)
        .write_image(&rgba, width, height, ExtendedColorType::Rgba8)
        .map_err(|error| error.to_string())?;
    Ok(encoded.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba_rows_with_padding_are_encoded_without_padding() {
        let pixels = [
            1, 2, 3, 255, 4, 5, 6, 255, 0, 0, 0, 0, 7, 8, 9, 255, 10, 11, 12, 255, 0, 0, 0, 0,
        ];
        let png = encode_rgba_png(2, 2, 12, &pixels).expect("frame must encode");
        let decoded = image::load_from_memory_with_format(&png, image::ImageFormat::Png)
            .expect("png must decode")
            .into_rgba8();
        assert_eq!(decoded.get_pixel(0, 0).0, [1, 2, 3, 255]);
        assert_eq!(decoded.get_pixel(1, 1).0, [10, 11, 12, 255]);
    }

    #[test]
    fn invalid_stride_and_short_buffers_are_rejected() {
        assert!(encode_rgba_png(2, 2, 4, &[0; 16]).is_err());
        assert!(encode_rgba_png(2, 2, 8, &[0; 8]).is_err());
    }
}
