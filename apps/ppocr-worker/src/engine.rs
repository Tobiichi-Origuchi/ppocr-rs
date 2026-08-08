use std::{io::Cursor, time::Instant};

use image::{DynamicImage, ImageFormat, RgbImage};
use ort::ep::ExecutionProvider;
use ppocr::{
    DetectionSizeLimit as CoreDetectionSizeLimit, GraphOptimizationLevel, ModelDownloader,
    ModelDownloaderOptions, OcrPipeline, OcrPipelineOptions, OrtSessionOptions, PipelineBuildPhase,
    TextDetectorOptions,
};
use ppocr_app_protocol::{
    ClassificationResult, DetectedRegion, DetectionSizeLimit, EffectiveDetectionParameters,
    EncodedImageFormat, ErrorCode, ImageDescriptor, ImageSize, ModelIds, ModelSettings,
    OcrMetadata, PipelineConfig, Point, ProcessedImagePolicy, ProviderCapability, ProviderConfig,
    ProviderKind, Recognize, TextLine, Word,
};

use crate::server::{Engine, EngineError, EngineOutput, EngineProgress, EngineProgressCallback};

#[derive(Default)]
pub struct OcrEngine {
    pipeline: Option<OcrPipeline>,
}

impl Engine for OcrEngine {
    fn capabilities(&self) -> Vec<ProviderCapability> {
        provider_capabilities()
    }

    fn configure(
        &mut self,
        config: &PipelineConfig,
        progress: EngineProgressCallback,
    ) -> Result<ProviderKind, EngineError> {
        validate_pipeline_config(config)?;
        let provider = provider_kind(&config.provider);
        let downloader = ModelDownloader::new(ModelDownloaderOptions::default())
            .map_err(|error| EngineError::new(ErrorCode::ModelDownload, error.to_string(), true))?;
        let mut builder = OcrPipeline::builder(&downloader)
            .text_detection_model(config.text_detection_model.clone())
            .text_recognition_model(config.text_recognition_model.clone())
            .session_options(session_options(config)?)
            .text_detection_options(text_detector_options(config, session_options(config)?))
            .pipeline_options(OcrPipelineOptions {
                textline_orientation_batch_size: config.textline_orientation_batch_size,
                text_recognition_batch_size: config.text_recognition_batch_size,
                text_recognition_score_threshold: config.text_recognition_score_threshold,
                return_word_boxes: config.return_word_boxes,
            });
        builder = match &config.document_orientation_model {
            Some(model) => builder.document_orientation_model(model.clone()),
            None => builder.disable_document_orientation(),
        };
        builder = match &config.document_rectification_model {
            Some(model) => builder.document_rectification_model(model.clone()),
            None => builder.disable_document_rectification(),
        };
        builder = match &config.textline_orientation_model {
            Some(model) => builder.textline_orientation_model(model.clone()),
            None => builder.disable_textline_orientation(),
        };
        let pipeline_progress = std::sync::Arc::new(move |event: ppocr::PipelineBuildProgress| {
            progress(EngineProgress {
                phase: match event.phase {
                    PipelineBuildPhase::Downloading => {
                        ppocr_app_protocol::ProgressPhase::PreparingModels
                    }
                    PipelineBuildPhase::Initializing => {
                        ppocr_app_protocol::ProgressPhase::InitializingSessions
                    }
                },
                model_index: event.model_index.try_into().unwrap_or(u32::MAX),
                model_count: event.model_count.try_into().unwrap_or(u32::MAX),
                model_id: event.model_id,
                bytes_completed: event.bytes_completed,
                total_bytes: event.total_bytes,
            });
        });
        self.pipeline = Some(
            builder
                .build_with_progress(pipeline_progress)
                .map_err(map_config_error)?,
        );
        Ok(provider)
    }

    fn recognize(
        &mut self,
        request: &Recognize,
        attachment: &[u8],
    ) -> Result<EngineOutput, EngineError> {
        let started = Instant::now();
        let image = decode_image(&request.image, attachment)?;
        let pipeline = self.pipeline.as_mut().ok_or_else(|| {
            EngineError::new(
                ErrorCode::InvalidState,
                "OCR pipeline is not configured",
                false,
            )
        })?;
        let result = pipeline
            .predict(&image)
            .map_err(|error| EngineError::new(ErrorCode::Inference, error.to_string(), true))?;
        let attach_image = match request.processed_image {
            ProcessedImagePolicy::Never => false,
            ProcessedImagePolicy::Always => true,
            ProcessedImagePolicy::WhenTransformed => {
                result.model_settings.use_document_orientation
                    || result.model_settings.use_document_rectification
            }
        };
        let metadata = metadata_from_result(&result);
        let processed_png = if attach_image {
            encode_png(result.processed_image)?
        } else {
            Vec::new()
        };
        Ok(EngineOutput {
            elapsed_ms: started.elapsed().as_millis().try_into().unwrap_or(u64::MAX),
            metadata,
            processed_png,
        })
    }
}

fn validate_pipeline_config(config: &PipelineConfig) -> Result<(), EngineError> {
    if !config.text_recognition_score_threshold.is_finite() {
        return Err(EngineError::new(
            ErrorCode::PipelineConfiguration,
            "recognition score threshold must be finite",
            false,
        ));
    }
    if matches!(config.provider, ProviderConfig::DirectMl { .. })
        && config.runtime.parallel_execution
    {
        return Err(EngineError::new(
            ErrorCode::PipelineConfiguration,
            "DirectML does not support parallel graph execution",
            false,
        ));
    }
    Ok(())
}

fn session_options(config: &PipelineConfig) -> Result<OrtSessionOptions, EngineError> {
    let execution_providers = match config.provider {
        ProviderConfig::Cpu => Vec::new(),
        ProviderConfig::Cuda { device_id } => vec![cuda_dispatch(device_id)?],
        ProviderConfig::DirectMl { device_id } => vec![directml_dispatch(device_id)?],
        ProviderConfig::CoreMl => vec![coreml_dispatch()?],
        ProviderConfig::WebGpu { device_id } => vec![webgpu_dispatch(device_id)?],
        ProviderConfig::MiGraphX { device_id } => vec![migraphx_dispatch(device_id)?],
    };
    let optimization_level = match config.runtime.optimization_level {
        ppocr_app_protocol::OptimizationLevel::Disable => GraphOptimizationLevel::Disable,
        ppocr_app_protocol::OptimizationLevel::Level1 => GraphOptimizationLevel::Level1,
        ppocr_app_protocol::OptimizationLevel::Level2 => GraphOptimizationLevel::Level2,
        ppocr_app_protocol::OptimizationLevel::All => GraphOptimizationLevel::All,
    };
    Ok(OrtSessionOptions {
        execution_providers,
        intra_threads: config.runtime.intra_threads,
        inter_threads: config.runtime.inter_threads,
        parallel_execution: config.runtime.parallel_execution,
        optimization_level,
    })
}

fn text_detector_options(
    config: &PipelineConfig,
    session: OrtSessionOptions,
) -> TextDetectorOptions {
    TextDetectorOptions {
        session,
        limit_side_len: config.detection.limit_side_len,
        limit_type: config.detection.limit_type.map(core_detection_limit),
        max_side_limit: config.detection.max_side_limit,
        threshold: config.detection.threshold,
        box_threshold: config.detection.box_threshold,
        unclip_ratio: config.detection.unclip_ratio,
    }
}

fn core_detection_limit(limit: DetectionSizeLimit) -> CoreDetectionSizeLimit {
    match limit {
        DetectionSizeLimit::Max => CoreDetectionSizeLimit::Max,
        DetectionSizeLimit::Min => CoreDetectionSizeLimit::Min,
        DetectionSizeLimit::ResizeLong => CoreDetectionSizeLimit::ResizeLong,
    }
}

fn provider_kind(config: &ProviderConfig) -> ProviderKind {
    match config {
        ProviderConfig::Cpu => ProviderKind::Cpu,
        ProviderConfig::Cuda { .. } => ProviderKind::Cuda,
        ProviderConfig::DirectMl { .. } => ProviderKind::DirectMl,
        ProviderConfig::CoreMl => ProviderKind::CoreMl,
        ProviderConfig::WebGpu { .. } => ProviderKind::WebGpu,
        ProviderConfig::MiGraphX { .. } => ProviderKind::MiGraphX,
    }
}

fn unsupported(provider: &str) -> EngineError {
    EngineError::new(
        ErrorCode::UnsupportedProvider,
        format!("worker was not compiled with {provider} support"),
        false,
    )
}

#[cfg(feature = "cuda")]
fn cuda_dispatch(device_id: i32) -> Result<ppocr::ExecutionProviderDispatch, EngineError> {
    Ok(ppocr::execution_providers::CUDA::default()
        .with_device_id(device_id)
        .build()
        .error_on_failure())
}

#[cfg(not(feature = "cuda"))]
fn cuda_dispatch(_: i32) -> Result<ppocr::ExecutionProviderDispatch, EngineError> {
    Err(unsupported("CUDA"))
}

#[cfg(feature = "directml")]
fn directml_dispatch(device_id: i32) -> Result<ppocr::ExecutionProviderDispatch, EngineError> {
    Ok(ppocr::execution_providers::DirectML::default()
        .with_device_id(device_id)
        .build()
        .error_on_failure())
}

#[cfg(not(feature = "directml"))]
fn directml_dispatch(_: i32) -> Result<ppocr::ExecutionProviderDispatch, EngineError> {
    Err(unsupported("DirectML"))
}

#[cfg(feature = "coreml")]
fn coreml_dispatch() -> Result<ppocr::ExecutionProviderDispatch, EngineError> {
    Ok(ppocr::execution_providers::CoreML::default()
        .build()
        .error_on_failure())
}

#[cfg(not(feature = "coreml"))]
fn coreml_dispatch() -> Result<ppocr::ExecutionProviderDispatch, EngineError> {
    Err(unsupported("CoreML"))
}

#[cfg(feature = "webgpu")]
fn webgpu_dispatch(device_id: i32) -> Result<ppocr::ExecutionProviderDispatch, EngineError> {
    Ok(ppocr::execution_providers::WebGPU::default()
        .with_device_id(device_id)
        .build()
        .error_on_failure())
}

#[cfg(not(feature = "webgpu"))]
fn webgpu_dispatch(_: i32) -> Result<ppocr::ExecutionProviderDispatch, EngineError> {
    Err(unsupported("WebGPU"))
}

#[cfg(feature = "migraphx")]
fn migraphx_dispatch(device_id: i32) -> Result<ppocr::ExecutionProviderDispatch, EngineError> {
    Ok(ppocr::execution_providers::MIGraphX::default()
        .with_device_id(device_id)
        .build()
        .error_on_failure())
}

#[cfg(not(feature = "migraphx"))]
fn migraphx_dispatch(_: i32) -> Result<ppocr::ExecutionProviderDispatch, EngineError> {
    Err(unsupported("MIGraphX"))
}

fn decode_image(descriptor: &ImageDescriptor, bytes: &[u8]) -> Result<RgbImage, EngineError> {
    let image = match descriptor {
        ImageDescriptor::Encoded { format } => {
            let format = image_format(*format);
            let dimensions = image::ImageReader::with_format(Cursor::new(bytes), format)
                .into_dimensions()
                .map_err(image_decode_error)?;
            validate_dimensions(dimensions.0, dimensions.1)?;
            image::load_from_memory_with_format(bytes, format).map_err(image_decode_error)?
        }
        ImageDescriptor::RawRgba8 {
            width,
            height,
            stride,
        } => {
            validate_dimensions(*width, *height)?;
            let row_bytes = width.checked_mul(4).ok_or_else(invalid_raw_image)?;
            if *stride < row_bytes {
                return Err(invalid_raw_image());
            }
            let expected = usize::try_from(*stride)
                .ok()
                .and_then(|stride| stride.checked_mul(*height as usize))
                .ok_or_else(invalid_raw_image)?;
            if bytes.len() != expected {
                return Err(invalid_raw_image());
            }
            let mut rgb = RgbImage::new(*width, *height);
            for y in 0..*height as usize {
                let row = &bytes[y * *stride as usize..][..row_bytes as usize];
                for (x, pixel) in row.chunks_exact(4).enumerate() {
                    rgb.put_pixel(
                        x as u32,
                        y as u32,
                        image::Rgb([pixel[0], pixel[1], pixel[2]]),
                    );
                }
            }
            return Ok(rgb);
        }
    };
    Ok(image.to_rgb8())
}

fn validate_dimensions(width: u32, height: u32) -> Result<(), EngineError> {
    let pixels = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or_else(invalid_raw_image)?;
    if width == 0 || height == 0 || width > 32_768 || height > 32_768 || pixels > 64 * 1024 * 1024 {
        return Err(EngineError::new(
            ErrorCode::ImageDecode,
            "image dimensions exceed the worker safety limits",
            false,
        ));
    }
    Ok(())
}

fn image_format(format: EncodedImageFormat) -> ImageFormat {
    match format {
        EncodedImageFormat::Png => ImageFormat::Png,
        EncodedImageFormat::Jpeg => ImageFormat::Jpeg,
        EncodedImageFormat::WebP => ImageFormat::WebP,
        EncodedImageFormat::Tiff => ImageFormat::Tiff,
        EncodedImageFormat::Bmp => ImageFormat::Bmp,
        EncodedImageFormat::Gif => ImageFormat::Gif,
    }
}

fn invalid_raw_image() -> EngineError {
    EngineError::new(
        ErrorCode::ImageDecode,
        "raw RGBA image metadata does not match its attachment",
        false,
    )
}

fn image_decode_error(error: image::ImageError) -> EngineError {
    EngineError::new(ErrorCode::ImageDecode, error.to_string(), false)
}

fn encode_png(image: RgbImage) -> Result<Vec<u8>, EngineError> {
    let mut output = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(image)
        .write_to(&mut output, ImageFormat::Png)
        .map_err(|error| EngineError::new(ErrorCode::Internal, error.to_string(), true))?;
    Ok(output.into_inner())
}

fn metadata_from_result(result: &ppocr::OcrPipelineResult) -> OcrMetadata {
    OcrMetadata {
        input_size: ImageSize {
            width: result.input_size.width,
            height: result.input_size.height,
        },
        processed_size: ImageSize {
            width: result.processed_size.width,
            height: result.processed_size.height,
        },
        model_settings: ModelSettings {
            use_document_orientation: result.model_settings.use_document_orientation,
            use_document_rectification: result.model_settings.use_document_rectification,
            use_textline_orientation: result.model_settings.use_textline_orientation,
        },
        model_ids: ModelIds {
            document_orientation: result.model_ids.document_orientation.clone(),
            document_rectification: result.model_ids.document_rectification.clone(),
            text_detection: result.model_ids.text_detection.clone(),
            textline_orientation: result.model_ids.textline_orientation.clone(),
            text_recognition: result.model_ids.text_recognition.clone(),
        },
        text_detection_parameters: EffectiveDetectionParameters {
            limit_side_len: result.text_detection_parameters.limit_side_len,
            limit_type: wire_detection_limit(result.text_detection_parameters.limit_type),
            max_side_limit: result.text_detection_parameters.max_side_limit,
            threshold: result.text_detection_parameters.threshold,
            box_threshold: result.text_detection_parameters.box_threshold,
            unclip_ratio: result.text_detection_parameters.unclip_ratio,
        },
        text_recognition_score_threshold: result.text_recognition_score_threshold,
        return_word_boxes: result.return_word_boxes,
        document_orientation: result.document_orientation.as_ref().map(|value| {
            ClassificationResult {
                class_id: value.class_id,
                angle: value.angle,
                label: value.label.clone(),
                score: value.score,
            }
        }),
        detections: result
            .detections
            .iter()
            .map(|region| DetectedRegion {
                points: region.points.map(wire_point),
                score: region.score,
            })
            .collect(),
        textline_orientations: result
            .textline_orientations
            .iter()
            .map(|orientation| {
                orientation.as_ref().map(|value| ClassificationResult {
                    class_id: value.class_id,
                    angle: value.angle,
                    label: value.label.clone(),
                    score: value.score,
                })
            })
            .collect(),
        text_lines: result
            .text_lines
            .iter()
            .map(|line| TextLine {
                detection_index: line.detection_index,
                polygon: line.polygon.map(wire_point),
                bounding_box: line.bounding_box,
                detection_score: line.detection_score,
                textline_orientation: line.textline_orientation.as_ref().map(|value| {
                    ClassificationResult {
                        class_id: value.class_id,
                        angle: value.angle,
                        label: value.label.clone(),
                        score: value.score,
                    }
                }),
                text: line.text.clone(),
                recognition_score: line.recognition_score,
                words: line.words.as_ref().map(|words| {
                    words
                        .iter()
                        .map(|word| Word {
                            text: word.text.clone(),
                            polygon: word.polygon.map(wire_point),
                            bounding_box: word.bounding_box,
                        })
                        .collect()
                }),
            })
            .collect(),
    }
}

fn wire_point(point: ppocr::Point) -> Point {
    Point {
        x: point.x,
        y: point.y,
    }
}

fn wire_detection_limit(limit: CoreDetectionSizeLimit) -> DetectionSizeLimit {
    match limit {
        CoreDetectionSizeLimit::Max => DetectionSizeLimit::Max,
        CoreDetectionSizeLimit::Min => DetectionSizeLimit::Min,
        CoreDetectionSizeLimit::ResizeLong => DetectionSizeLimit::ResizeLong,
    }
}

fn map_config_error(error: ppocr::OcrError) -> EngineError {
    let code = if matches!(error, ppocr::OcrError::Download(_)) {
        ErrorCode::ModelDownload
    } else {
        ErrorCode::PipelineConfiguration
    };
    EngineError::new(code, error.to_string(), true)
}

fn provider_capabilities() -> Vec<ProviderCapability> {
    vec![
        capability(
            ProviderKind::Cpu,
            true,
            ppocr::execution_providers::CPU::default().is_available(),
        ),
        cuda_capability(),
        directml_capability(),
        coreml_capability(),
        webgpu_capability(),
        migraphx_capability(),
    ]
}

fn capability(
    provider: ProviderKind,
    compiled: bool,
    availability: Result<bool, ort::Error>,
) -> ProviderCapability {
    match availability {
        Ok(available) => ProviderCapability {
            provider,
            compiled,
            ort_reported_available: available,
            detail: None,
        },
        Err(error) => ProviderCapability {
            provider,
            compiled,
            ort_reported_available: false,
            detail: Some(error.to_string()),
        },
    }
}

fn unavailable_capability(provider: ProviderKind, name: &str) -> ProviderCapability {
    ProviderCapability {
        provider,
        compiled: false,
        ort_reported_available: false,
        detail: Some(format!("worker was not compiled with {name} support")),
    }
}

#[cfg(feature = "cuda")]
fn cuda_capability() -> ProviderCapability {
    capability(
        ProviderKind::Cuda,
        true,
        ppocr::execution_providers::CUDA::default().is_available(),
    )
}

#[cfg(not(feature = "cuda"))]
fn cuda_capability() -> ProviderCapability {
    unavailable_capability(ProviderKind::Cuda, "CUDA")
}

#[cfg(feature = "directml")]
fn directml_capability() -> ProviderCapability {
    capability(
        ProviderKind::DirectMl,
        true,
        ppocr::execution_providers::DirectML::default().is_available(),
    )
}

#[cfg(not(feature = "directml"))]
fn directml_capability() -> ProviderCapability {
    unavailable_capability(ProviderKind::DirectMl, "DirectML")
}

#[cfg(feature = "coreml")]
fn coreml_capability() -> ProviderCapability {
    capability(
        ProviderKind::CoreMl,
        true,
        ppocr::execution_providers::CoreML::default().is_available(),
    )
}

#[cfg(not(feature = "coreml"))]
fn coreml_capability() -> ProviderCapability {
    unavailable_capability(ProviderKind::CoreMl, "CoreML")
}

#[cfg(feature = "webgpu")]
fn webgpu_capability() -> ProviderCapability {
    capability(
        ProviderKind::WebGpu,
        true,
        ppocr::execution_providers::WebGPU::default().is_available(),
    )
}

#[cfg(not(feature = "webgpu"))]
fn webgpu_capability() -> ProviderCapability {
    unavailable_capability(ProviderKind::WebGpu, "WebGPU")
}

#[cfg(feature = "migraphx")]
fn migraphx_capability() -> ProviderCapability {
    capability(
        ProviderKind::MiGraphX,
        true,
        ppocr::execution_providers::MIGraphX::default().is_available(),
    )
}

#[cfg(not(feature = "migraphx"))]
fn migraphx_capability() -> ProviderCapability {
    unavailable_capability(ProviderKind::MiGraphX, "MIGraphX")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_rgba_validation_and_conversion_are_exact() {
        let image = decode_image(
            &ImageDescriptor::RawRgba8 {
                width: 2,
                height: 1,
                stride: 8,
            },
            &[1, 2, 3, 255, 4, 5, 6, 128],
        )
        .unwrap();
        assert_eq!(image.get_pixel(0, 0).0, [1, 2, 3]);
        assert_eq!(image.get_pixel(1, 0).0, [4, 5, 6]);
        assert!(
            decode_image(
                &ImageDescriptor::RawRgba8 {
                    width: 2,
                    height: 1,
                    stride: 7,
                },
                &[0; 7]
            )
            .is_err()
        );
    }

    #[test]
    fn non_compiled_provider_is_rejected_without_fallback() {
        #[cfg(not(feature = "directml"))]
        assert!(directml_dispatch(0).is_err());
    }

    #[test]
    fn pipeline_defaults_enable_the_complete_pipeline() {
        let config = PipelineConfig::default();
        assert!(config.document_orientation_model.is_some());
        assert!(config.document_rectification_model.is_some());
        assert!(config.textline_orientation_model.is_some());
        assert!(config.return_word_boxes);
    }
}
