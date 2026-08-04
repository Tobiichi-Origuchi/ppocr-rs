use image::{Rgb, RgbImage};

/// Resize through OpenCV's uint8 `INTER_LINEAR` path and emit BGR/CHW data.
pub(crate) fn resize_rgb_to_bgr_chw(
    image: &RgbImage,
    source_width: u32,
    source_height: u32,
    destination_width: usize,
    destination_height: usize,
    channel_transform: impl Fn(usize, f32) -> f32,
) -> Vec<f32> {
    let plane = destination_width * destination_height;
    let mut output = vec![0.0; plane * 3];
    for destination_y in 0..destination_height {
        let y_axis =
            opencv_u8_resize_y_axis(destination_y, destination_height, source_height as usize);
        for destination_x in 0..destination_width {
            let x_axis =
                opencv_u8_resize_x_axis(destination_x, destination_width, source_width as usize);
            let destination_index = destination_y * destination_width + destination_x;

            for bgr_channel in 0..3 {
                let rgb_channel = 2 - bgr_channel;
                let value = opencv_u8_resized_channel(image, x_axis, y_axis, rgb_channel);
                output[bgr_channel * plane + destination_index] =
                    channel_transform(bgr_channel, f32::from(value));
            }
        }
    }
    output
}

/// Resize an RGB image with OpenCV-style coordinates and quantize back to u8,
/// matching the classification pipeline's `cv2.resize` boundary.
pub(crate) fn resize_rgb(
    image: &RgbImage,
    destination_width: usize,
    destination_height: usize,
) -> RgbImage {
    let mut output = RgbImage::new(destination_width as u32, destination_height as u32);
    for destination_y in 0..destination_height {
        let (source_y0, source_y1, y0_weight, y1_weight) =
            opencv_u8_resize_y_axis(destination_y, destination_height, image.height() as usize);
        for destination_x in 0..destination_width {
            let (source_x0, source_x1, x0_weight, x1_weight) =
                opencv_u8_resize_x_axis(destination_x, destination_width, image.width() as usize);
            let mut pixel = [0_u8; 3];
            for (channel, output_channel) in pixel.iter_mut().enumerate() {
                *output_channel = opencv_u8_resized_channel(
                    image,
                    (source_x0, source_x1, x0_weight, x1_weight),
                    (source_y0, source_y1, y0_weight, y1_weight),
                    channel,
                );
            }
            output.put_pixel(destination_x as u32, destination_y as u32, Rgb(pixel));
        }
    }
    output
}

fn opencv_u8_resize_x_axis(
    destination: usize,
    destination_size: usize,
    source_size: usize,
) -> (usize, usize, i32, i32) {
    let coordinate =
        ((destination as f64 + 0.5) * source_size as f64 / destination_size as f64 - 0.5) as f32;
    let mut lower = coordinate.floor() as isize;
    let mut fraction = coordinate - lower as f32;
    if lower < 0 {
        lower = 0;
        fraction = 0.0;
    } else if lower >= source_size as isize - 1 {
        lower = source_size as isize - 1;
        fraction = 0.0;
    }
    let lower = lower as usize;
    let upper = (lower + 1).min(source_size - 1);
    let lower_weight = ((1.0 - fraction) * 2048.0).round_ties_even() as i32;
    let upper_weight = (fraction * 2048.0).round_ties_even() as i32;
    (lower, upper, lower_weight, upper_weight)
}

fn opencv_u8_resize_y_axis(
    destination: usize,
    destination_size: usize,
    source_size: usize,
) -> (usize, usize, i32, i32) {
    let coordinate =
        ((destination as f64 + 0.5) * source_size as f64 / destination_size as f64 - 0.5) as f32;
    let lower = coordinate.floor() as isize;
    let fraction = coordinate - lower as f32;
    let maximum = source_size.saturating_sub(1) as isize;
    let lower_index = lower.clamp(0, maximum) as usize;
    let upper_index = (lower + 1).clamp(0, maximum) as usize;
    let lower_weight = ((1.0 - fraction) * 2048.0).round_ties_even() as i32;
    let upper_weight = (fraction * 2048.0).round_ties_even() as i32;
    (lower_index, upper_index, lower_weight, upper_weight)
}

pub(crate) fn rgb_region_to_chw(
    image: &RgbImage,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    channel_transform: impl Fn(usize, f32) -> f32,
) -> Vec<f32> {
    let plane = width * height;
    let mut output = vec![0.0; plane * 3];
    for row in 0..height {
        for column in 0..width {
            let pixel = image.get_pixel((x + column) as u32, (y + row) as u32);
            let destination = row * width + column;
            for channel in 0..3 {
                output[channel * plane + destination] =
                    channel_transform(channel, pixel.0[channel] as f32);
            }
        }
    }
    output
}

fn source_channel(image: &RgbImage, x: usize, y: usize, channel: usize) -> i32 {
    if x < image.width() as usize && y < image.height() as usize {
        i32::from(image.get_pixel(x as u32, y as u32).0[channel])
    } else {
        // PaddleX pads very small inputs at the bottom and right with black.
        0
    }
}

fn opencv_u8_resized_channel(
    image: &RgbImage,
    x_axis: (usize, usize, i32, i32),
    y_axis: (usize, usize, i32, i32),
    channel: usize,
) -> u8 {
    let (source_x0, source_x1, x0_weight, x1_weight) = x_axis;
    let (source_y0, source_y1, y0_weight, y1_weight) = y_axis;
    let top = source_channel(image, source_x0, source_y0, channel) * x0_weight
        + source_channel(image, source_x1, source_y0, channel) * x1_weight;
    let bottom = source_channel(image, source_x0, source_y1, channel) * x0_weight
        + source_channel(image, source_x1, source_y1, channel) * x1_weight;

    // OpenCV's legacy INTER_LINEAR u8 path uses 11-bit horizontal
    // coefficients, then its SIMD/scalar vertical fixed-point cast.
    let value = (((y0_weight * (top >> 4)) >> 16) + ((y1_weight * (bottom >> 4)) >> 16) + 2) >> 2;
    value.clamp(0, 255) as u8
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};
    use sha2::{Digest, Sha256};

    use super::*;

    #[test]
    fn reverses_rgb_to_bgr() {
        let image = RgbImage::from_pixel(1, 1, Rgb([10, 20, 30]));
        let result = resize_rgb_to_bgr_chw(&image, 1, 1, 1, 1, |_, value| value);
        assert_eq!(result, vec![30.0, 20.0, 10.0]);
    }

    #[test]
    fn virtual_padding_is_black() {
        let image = RgbImage::from_pixel(1, 1, Rgb([255, 255, 255]));
        let result = resize_rgb_to_bgr_chw(&image, 2, 2, 2, 2, |_, value| value);
        assert_eq!(&result[0..4], &[255.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn rgb_region_keeps_channel_order() {
        let image = RgbImage::from_pixel(1, 1, Rgb([10, 20, 30]));
        let result = rgb_region_to_chw(&image, 0, 0, 1, 1, |_, value| value);
        assert_eq!(result, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn u8_resize_matches_opencv_linear_reference() {
        let image = RgbImage::from_fn(2, 2, |x, y| match (x, y) {
            (0, 0) => Rgb([0, 10, 20]),
            (1, 0) => Rgb([100, 110, 120]),
            (0, 1) => Rgb([200, 210, 220]),
            (1, 1) => Rgb([250, 240, 230]),
            _ => unreachable!(),
        });
        let resized = resize_rgb(&image, 3, 3);
        let expected = [
            [0, 10, 20],
            [50, 60, 70],
            [100, 110, 120],
            [100, 110, 120],
            [138, 143, 148],
            [175, 175, 175],
            [200, 210, 220],
            [225, 225, 225],
            [250, 240, 230],
        ];
        assert_eq!(
            resized.pixels().map(|pixel| pixel.0).collect::<Vec<_>>(),
            expected
        );

        let chw = resize_rgb_to_bgr_chw(&image, 2, 2, 3, 3, |_, value| value);
        for channel in 0..3 {
            let rgb_channel = 2 - channel;
            let expected_plane: Vec<f32> = expected
                .iter()
                .map(|pixel| f32::from(pixel[rgb_channel]))
                .collect();
            assert_eq!(&chw[channel * 9..(channel + 1) * 9], expected_plane);
        }
    }

    #[test]
    fn u8_resize_matches_opencv_linear_corpus() {
        let cases = [
            (
                (17, 13),
                (31, 29),
                "12f36549dab8c7b5478c04acc0c161a39cd3a9248333fd57e0e8548f305c0680",
            ),
            (
                (31, 29),
                (17, 13),
                "4dd0bbcb81932bd75ea07893aa28fa174951a195747b84db06bf2cae0fe86c5e",
            ),
            (
                (257, 193),
                (960, 736),
                "fccfa91d57e396d514823f5dd5809070b2293db4d6178b60f2c7f3ff2005d874",
            ),
            (
                (514, 64),
                (385, 48),
                "37b91284f61f7dd4b3d85fa9e5b41ddbe6665a0abedf9159eef42249382c6cab",
            ),
            (
                (1001, 37),
                (1298, 48),
                "88634756cae6cc7e146647addb630861bce4c3ae99d89d77a823451da47eae4e",
            ),
        ];

        for ((source_width, source_height), (width, height), expected) in cases {
            let image = RgbImage::from_fn(source_width, source_height, |x, y| {
                Rgb([
                    ((x * 13 + y * 3) % 256) as u8,
                    ((x * 2 + y * 17) % 256) as u8,
                    ((x * 7 + y * 11) % 256) as u8,
                ])
            });
            let resized = resize_rgb(&image, width as usize, height as usize);
            let digest = Sha256::digest(resized.as_raw())
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            assert_eq!(digest, expected);
        }
    }
}
