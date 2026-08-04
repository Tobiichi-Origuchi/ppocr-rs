use image::{Rgb, RgbImage};

/// Resize with OpenCV-compatible half-pixel coordinates and emit BGR/CHW data.
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
        let (source_y0, source_y1, y_weight) =
            interpolation_axis(destination_y, destination_height, source_height as usize);
        for destination_x in 0..destination_width {
            let (source_x0, source_x1, x_weight) =
                interpolation_axis(destination_x, destination_width, source_width as usize);
            let destination_index = destination_y * destination_width + destination_x;

            for bgr_channel in 0..3 {
                let rgb_channel = 2 - bgr_channel;
                let top_left = source_channel(image, source_x0, source_y0, rgb_channel);
                let top_right = source_channel(image, source_x1, source_y0, rgb_channel);
                let bottom_left = source_channel(image, source_x0, source_y1, rgb_channel);
                let bottom_right = source_channel(image, source_x1, source_y1, rgb_channel);
                let top = top_left + (top_right - top_left) * x_weight;
                let bottom = bottom_left + (bottom_right - bottom_left) * x_weight;
                let value = top + (bottom - top) * y_weight;
                output[bgr_channel * plane + destination_index] =
                    channel_transform(bgr_channel, value);
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
            opencv_u8_resize_axis(destination_y, destination_height, image.height() as usize);
        for destination_x in 0..destination_width {
            let (source_x0, source_x1, x0_weight, x1_weight) =
                opencv_u8_resize_axis(destination_x, destination_width, image.width() as usize);
            let mut pixel = [0_u8; 3];
            for (channel, output_channel) in pixel.iter_mut().enumerate() {
                let top_left = source_channel(image, source_x0, source_y0, channel) as i32;
                let top_right = source_channel(image, source_x1, source_y0, channel) as i32;
                let bottom_left = source_channel(image, source_x0, source_y1, channel) as i32;
                let bottom_right = source_channel(image, source_x1, source_y1, channel) as i32;
                let top = top_left * x0_weight + top_right * x1_weight;
                let bottom = bottom_left * x0_weight + bottom_right * x1_weight;

                // OpenCV's legacy INTER_LINEAR u8 path uses 11-bit horizontal
                // coefficients, then its SIMD/scalar vertical fixed-point cast.
                let value =
                    (((y0_weight * (top >> 4)) >> 16) + ((y1_weight * (bottom >> 4)) >> 16) + 2)
                        >> 2;
                *output_channel = value.clamp(0, 255) as u8;
            }
            output.put_pixel(destination_x as u32, destination_y as u32, Rgb(pixel));
        }
    }
    output
}

fn opencv_u8_resize_axis(
    destination: usize,
    destination_size: usize,
    source_size: usize,
) -> (usize, usize, i32, i32) {
    if source_size <= 1 {
        return (0, 0, 2048, 0);
    }
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

fn interpolation_axis(
    destination: usize,
    destination_size: usize,
    source_size: usize,
) -> (usize, usize, f32) {
    if source_size <= 1 {
        return (0, 0, 0.0);
    }
    let coordinate =
        (destination as f32 + 0.5) * source_size as f32 / destination_size as f32 - 0.5;
    if coordinate <= 0.0 {
        return (0, 0, 0.0);
    }
    let lower = coordinate.floor() as usize;
    if lower >= source_size - 1 {
        return (source_size - 1, source_size - 1, 0.0);
    }
    (lower, lower + 1, coordinate - lower as f32)
}

fn source_channel(image: &RgbImage, x: usize, y: usize, channel: usize) -> f32 {
    if x < image.width() as usize && y < image.height() as usize {
        image.get_pixel(x as u32, y as u32).0[channel] as f32
    } else {
        // PaddleX pads very small inputs at the bottom and right with black.
        0.0
    }
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

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
    }
}
