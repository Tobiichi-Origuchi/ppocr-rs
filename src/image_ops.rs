use image::RgbImage;

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
}
