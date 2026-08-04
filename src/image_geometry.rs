//! Image geometry used to compose the PaddleOCR modules.

use std::cmp::Ordering;

use image::{Rgb, RgbImage, imageops};

use crate::text_detection::minimum_area_rectangle_points;
use crate::{DetectedTextRegion, OcrError, Point, Result};

/// Sort detected quadrilaterals from top to bottom and left to right.
///
/// This follows PaddleX `SortQuadBoxes`, including its 10-pixel tolerance for
/// treating neighboring top-left points as belonging to the same text row.
pub fn sort_text_regions(regions: &[DetectedTextRegion]) -> Vec<DetectedTextRegion> {
    let mut sorted = regions.to_vec();
    sorted.sort_by(|left, right| {
        left.points[0]
            .y
            .total_cmp(&right.points[0].y)
            .then_with(|| left.points[0].x.total_cmp(&right.points[0].x))
    });
    for index in 0..sorted.len().saturating_sub(1) {
        for previous in (0..=index).rev() {
            let left = sorted[previous].points[0];
            let right = sorted[previous + 1].points[0];
            if (right.y - left.y).abs() < 10.0 && right.x < left.x {
                sorted.swap(previous, previous + 1);
            } else {
                break;
            }
        }
    }
    sorted
}

/// Apply PaddleX's document/text-line correction rotation.
///
/// The only valid angles are the orthogonal classifier labels. PaddleX uses
/// `getRotationMatrix2D((w/2, h/2), angle)` followed by `warpAffine`; the exact
/// integer mappings below preserve that convention, including its zero-filled
/// leading border for non-zero angles.
pub fn rotate_image(image: &RgbImage, angle: u16) -> Result<RgbImage> {
    if image.width() == 0 || image.height() == 0 {
        return Err(OcrError::EmptyImage);
    }
    match angle {
        0 => Ok(image.clone()),
        90 => Ok(RgbImage::from_fn(image.height(), image.width(), |x, y| {
            let source_x = image.width() as i64 - y as i64;
            let source_y = x as i64;
            pixel_or_black(image, source_x, source_y)
        })),
        180 => Ok(RgbImage::from_fn(image.width(), image.height(), |x, y| {
            let source_x = image.width() as i64 - x as i64;
            let source_y = image.height() as i64 - y as i64;
            pixel_or_black(image, source_x, source_y)
        })),
        270 => Ok(RgbImage::from_fn(image.height(), image.width(), |x, y| {
            let source_x = y as i64;
            let source_y = image.height() as i64 - x as i64;
            pixel_or_black(image, source_x, source_y)
        })),
        _ => Err(OcrError::InvalidGeometry {
            reason: format!("rotation angle must be one of 0, 90, 180, or 270; received {angle}"),
        }),
    }
}

/// Perspective-crop one PaddleOCR quadrilateral into a horizontal text image.
///
/// `points` must be ordered top-left, top-right, bottom-right, bottom-left.
/// PaddleX computes output dimensions from opposite-edge maxima, uses cubic
/// perspective sampling with replicated borders, and rotates tall crops 90°
/// counter-clockwise.
pub fn crop_text_region(image: &RgbImage, points: &[Point; 4]) -> Result<RgbImage> {
    if image.width() == 0 || image.height() == 0 {
        return Err(OcrError::EmptyImage);
    }
    if points
        .iter()
        .any(|point| !point.x.is_finite() || !point.y.is_finite())
    {
        return Err(OcrError::InvalidGeometry {
            reason: "text-region points must be finite".to_owned(),
        });
    }

    // PaddleX casts detector points to int32 and runs minAreaRect again before
    // ordering the resulting box TL/TR/BR/BL.
    let integer_points = points.map(|point| Point {
        x: point.x as i32 as f32,
        y: point.y as i32 as f32,
    });
    let points = minimum_area_rectangle_points(&integer_points).ok_or_else(|| {
        OcrError::InvalidGeometry {
            reason: "text-region points do not define a minimum-area rectangle".to_owned(),
        }
    })?;
    let width = edge_length(points[0], points[1])
        .max(edge_length(points[2], points[3]))
        .floor() as u32;
    let height = edge_length(points[0], points[3])
        .max(edge_length(points[1], points[2]))
        .floor() as u32;
    if width == 0 || height == 0 {
        return Err(OcrError::InvalidGeometry {
            reason: format!("text-region crop has zero dimensions ({width}x{height})"),
        });
    }

    let destination = [
        (0.0, 0.0),
        (width as f64, 0.0),
        (width as f64, height as f64),
        (0.0, height as f64),
    ];
    let source = points.map(|point| (point.x as f64, point.y as f64));
    let forward = homography(source, destination).ok_or_else(|| OcrError::InvalidGeometry {
        reason: "text-region points do not define an invertible perspective transform".to_owned(),
    })?;
    let inverse = invert_3x3(forward).ok_or_else(|| OcrError::InvalidGeometry {
        reason: "text-region perspective transform is singular".to_owned(),
    })?;

    let mut crop = RgbImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let denominator = inverse[6] * x as f64 + inverse[7] * y as f64 + inverse[8];
            if denominator.abs() <= f64::EPSILON {
                return Err(OcrError::InvalidGeometry {
                    reason: "text-region perspective transform mapped a pixel to infinity"
                        .to_owned(),
                });
            }
            let source_x =
                (inverse[0] * x as f64 + inverse[1] * y as f64 + inverse[2]) / denominator;
            let source_y =
                (inverse[3] * x as f64 + inverse[4] * y as f64 + inverse[5]) / denominator;
            crop.put_pixel(x, y, sample_cubic_replicate(image, source_x, source_y));
        }
    }

    if height as f64 / width as f64 >= 1.5 {
        Ok(imageops::rotate270(&crop))
    } else {
        Ok(crop)
    }
}

/// Crop multiple sorted/unsorted quadrilaterals without changing their order.
pub fn crop_text_regions(
    image: &RgbImage,
    regions: &[DetectedTextRegion],
) -> Result<Vec<RgbImage>> {
    regions
        .iter()
        .map(|region| crop_text_region(image, &region.points))
        .collect()
}

fn pixel_or_black(image: &RgbImage, x: i64, y: i64) -> Rgb<u8> {
    if x >= 0 && y >= 0 && x < image.width() as i64 && y < image.height() as i64 {
        *image.get_pixel(x as u32, y as u32)
    } else {
        Rgb([0, 0, 0])
    }
}

fn edge_length(left: Point, right: Point) -> f32 {
    (right.x - left.x).hypot(right.y - left.y)
}

fn homography(from: [(f64, f64); 4], to: [(f64, f64); 4]) -> Option<[f64; 9]> {
    let mut matrix = [[0.0_f64; 9]; 8];
    for index in 0..4 {
        let (source_x, source_y) = from[index];
        let (target_x, target_y) = to[index];
        matrix[index * 2] = [
            source_x,
            source_y,
            1.0,
            0.0,
            0.0,
            0.0,
            -target_x * source_x,
            -target_x * source_y,
            target_x,
        ];
        matrix[index * 2 + 1] = [
            0.0,
            0.0,
            0.0,
            source_x,
            source_y,
            1.0,
            -target_y * source_x,
            -target_y * source_y,
            target_y,
        ];
    }

    for pivot in 0..8 {
        let best = (pivot..8).max_by(|left, right| {
            matrix[*left][pivot]
                .abs()
                .partial_cmp(&matrix[*right][pivot].abs())
                .unwrap_or(Ordering::Equal)
        })?;
        if matrix[best][pivot].abs() <= 1e-12 {
            return None;
        }
        matrix.swap(pivot, best);
        let divisor = matrix[pivot][pivot];
        for column in pivot..=8 {
            matrix[pivot][column] /= divisor;
        }
        for row in 0..8 {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            for column in pivot..=8 {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
        }
    }
    Some([
        matrix[0][8],
        matrix[1][8],
        matrix[2][8],
        matrix[3][8],
        matrix[4][8],
        matrix[5][8],
        matrix[6][8],
        matrix[7][8],
        1.0,
    ])
}

fn invert_3x3(matrix: [f64; 9]) -> Option<[f64; 9]> {
    let determinant = matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
        - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
        + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
    if determinant.abs() <= 1e-12 {
        return None;
    }
    let inverse = 1.0 / determinant;
    Some([
        (matrix[4] * matrix[8] - matrix[5] * matrix[7]) * inverse,
        (matrix[2] * matrix[7] - matrix[1] * matrix[8]) * inverse,
        (matrix[1] * matrix[5] - matrix[2] * matrix[4]) * inverse,
        (matrix[5] * matrix[6] - matrix[3] * matrix[8]) * inverse,
        (matrix[0] * matrix[8] - matrix[2] * matrix[6]) * inverse,
        (matrix[2] * matrix[3] - matrix[0] * matrix[5]) * inverse,
        (matrix[3] * matrix[7] - matrix[4] * matrix[6]) * inverse,
        (matrix[1] * matrix[6] - matrix[0] * matrix[7]) * inverse,
        (matrix[0] * matrix[4] - matrix[1] * matrix[3]) * inverse,
    ])
}

fn sample_cubic_replicate(image: &RgbImage, x: f64, y: f64) -> Rgb<u8> {
    let base_x = x.floor() as i64;
    let base_y = y.floor() as i64;
    // OpenCV's interpolation lookup table quantizes each fractional coordinate
    // to 5 bits before applying its A=-0.75 cubic kernel.
    let fraction_x = quantize_fraction(x - base_x as f64);
    let fraction_y = quantize_fraction(y - base_y as f64);
    let x_weights = cubic_weights(fraction_x);
    let y_weights = cubic_weights(fraction_y);
    let mut channels = [0_u8; 3];
    for (channel, output) in channels.iter_mut().enumerate() {
        let mut value = 0.0_f64;
        for (row, y_weight) in y_weights.iter().copied().enumerate() {
            let source_y = (base_y + row as i64 - 1).clamp(0, image.height() as i64 - 1);
            let mut horizontal = 0.0;
            for (column, x_weight) in x_weights.iter().copied().enumerate() {
                let source_x = (base_x + column as i64 - 1).clamp(0, image.width() as i64 - 1);
                horizontal +=
                    image.get_pixel(source_x as u32, source_y as u32).0[channel] as f64 * x_weight;
            }
            value += horizontal * y_weight;
        }
        *output = value.round_ties_even().clamp(0.0, 255.0) as u8;
    }
    Rgb(channels)
}

fn quantize_fraction(fraction: f64) -> f64 {
    let index = (fraction * 32.0).round_ties_even().clamp(0.0, 31.0);
    index / 32.0
}

fn cubic_weights(value: f64) -> [f64; 4] {
    const A: f64 = -0.75;
    let first = ((A * (value + 1.0) - 5.0 * A) * (value + 1.0) + 8.0 * A) * (value + 1.0) - 4.0 * A;
    let second = ((A + 2.0) * value - (A + 3.0)) * value * value + 1.0;
    let inverse = 1.0 - value;
    let third = ((A + 2.0) * inverse - (A + 3.0)) * inverse * inverse + 1.0;
    let fourth = 1.0 - first - second - third;
    [first, second, third, fourth]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn region(x: f32, y: f32) -> DetectedTextRegion {
        DetectedTextRegion {
            points: [
                Point { x, y },
                Point { x: x + 8.0, y },
                Point {
                    x: x + 8.0,
                    y: y + 4.0,
                },
                Point { x, y: y + 4.0 },
            ],
            score: 1.0,
        }
    }

    fn scalar_image() -> RgbImage {
        RgbImage::from_fn(3, 2, |x, y| {
            let value = (y * 3 + x + 1) as u8;
            Rgb([value, value, value])
        })
    }

    fn values(image: &RgbImage) -> Vec<u8> {
        image.pixels().map(|pixel| pixel[0]).collect()
    }

    #[test]
    fn sorts_with_paddlex_row_tolerance() {
        let input = [region(40.0, 2.0), region(20.0, 15.0), region(5.0, 9.0)];
        let sorted = sort_text_regions(&input);
        let positions: Vec<(f32, f32)> = sorted
            .iter()
            .map(|item| (item.points[0].x, item.points[0].y))
            .collect();
        assert_eq!(positions, vec![(5.0, 9.0), (40.0, 2.0), (20.0, 15.0)]);
    }

    #[test]
    fn rotations_match_opencv_paddlex_reference() {
        let image = scalar_image();
        let angle_90 = rotate_image(&image, 90).unwrap();
        assert_eq!(angle_90.dimensions(), (2, 3));
        assert_eq!(values(&angle_90), vec![0, 0, 3, 6, 2, 5]);

        let angle_180 = rotate_image(&image, 180).unwrap();
        assert_eq!(angle_180.dimensions(), (3, 2));
        assert_eq!(values(&angle_180), vec![0, 0, 0, 0, 6, 5]);

        let angle_270 = rotate_image(&image, 270).unwrap();
        assert_eq!(angle_270.dimensions(), (2, 3));
        assert_eq!(values(&angle_270), vec![0, 4, 0, 5, 0, 6]);
    }

    #[test]
    fn axis_aligned_crop_has_paddlex_dimensions() {
        let image = RgbImage::from_fn(6, 4, |x, y| Rgb([x as u8, y as u8, 0]));
        let points = [
            Point { x: 1.0, y: 1.0 },
            Point { x: 5.0, y: 1.0 },
            Point { x: 5.0, y: 3.0 },
            Point { x: 1.0, y: 3.0 },
        ];
        let crop = crop_text_region(&image, &points).unwrap();
        assert_eq!(crop.dimensions(), (4, 2));
        assert_eq!(crop.get_pixel(0, 0), &Rgb([1, 1, 0]));
        assert_eq!(crop.get_pixel(3, 1), &Rgb([4, 2, 0]));
    }

    #[test]
    fn tall_crop_is_rotated_counter_clockwise() {
        let image = RgbImage::from_fn(4, 7, |x, y| Rgb([(y * 10 + x) as u8, 0, 0]));
        let points = [
            Point { x: 1.0, y: 1.0 },
            Point { x: 3.0, y: 1.0 },
            Point { x: 3.0, y: 6.0 },
            Point { x: 1.0, y: 6.0 },
        ];
        let crop = crop_text_region(&image, &points).unwrap();
        assert_eq!(crop.dimensions(), (5, 2));
        assert_eq!(crop.get_pixel(0, 0)[0], 12);
        assert_eq!(crop.get_pixel(4, 1)[0], 51);
    }

    #[test]
    fn perspective_crop_matches_opencv_reference_pixels() {
        let image = RgbImage::from_fn(72, 48, |x, y| {
            Rgb([
                ((x * 13 + y * 3) % 256) as u8,
                ((x * 2 + y * 17) % 256) as u8,
                ((x * 7 + y * 11) % 256) as u8,
            ])
        });
        let points = [
            Point { x: 8.8, y: 11.2 },
            Point { x: 60.4, y: 5.9 },
            Point { x: 63.1, y: 32.7 },
            Point { x: 11.3, y: 39.6 },
        ];
        let crop = crop_text_region(&image, &points).unwrap();
        assert_eq!(crop.dimensions(), (52, 28));

        // Golden samples generated with PaddleX's OpenCV implementation. The
        // pure-Rust interpolation is normally exact and differs by at most six
        // intensity levels at OpenCV's fixed-point rounding boundaries.
        let reference = [
            (0, 0, [134, 203, 176]),
            (51, 0, [14, 206, 214]),
            (0, 27, [26, 156, 238]),
            (51, 27, [136, 158, 20]),
            (7, 5, [242, 34, 23]),
            (24, 14, [255, 190, 223]),
            (40, 9, [171, 104, 164]),
            (26, 16, [24, 223, 68]),
        ];
        for (x, y, expected) in reference {
            for (actual, expected) in crop.get_pixel(x, y).0.into_iter().zip(expected) {
                assert!(actual.abs_diff(expected) <= 6);
            }
        }
    }

    #[test]
    fn rejects_invalid_geometry() {
        let image = scalar_image();
        assert!(rotate_image(&image, 45).is_err());
        let points = [Point { x: 1.0, y: 1.0 }; 4];
        assert!(crop_text_region(&image, &points).is_err());
    }
}
