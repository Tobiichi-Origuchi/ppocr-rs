use clipper_sys::{
    EndType_etClosedPolygon, JoinType_jtRound, Path as ClipperPath, PolyType_ptSubject,
    Polygon as ClipperPolygon, Polygons as ClipperPolygons, free_polygons, offset,
};
use image::{GrayImage, Luma};
use imageproc::{contours::find_contours, point::Point as ImagePoint};
use ndarray::ArrayView2;

use super::{DetectedTextRegion, TextDetectionResult};
use crate::text_detection::geometry::{Point, minimum_area_rect};

#[derive(Debug, Clone)]
pub(super) struct DbOptions {
    pub threshold: f32,
    pub box_threshold: f32,
    pub max_candidates: usize,
    pub unclip_ratio: f32,
    pub use_dilation: bool,
}

pub(super) fn process(
    prediction: ArrayView2<'_, f32>,
    original_width: u32,
    original_height: u32,
    options: &DbOptions,
) -> TextDetectionResult {
    let height = prediction.shape()[0];
    let width = prediction.shape()[1];
    let mut bitmap = GrayImage::from_fn(width as u32, height as u32, |x, y| {
        Luma([u8::from(
            prediction[(y as usize, x as usize)] > options.threshold,
        )])
    });
    if options.use_dilation {
        bitmap = dilate_two_by_two(&bitmap);
    }
    let contours = find_contours::<u32>(&bitmap);
    let mut regions = Vec::with_capacity(contours.len().min(options.max_candidates));
    for contour in contours.into_iter().take(options.max_candidates) {
        let points: Vec<Point> = contour
            .points
            .iter()
            .map(|point| Point::new(point.x as f32, point.y as f32))
            .collect();
        let points = simplify_chain_points(&points);
        let Some(initial) = minimum_area_rect(&points) else {
            continue;
        };
        if initial.min_side() < 3.0 {
            continue;
        }
        let initial_points = initial.ordered_points();
        let score = polygon_score(prediction, &initial_points);
        if score < options.box_threshold {
            continue;
        }
        let Some(expanded) = unclip(&initial_points, options.unclip_ratio) else {
            continue;
        };
        let Some(final_rectangle) = minimum_area_rect(&expanded) else {
            continue;
        };
        if final_rectangle.min_side() < 5.0 {
            continue;
        }

        let width_scale = original_width as f32 / width as f32;
        let height_scale = original_height as f32 / height as f32;
        let points = final_rectangle.ordered_points().map(|point| {
            Point::new(
                (point.x * width_scale)
                    .round_ties_even()
                    .clamp(0.0, original_width as f32),
                (point.y * height_scale)
                    .round_ties_even()
                    .clamp(0.0, original_height as f32),
            )
        });
        regions.push(DetectedTextRegion { points, score });
    }

    TextDetectionResult { regions }
}

fn polygon_score(prediction: ArrayView2<'_, f32>, points: &[Point; 4]) -> f32 {
    let height = prediction.shape()[0];
    let width = prediction.shape()[1];
    let min_x = points
        .iter()
        .map(|point| point.x)
        .fold(f32::INFINITY, f32::min)
        .floor()
        .clamp(0.0, width.saturating_sub(1) as f32) as usize;
    let max_x = points
        .iter()
        .map(|point| point.x)
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .clamp(0.0, width.saturating_sub(1) as f32) as usize;
    let min_y = points
        .iter()
        .map(|point| point.y)
        .fold(f32::INFINITY, f32::min)
        .floor()
        .clamp(0.0, height.saturating_sub(1) as f32) as usize;
    let max_y = points
        .iter()
        .map(|point| point.y)
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .clamp(0.0, height.saturating_sub(1) as f32) as usize;
    let mut mask = GrayImage::new((max_x - min_x + 1) as u32, (max_y - min_y + 1) as u32);
    let mask_points: Vec<ImagePoint<i32>> = points
        .iter()
        .map(|point| ImagePoint::new(point.x as i32 - min_x as i32, point.y as i32 - min_y as i32))
        .collect();
    fill_polygon_like_opencv(&mut mask, &mask_points);

    let mut total = 0.0;
    let mut count = 0usize;
    for (x, y, pixel) in mask.enumerate_pixels() {
        if pixel[0] != 0 {
            total += prediction[(min_y + y as usize, min_x + x as usize)];
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}

#[derive(Clone, Copy)]
struct PolygonEdge {
    y0: i32,
    y1: i32,
    x: i64,
    dx: i64,
}

fn fill_polygon_like_opencv(mask: &mut GrayImage, points: &[ImagePoint<i32>]) {
    const XY_SHIFT: u32 = 16;
    const XY_HALF: i64 = 1 << (XY_SHIFT - 1);

    if points.len() < 3 {
        return;
    }
    let mut edges = Vec::with_capacity(points.len());
    let mut previous = points[points.len() - 1];
    for &point in points {
        let mut clipped_previous = previous;
        let mut clipped_point = point;
        let endpoints_outside = !point_in_image(mask, previous) || !point_in_image(mask, point);
        if clip_line_like_opencv(mask, &mut clipped_previous, &mut clipped_point) {
            draw_line_like_opencv(mask, clipped_previous, clipped_point);
        }
        if previous.y != point.y {
            let mut previous_x = i64::from(previous.x) << XY_SHIFT;
            let mut point_x = i64::from(point.x) << XY_SHIFT;
            let mut previous_y = i64::from(previous.y);
            let mut point_y = i64::from(point.y);
            if endpoints_outside {
                if clipped_previous.y != clipped_point.y {
                    previous_x = i64::from(clipped_previous.x) << XY_SHIFT;
                    point_x = i64::from(clipped_point.x) << XY_SHIFT;
                    previous_y = i64::from(clipped_previous.y);
                    point_y = i64::from(clipped_point.y);
                }
            } else {
                previous_x += XY_HALF;
                point_x += XY_HALF;
            }
            let dx = (point_x - previous_x) / (point_y - previous_y);
            let (y0, y1, x) = if previous.y < point.y {
                (
                    previous.y,
                    point.y,
                    previous_x + (i64::from(previous.y) - previous_y) * dx,
                )
            } else {
                (
                    point.y,
                    previous.y,
                    point_x + (i64::from(point.y) - point_y) * dx,
                )
            };
            edges.push(PolygonEdge { y0, y1, x, dx });
        }
        previous = point;
    }

    let first_y = edges.iter().map(|edge| edge.y0).min().unwrap_or(0).max(0);
    let last_y = edges
        .iter()
        .map(|edge| edge.y1)
        .max()
        .unwrap_or(0)
        .min(mask.height() as i32);
    for y in first_y..last_y {
        let mut intersections: Vec<i64> = edges
            .iter()
            .filter(|edge| edge.y0 <= y && y < edge.y1)
            .map(|edge| edge.x + i64::from(y - edge.y0) * edge.dx)
            .collect();
        intersections.sort_unstable();
        for pair in intersections.chunks_exact(2) {
            let mut left = (pair[0] >> XY_SHIFT) as i32;
            let mut right = (pair[1] >> XY_SHIFT) as i32;
            left = left.max(0);
            right = right.min(mask.width() as i32 - 1);
            if left <= right {
                for x in left..=right {
                    mask.put_pixel(x as u32, y as u32, Luma([1]));
                }
            }
        }
    }
}

fn draw_line_like_opencv(
    image: &mut GrayImage,
    mut first: ImagePoint<i32>,
    mut second: ImagePoint<i32>,
) {
    let mut delta_x = 1;
    let mut delta_y = 1;
    let mut dx = second.x - first.x;
    let mut dy = second.y - first.y;
    if dx < 0 {
        dx = -dx;
        dy = -dy;
        std::mem::swap(&mut first, &mut second);
    }
    if dy < 0 {
        dy = -dy;
        delta_y = -1;
    }
    let vertical = dy > dx;
    if vertical {
        std::mem::swap(&mut dx, &mut dy);
        std::mem::swap(&mut delta_x, &mut delta_y);
    }

    let mut error = dx - 2 * dy;
    let plus_delta = 2 * dx;
    let minus_delta = -2 * dy;
    let mut minus_x = delta_x;
    let mut plus_x = 0;
    let mut minus_y = 0;
    let mut plus_y = delta_y;
    if vertical {
        std::mem::swap(&mut plus_y, &mut plus_x);
        std::mem::swap(&mut minus_y, &mut minus_x);
    }

    let mut point = first;
    for _ in 0..=dx {
        image.put_pixel(point.x as u32, point.y as u32, Luma([1]));
        let mask = if error < 0 { -1 } else { 0 };
        error += minus_delta + (plus_delta & mask);
        point.x += minus_x + (plus_x & mask);
        point.y += minus_y + (plus_y & mask);
    }
}

fn point_in_image(image: &GrayImage, point: ImagePoint<i32>) -> bool {
    point.x >= 0
        && point.y >= 0
        && point.x < image.width() as i32
        && point.y < image.height() as i32
}

fn clip_line_like_opencv(
    image: &GrayImage,
    first: &mut ImagePoint<i32>,
    second: &mut ImagePoint<i32>,
) -> bool {
    let right = i64::from(image.width()) - 1;
    let bottom = i64::from(image.height()) - 1;
    let (mut x1, mut y1) = (i64::from(first.x), i64::from(first.y));
    let (mut x2, mut y2) = (i64::from(second.x), i64::from(second.y));
    let outcode = |x: i64, y: i64| {
        i32::from(x < 0)
            + i32::from(x > right) * 2
            + i32::from(y < 0) * 4
            + i32::from(y > bottom) * 8
    };
    let mut first_code = outcode(x1, y1);
    let mut second_code = outcode(x2, y2);

    if first_code & second_code == 0 && first_code | second_code != 0 {
        if first_code & 12 != 0 {
            let boundary = if first_code < 8 { 0 } else { bottom };
            x1 += ((boundary - y1) as f64 * (x2 - x1) as f64 / (y2 - y1) as f64) as i64;
            y1 = boundary;
            first_code = i32::from(x1 < 0) + i32::from(x1 > right) * 2;
        }
        if second_code & 12 != 0 {
            let boundary = if second_code < 8 { 0 } else { bottom };
            x2 += ((boundary - y2) as f64 * (x2 - x1) as f64 / (y2 - y1) as f64) as i64;
            y2 = boundary;
            second_code = i32::from(x2 < 0) + i32::from(x2 > right) * 2;
        }
        if first_code & second_code == 0 && first_code | second_code != 0 {
            if first_code != 0 {
                let boundary = if first_code == 1 { 0 } else { right };
                y1 += ((boundary - x1) as f64 * (y2 - y1) as f64 / (x2 - x1) as f64) as i64;
                x1 = boundary;
                first_code = 0;
            }
            if second_code != 0 {
                let boundary = if second_code == 1 { 0 } else { right };
                y2 += ((boundary - x2) as f64 * (y2 - y1) as f64 / (x2 - x1) as f64) as i64;
                x2 = boundary;
                second_code = 0;
            }
        }
    }
    *first = ImagePoint::new(x1 as i32, y1 as i32);
    *second = ImagePoint::new(x2 as i32, y2 as i32);
    first_code | second_code == 0
}

fn unclip(points: &[Point; 4], ratio: f32) -> Option<Vec<Point>> {
    let polygon_area = points
        .iter()
        .zip(points.iter().cycle().skip(1))
        .take(points.len())
        .map(|(left, right)| {
            f64::from(left.x) * f64::from(right.y) - f64::from(right.x) * f64::from(left.y)
        })
        .sum::<f64>()
        .abs()
        * 0.5;
    let perimeter = points
        .iter()
        .zip(points.iter().cycle().skip(1))
        .take(points.len())
        .map(|(left, right)| {
            (f64::from(right.x) - f64::from(left.x)).hypot(f64::from(right.y) - f64::from(left.y))
        })
        .sum::<f64>();
    if polygon_area <= f64::EPSILON || perimeter <= f64::EPSILON {
        return None;
    }
    let mut vertices: Vec<[i64; 2]> = points
        .iter()
        .map(|point| {
            // pyclipper accepts these f32 values through its integer path,
            // truncating each coordinate toward zero before offsetting.
            [point.x as i64, point.y as i64]
        })
        .collect();
    let mut path = ClipperPath {
        vertices: vertices.as_mut_ptr(),
        vertices_count: vertices.len(),
        closed: 1,
    };
    let mut polygon = ClipperPolygon {
        paths: &mut path,
        paths_count: 1,
        type_: PolyType_ptSubject,
    };
    let input = ClipperPolygons {
        polygons: &mut polygon,
        polygons_count: 1,
    };
    // clipper-sys wraps the same Clipper 6 integer offsetter used by pyclipper.
    // SAFETY: every input pointer refers to a live vector/struct for the call,
    // and the wrapper copies the input before returning its owned result.
    let output = unsafe {
        offset(
            2.0,
            0.25,
            JoinType_jtRound,
            EndType_etClosedPolygon,
            input,
            polygon_area * f64::from(ratio) / perimeter,
        )
    };
    let mut paths = Vec::new();
    if output.polygons_count != 0 && !output.polygons.is_null() {
        // SAFETY: the FFI result reports this non-null allocation and length.
        let polygons =
            unsafe { std::slice::from_raw_parts(output.polygons, output.polygons_count) };
        for polygon in polygons {
            if polygon.paths_count == 0 || polygon.paths.is_null() {
                continue;
            }
            // SAFETY: this polygon reports a non-null path allocation and length.
            let output_paths =
                unsafe { std::slice::from_raw_parts(polygon.paths, polygon.paths_count) };
            for path in output_paths {
                if path.vertices_count == 0 || path.vertices.is_null() {
                    continue;
                }
                // SAFETY: this path reports a non-null vertex allocation and length.
                let output_vertices =
                    unsafe { std::slice::from_raw_parts(path.vertices, path.vertices_count) };
                paths.push(
                    output_vertices
                        .iter()
                        .map(|point| Point::new(point[0] as f32, point[1] as f32))
                        .collect::<Vec<_>>(),
                );
            }
        }
    }
    // SAFETY: output was allocated by clipper-sys and has only been copied above.
    unsafe { free_polygons(output) };
    if paths.len() != 1 || paths[0].len() < 3 {
        return None;
    }
    paths.into_iter().next()
}

fn simplify_chain_points(points: &[Point]) -> Vec<Point> {
    if points.len() <= 3 {
        return points.to_vec();
    }
    let mut result = Vec::with_capacity(points.len());
    for index in 0..points.len() {
        let previous = points[(index + points.len() - 1) % points.len()];
        let current = points[index];
        let next = points[(index + 1) % points.len()];
        let incoming = (
            direction_component(current.x - previous.x),
            direction_component(current.y - previous.y),
        );
        let outgoing = (
            direction_component(next.x - current.x),
            direction_component(next.y - current.y),
        );
        if incoming != outgoing {
            result.push(current);
        }
    }
    if result.len() < 3 {
        points.to_vec()
    } else {
        result
    }
}

fn direction_component(value: f32) -> i8 {
    if value < 0.0 {
        -1
    } else if value > 0.0 {
        1
    } else {
        0
    }
}

fn dilate_two_by_two(source: &GrayImage) -> GrayImage {
    GrayImage::from_fn(source.width(), source.height(), |x, y| {
        let active = [
            (x, y),
            (x.saturating_sub(1), y),
            (x, y.saturating_sub(1)),
            (x.saturating_sub(1), y.saturating_sub(1)),
        ]
        .into_iter()
        .any(|(sample_x, sample_y)| source.get_pixel(sample_x, sample_y)[0] != 0);
        Luma([u8::from(active)])
    })
}

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};

    use super::*;

    #[test]
    fn polygon_fill_matches_opencv_line_8() {
        let cases: &[(u32, u32, &[[i32; 2]], &str)] = &[
            (
                50,
                30,
                &[[2, 5], [45, 1], [47, 20], [3, 28]],
                "10863fdfb1d17bf2343375cd55f25b617e8f59b8e5cc147dec7c141d53924a30",
            ),
            (
                50,
                15,
                &[[0, 5], [48, 1], [49, 8], [1, 12]],
                "19c02571839a2eb4ab0aa26b46a6f4163053cc420802f3920e34dfe764e63096",
            ),
            (
                31,
                41,
                &[[4, 0], [25, 3], [29, 39], [1, 35]],
                "79ba4dee5731f718b1a742005b32f1100afd42c61c1090371fb8650b83e83546",
            ),
            (
                20,
                20,
                &[[0, 0], [19, 0], [19, 19], [0, 19]],
                "afb6cecb558a0858d1ae9afeff0650dbfe2c109a2e6ee8a4d71e94e88bc43015",
            ),
            (
                20,
                20,
                &[[10, 0], [19, 10], [10, 19], [0, 10]],
                "98362c7ed3aa75f59b5f3210fc4bae5f6495e4725cf04dbefe87b15ce9cca7f9",
            ),
            (
                20,
                20,
                &[[-5, 3], [15, -4], [25, 12], [2, 25]],
                "5d73a3adfdfceb7b53ac59028115346593519fb144621976ccb6ae013a49dcda",
            ),
            (
                20,
                20,
                &[[-10, 5], [10, 2], [10, 17], [-10, 15]],
                "34893cbc25067e3bb7adb040aab52516b9b585a7e192ec34c077a63c17358e61",
            ),
            (
                20,
                20,
                &[[5, -10], [17, 5], [14, 25], [2, 10]],
                "51f188a715874d2644b1218c0b853d9562f4dd578a191ca33a9f010c3d51f8a7",
            ),
        ];

        for &(width, height, source_points, expected) in cases {
            let points: Vec<_> = source_points
                .iter()
                .map(|point| ImagePoint::new(point[0], point[1]))
                .collect();
            let mut mask = GrayImage::new(width, height);
            fill_polygon_like_opencv(&mut mask, &points);
            let digest = Sha256::digest(mask.as_raw())
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            assert_eq!(digest, expected, "{source_points:?}");
        }
    }

    #[test]
    fn chain_simplification_keeps_axis_diagonal_transitions() {
        let points = [
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(3.0, 1.0),
            Point::new(4.0, 2.0),
            Point::new(4.0, 3.0),
            Point::new(4.0, 4.0),
            Point::new(3.0, 4.0),
            Point::new(2.0, 4.0),
            Point::new(1.0, 4.0),
            Point::new(0.0, 3.0),
            Point::new(0.0, 2.0),
            Point::new(0.0, 1.0),
        ];
        assert_eq!(
            simplify_chain_points(&points),
            vec![
                Point::new(0.0, 0.0),
                Point::new(2.0, 0.0),
                Point::new(4.0, 2.0),
                Point::new(4.0, 4.0),
                Point::new(1.0, 4.0),
                Point::new(0.0, 3.0),
            ]
        );
    }

    #[test]
    fn unclip_matches_pyclipper_round_offset() {
        let points = [
            Point::new(0.95121765, 6.439024),
            Point::new(11.682926, -2.1463418),
            Point::new(20.560974, 8.951219),
            Point::new(9.829266, 17.536585),
        ];
        let expected = [
            [15.0, -5.0],
            [24.0, 5.0],
            [25.0, 8.0],
            [24.0, 11.0],
            [23.0, 12.0],
            [12.0, 21.0],
            [9.0, 22.0],
            [5.0, 20.0],
            [-4.0, 9.0],
            [-5.0, 6.0],
            [-4.0, 3.0],
            [-3.0, 2.0],
            [8.0, -6.0],
            [11.0, -7.0],
        ]
        .map(|point| Point::new(point[0], point[1]));
        assert_eq!(unclip(&points, 1.4).unwrap(), expected);
    }
}
