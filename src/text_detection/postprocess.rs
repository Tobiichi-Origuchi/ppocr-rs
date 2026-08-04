use clipper2_rust::{
    clipper::inflate_paths_d,
    core::{PathD, PathsD, PointD, area},
    offset::{EndType, JoinType},
};
use image::{GrayImage, Luma};
use imageproc::{contours::find_contours, drawing::draw_polygon_mut, point::Point as ImagePoint};
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
    draw_polygon_mut(&mut mask, &mask_points, Luma([1]));

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

fn unclip(points: &[Point; 4], ratio: f32) -> Option<Vec<Point>> {
    let path: PathD = points
        .iter()
        .map(|point| PointD {
            x: point.x as f64,
            y: point.y as f64,
        })
        .collect();
    let polygon_area = area(&path).abs();
    let perimeter = path
        .iter()
        .zip(path.iter().cycle().skip(1))
        .take(path.len())
        .map(|(left, right)| (right.x - left.x).hypot(right.y - left.y))
        .sum::<f64>();
    if polygon_area <= f64::EPSILON || perimeter <= f64::EPSILON {
        return None;
    }
    let paths: PathsD = vec![path];
    let offset = inflate_paths_d(
        &paths,
        polygon_area * ratio as f64 / perimeter,
        JoinType::Round,
        EndType::Polygon,
        2.0,
        2,
        0.0,
    );
    if offset.len() != 1 {
        return None;
    }
    let points = offset[0]
        .iter()
        .map(|point| Point::new(point.x as f32, point.y as f32))
        .collect::<Vec<_>>();
    (points.len() >= 3).then_some(points)
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
            (current.x - previous.x).signum(),
            (current.y - previous.y).signum(),
        );
        let outgoing = ((next.x - current.x).signum(), (next.y - current.y).signum());
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
