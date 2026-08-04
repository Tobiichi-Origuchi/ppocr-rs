use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

/// A point in source-image pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    pub(crate) const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RotatedRect {
    pub center: Point,
    pub width: f32,
    pub height: f32,
    pub angle: f32,
}

impl RotatedRect {
    pub fn min_side(self) -> f32 {
        self.width.min(self.height)
    }

    pub fn ordered_points(self) -> [Point; 4] {
        let (sin, cos) = self.angle.sin_cos();
        let half_width = self.width * 0.5;
        let half_height = self.height * 0.5;
        let mut points = [
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height),
        ]
        .map(|(x, y)| {
            Point::new(
                x * cos - y * sin + self.center.x,
                x * sin + y * cos + self.center.y,
            )
        });

        // PaddleX sorts by x and then chooses TL, TR, BR, BL by y.
        points.sort_by(|left, right| left.x.partial_cmp(&right.x).unwrap_or(Ordering::Equal));
        let (top_left, bottom_left) = if points[1].y > points[0].y {
            (points[0], points[1])
        } else {
            (points[1], points[0])
        };
        let (top_right, bottom_right) = if points[3].y > points[2].y {
            (points[2], points[3])
        } else {
            (points[3], points[2])
        };
        [top_left, top_right, bottom_right, bottom_left]
    }
}

pub(crate) fn minimum_area_rect(points: &[Point]) -> Option<RotatedRect> {
    let hull = convex_hull(points);
    if hull.len() < 3 {
        return None;
    }

    let mut best: Option<(f32, RotatedRect)> = None;
    for index in 0..hull.len() {
        let origin = hull[index];
        let end = hull[(index + 1) % hull.len()];
        let edge_x = end.x - origin.x;
        let edge_y = end.y - origin.y;
        let length = edge_x.hypot(edge_y);
        if length <= f32::EPSILON {
            continue;
        }
        let axis_x = edge_x / length;
        let axis_y = edge_y / length;
        let perpendicular_x = -axis_y;
        let perpendicular_y = axis_x;
        let (mut min_axis, mut max_axis) = (f32::INFINITY, f32::NEG_INFINITY);
        let (mut min_perp, mut max_perp) = (f32::INFINITY, f32::NEG_INFINITY);
        for point in &hull {
            let relative_x = point.x - origin.x;
            let relative_y = point.y - origin.y;
            let projected_axis = relative_x * axis_x + relative_y * axis_y;
            let projected_perp = relative_x * perpendicular_x + relative_y * perpendicular_y;
            min_axis = min_axis.min(projected_axis);
            max_axis = max_axis.max(projected_axis);
            min_perp = min_perp.min(projected_perp);
            max_perp = max_perp.max(projected_perp);
        }
        let width = max_axis - min_axis;
        let height = max_perp - min_perp;
        let area = width * height;
        if best.as_ref().is_none_or(|(best_area, _)| area < *best_area) {
            let axis_center = (min_axis + max_axis) * 0.5;
            let perp_center = (min_perp + max_perp) * 0.5;
            let center = Point::new(
                origin.x + axis_center * axis_x + perp_center * perpendicular_x,
                origin.y + axis_center * axis_y + perp_center * perpendicular_y,
            );
            best = Some((
                area,
                RotatedRect {
                    center,
                    width,
                    height,
                    angle: axis_y.atan2(axis_x),
                },
            ));
        }
    }
    best.map(|(_, rectangle)| rectangle)
}

fn convex_hull(points: &[Point]) -> Vec<Point> {
    let mut points = points.to_vec();
    points.sort_by(|left, right| {
        left.x
            .total_cmp(&right.x)
            .then_with(|| left.y.total_cmp(&right.y))
    });
    points.dedup_by(|left, right| left.x == right.x && left.y == right.y);
    if points.len() <= 2 {
        return points;
    }

    let mut lower = Vec::with_capacity(points.len());
    for point in &points {
        while lower.len() >= 2
            && cross(lower[lower.len() - 2], lower[lower.len() - 1], *point) <= 0.0
        {
            lower.pop();
        }
        lower.push(*point);
    }
    let mut upper = Vec::with_capacity(points.len());
    for point in points.iter().rev() {
        while upper.len() >= 2
            && cross(upper[upper.len() - 2], upper[upper.len() - 1], *point) <= 0.0
        {
            upper.pop();
        }
        upper.push(*point);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

fn cross(origin: Point, left: Point, right: Point) -> f32 {
    (left.x - origin.x) * (right.y - origin.y) - (left.y - origin.y) * (right.x - origin.x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_axis_aligned_rectangle() {
        let points = [
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 5.0),
            Point::new(0.0, 5.0),
        ];
        let rectangle = minimum_area_rect(&points).unwrap();
        assert!((rectangle.min_side() - 5.0).abs() < 1e-5);
        assert_eq!(rectangle.ordered_points()[0], Point::new(0.0, 0.0));
    }
}
