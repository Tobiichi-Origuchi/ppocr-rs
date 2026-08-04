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
    pub angle_degrees: f32,
}

impl RotatedRect {
    pub fn min_side(self) -> f32 {
        self.width.min(self.height)
    }

    pub fn ordered_points(self) -> [Point; 4] {
        // Match RotatedRect::points: OpenCV converts its stored f32 degree
        // angle back to radians in f64, then rounds sin/cos to f32.
        let angle = f64::from(self.angle_degrees) * std::f64::consts::PI / 180.0;
        let b = (angle.cos() as f32) * 0.5;
        let a = (angle.sin() as f32) * 0.5;
        let mut points = [
            Point::new(
                self.center.x - a * self.height - b * self.width,
                self.center.y + b * self.height - a * self.width,
            ),
            Point::new(
                self.center.x + a * self.height - b * self.width,
                self.center.y - b * self.height - a * self.width,
            ),
            Point::new(0.0, 0.0),
            Point::new(0.0, 0.0),
        ];
        points[2] = Point::new(
            2.0 * self.center.x - points[0].x,
            2.0 * self.center.y - points[0].y,
        );
        points[3] = Point::new(
            2.0 * self.center.x - points[1].x,
            2.0 * self.center.y - points[1].y,
        );

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

    let count = hull.len();
    let mut vectors = Vec::with_capacity(count);
    let mut inverse_lengths = Vec::with_capacity(count);
    let mut left = 0;
    let mut bottom = 0;
    let mut right = 0;
    let mut top = 0;
    for index in 0..hull.len() {
        if hull[index].x < hull[left].x {
            left = index;
        }
        if hull[index].x > hull[right].x {
            right = index;
        }
        if hull[index].y > hull[top].y {
            top = index;
        }
        if hull[index].y < hull[bottom].y {
            bottom = index;
        }

        let next = hull[(index + 1) % count];
        let dx = f64::from(next.x) - f64::from(hull[index].x);
        let dy = f64::from(next.y) - f64::from(hull[index].y);
        vectors.push(Point::new(dx as f32, dy as f32));
        inverse_lengths.push((1.0 / (dx * dx + dy * dy).sqrt()) as f32);
    }

    let mut orientation = 0.0_f32;
    let mut previous = vectors[count - 1];
    for vector in &vectors {
        let convexity = f64::from(previous.x) * f64::from(vector.y)
            - f64::from(previous.y) * f64::from(vector.x);
        if convexity != 0.0 {
            orientation = if convexity > 0.0 { 1.0 } else { -1.0 };
            break;
        }
        previous = *vector;
    }
    if orientation == 0.0 {
        return None;
    }

    let mut sequence = [bottom, right, top, left];
    let mut best: Option<(usize, f32, f32, f32, usize)> = None;
    let mut best_height = 0.0_f32;
    let mut minimum_area = f32::MAX;

    for _ in 0..count {
        let rotated = [
            vectors[sequence[0]],
            Point::new(vectors[sequence[1]].y, -vectors[sequence[1]].x),
            Point::new(-vectors[sequence[2]].x, -vectors[sequence[2]].y),
            Point::new(-vectors[sequence[3]].y, vectors[sequence[3]].x),
        ];
        let mut main = 0;
        for index in 1..4 {
            if cross_vectors(rotated[index], rotated[main]) > 0.0 {
                main = index;
            }
        }

        let edge_index = sequence[main];
        let lead_x = vectors[edge_index].x * inverse_lengths[edge_index];
        let lead_y = vectors[edge_index].y * inverse_lengths[edge_index];
        let (base_a, base_b) = match main {
            0 => (lead_x, lead_y),
            1 => (lead_y, -lead_x),
            2 => (-lead_x, -lead_y),
            3 => (-lead_y, lead_x),
            _ => unreachable!(),
        };
        sequence[main] = (sequence[main] + 1) % count;

        let width_delta = Point::new(
            hull[sequence[1]].x - hull[sequence[3]].x,
            hull[sequence[1]].y - hull[sequence[3]].y,
        );
        let width = width_delta.x * base_a + width_delta.y * base_b;
        let height_delta = Point::new(
            hull[sequence[2]].x - hull[sequence[0]].x,
            hull[sequence[2]].y - hull[sequence[0]].y,
        );
        let height = -height_delta.x * base_b + height_delta.y * base_a;
        let area = width * height;
        if area <= minimum_area {
            minimum_area = area;
            best = Some((sequence[3], base_a, width, base_b, sequence[0]));
            // OpenCV stores height beside the other state in a byte buffer.
            best_height = height;
        }
    }

    let (left_index, axis_x, width, axis_y, bottom_index) = best?;
    let perpendicular_x = -axis_y;
    let perpendicular_y = axis_x;
    let left_constant = axis_x * hull[left_index].x + hull[left_index].y * axis_y;
    let bottom_constant =
        perpendicular_x * hull[bottom_index].x + hull[bottom_index].y * perpendicular_y;
    let inverse_determinant = 1.0 / (axis_x * perpendicular_y - perpendicular_x * axis_y);
    let corner = Point::new(
        (left_constant * perpendicular_y - bottom_constant * axis_y) * inverse_determinant,
        (axis_x * bottom_constant - perpendicular_x * left_constant) * inverse_determinant,
    );
    let width_vector = Point::new(axis_x * width, axis_y * width);
    let height_vector = Point::new(perpendicular_x * best_height, perpendicular_y * best_height);
    let center = Point::new(
        corner.x + (width_vector.x + height_vector.x) * 0.5,
        corner.y + (width_vector.y + height_vector.y) * 0.5,
    );
    let angle_radians = (f64::from(width_vector.y)).atan2(f64::from(width_vector.x));
    Some(RotatedRect {
        center,
        width: ((f64::from(width_vector.x) * f64::from(width_vector.x)
            + f64::from(width_vector.y) * f64::from(width_vector.y))
        .sqrt()) as f32,
        height: ((f64::from(height_vector.x) * f64::from(height_vector.x)
            + f64::from(height_vector.y) * f64::from(height_vector.y))
        .sqrt()) as f32,
        angle_degrees: (angle_radians * 180.0 / std::f64::consts::PI) as f32,
    })
}

fn cross_vectors(left: Point, right: Point) -> f32 {
    left.x * right.y - left.y * right.x
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
        let top_left = rectangle.ordered_points()[0];
        assert!(top_left.x.abs() < 1e-12);
        assert_eq!(top_left.y, 0.0);
    }

    #[test]
    fn slanted_rectangles_match_opencv() {
        let cases = [
            (
                &[
                    [3.0, 8.0],
                    [4.0, 4.0],
                    [8.0, 1.0],
                    [15.0, 2.0],
                    [19.0, 7.0],
                    [18.0, 11.0],
                    [13.0, 15.0],
                    [7.0, 14.0],
                ][..],
                [
                    [0.95121765, 6.439024],
                    [11.682926, -2.1463418],
                    [20.560974, 8.951219],
                    [9.829266, 17.536585],
                ],
            ),
            (
                &[
                    [10.0, 10.0],
                    [11.0, 2.0],
                    [15.0, 1.0],
                    [25.0, 5.0],
                    [29.0, 16.0],
                    [24.0, 22.0],
                    [18.0, 25.0],
                    [9.0, 20.0],
                    [4.0, 14.0],
                ][..],
                [
                    [13.248707, -1.8549232],
                    [32.772022, 9.53368],
                    [22.181347, 27.689117],
                    [2.6580324, 16.300514],
                ],
            ),
            (
                &[
                    [101.0, 210.0],
                    [104.0, 193.0],
                    [223.0, 170.0],
                    [397.0, 145.0],
                    [402.0, 180.0],
                    [228.0, 205.0],
                ][..],
                [
                    [97.83674, 187.98323],
                    [397.00003, 145.00002],
                    [402.0282, 179.99596],
                    [102.8649, 222.97917],
                ],
            ),
        ];
        for (case, expected) in cases {
            let points: Vec<_> = case
                .iter()
                .map(|point| Point::new(point[0], point[1]))
                .collect();
            for (actual, expected) in minimum_area_rect(&points)
                .unwrap()
                .ordered_points()
                .iter()
                .zip(expected)
            {
                assert!((actual.x - expected[0]).abs() < 3e-5);
                assert!((actual.y - expected[1]).abs() < 3e-5);
            }
        }
    }
}
