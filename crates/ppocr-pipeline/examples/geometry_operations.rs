use std::{env, fs, path::PathBuf};

use ppocr::{Point, image_geometry};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args_os().skip(1);
    let input = PathBuf::from(arguments.next().ok_or("missing input image")?);
    let output = PathBuf::from(arguments.next().ok_or("missing output image")?);
    let coordinates: Vec<f32> = arguments
        .map(|value| value.to_string_lossy().parse())
        .collect::<Result<_, _>>()?;
    if coordinates.len() != 8 {
        return Err("expected eight coordinates: x0 y0 ... x3 y3".into());
    }
    let image = image::load_from_memory(&fs::read(input)?)?.into_rgb8();
    let points = [
        Point {
            x: coordinates[0],
            y: coordinates[1],
        },
        Point {
            x: coordinates[2],
            y: coordinates[3],
        },
        Point {
            x: coordinates[4],
            y: coordinates[5],
        },
        Point {
            x: coordinates[6],
            y: coordinates[7],
        },
    ];
    image_geometry::crop_text_region(&image, &points)?.save(output)?;
    Ok(())
}
