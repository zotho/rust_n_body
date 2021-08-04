use vecmath::Vector2;

#[derive(Copy, Clone)]
pub struct TrailPoint {
    pub pos: Vector2<f64>,
    pub size: f32,
    pub keep: bool,
    pub area: f32,
}

impl TrailPoint {
    pub fn new(pos: Vector2<f64>, size: f32, keep: bool) -> TrailPoint {
        TrailPoint {
            pos,
            size,
            keep,
            area: 0.0,
        }
    }
}

impl From<TrailPoint> for (Vector2<f64>, f32, bool, f32) {
    fn from(e: TrailPoint) -> (Vector2<f64>, f32, bool, f32) {
        let TrailPoint {
            pos,
            size,
            keep,
            area,
        } = e;
        (pos, size, keep, area)
    }
}
