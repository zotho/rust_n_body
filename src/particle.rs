use std::collections::vec_deque::VecDeque;

use macroquad::prelude::*;
use vecmath::{vec2_add, vec2_scale, Vector2};

use crate::trail_point::TrailPoint;

const DRAW_END: bool = true;
const TRAIL_END: usize = 100;

// #[cfg(feature = "raindow_trail")]
// const COLORS: [Color; 23] = [
//     LIGHTGRAY, DARKGRAY, YELLOW, GOLD, ORANGE, PINK, RED, MAROON, GREEN, LIME, DARKGREEN, SKYBLUE,
//     BLUE, DARKBLUE, PURPLE, VIOLET, DARKPURPLE, BEIGE, BROWN, DARKBROWN, WHITE, BLACK, MAGENTA,
// ];

// Sorted by hue
#[cfg(feature = "raindow_trail")]
const COLORS: [Color; 23] = [
    BLACK, DARKGRAY, LIGHTGRAY, WHITE, BEIGE, BROWN, DARKBROWN, ORANGE, GOLD, YELLOW, GREEN, LIME,
    DARKGREEN, SKYBLUE, BLUE, DARKBLUE, PURPLE, VIOLET, DARKPURPLE, MAGENTA, PINK, MAROON, RED,
];

#[cfg(feature = "tail_area_optimize")]
fn triangle_area(length1: f32, length2: f32, angle_sin: f32) -> f32 {
    length1 * length2 * angle_sin / 2.0
}

#[cfg(feature = "tail_area_optimize")]
fn need_retain(l0: f32, l1: f32, _: usize, angle_cos: f32, retained_area: f32) -> (bool, f32) {
    // if angle_cos < 0.8 {
    // return (false, 0.0);
    // }
    let angle_sin = (1.0 - angle_cos.powi(2)).sqrt();
    let area = triangle_area(l0, l1, angle_sin);

    if area + retained_area < 2.0 {
        (true, area)
    } else {
        (false, 0.0)
    }
}

#[cfg(not(feature = "tail_area_optimize"))]
fn need_retain(l0: f32, l1: f32, i: usize, angle_cos: f32, _: f32) -> (bool, f32) {
    let sum = l0 + l1;

    let (collinear_factor, max_len) = match i {
        i if i > 10000 => (0.8, 300.0),
        i if i > 1000 => (0.9, 100.0),
        i if i > 100 => (0.99, 30.0),
        _ => (0.999, 10.0),
    };

    let collinear = angle_cos > collinear_factor;
    // let equal_len = l0 > sum / 3.0 && l1 > sum / 3.0;
    let not_very_large = sum < max_len;
    let need_retain = collinear && not_very_large;
    (need_retain, 0.0)
}

pub struct Particle {
    pub pos: Vector2<f64>,
    pub vel: Vector2<f64>,
    pub mass: f64,
    pub trail: VecDeque<TrailPoint>,
}

impl Particle {
    pub fn new() -> Particle {
        Particle {
            pos: [0.0, 0.0],
            vel: [0.0, 0.0],
            mass: 1.0,
            trail: VecDeque::with_capacity(256),
        }
    }

    pub fn draw(&self) {
        if DRAW_END {
            let size = self.size();
            draw_poly(
                self.pos[0] as f32,
                self.pos[1] as f32,
                self.sides(),
                size,
                0.0,
                WHITE,
            );
        } else {
            let trail_len = self.trail.len();
            if trail_len > TRAIL_END {
                for i in 1..trail_len {
                    let [x1, y1] = self.trail[i - 1].pos;
                    let [x2, y2] = self.trail[i].pos;

                    if i == TRAIL_END {
                        let size = self.trail[i].size;
                        draw_poly(x2 as f32, y2 as f32, self.sides(), size, 0.0, WHITE);
                        break;
                    }

                    draw_line(x1 as f32, y1 as f32, x2 as f32, y2 as f32, 1.0, RED);
                }
            }
        }
    }

    #[cfg(feature = "raindow_trail")]
    pub fn draw_trail(&self) {
        #[cfg(not(feature = "draw_lines"))]
        {
            let len = self.trail.len();
            for i in (1..len).rev() {
                let [x1, y1] = self.trail[i - 1].pos;
                let [x2, y2] = self.trail[i].pos;

                draw_line(
                    x1 as f32,
                    y1 as f32,
                    x2 as f32,
                    y2 as f32,
                    2.0,
                    COLORS[(len - i) % COLORS.len()],
                );
            }
        }

        #[cfg(feature = "draw_lines")]
        {
            let points = self
                .trail
                .iter()
                .map(|p| {
                    let [x, y] = p.pos;
                    (x as f32, y as f32)
                })
                .collect::<Vec<(f32, f32)>>();
            draw_lines(points, 2.0, RED);
        }
    }

    #[cfg(not(feature = "raindow_trail"))]
    pub fn draw_trail(&self) {
        let len = self.trail.len();
        for i in (1..len).rev() {
            let [x1, y1] = self.trail[i - 1].pos;
            let [x2, y2] = self.trail[i].pos;

            draw_line(x1 as f32, y1 as f32, x2 as f32, y2 as f32, 1.0, RED);
        }
    }

    pub fn optimize_trail(&mut self) -> usize {
        let start_retain = 2;
        let n_iter = self.trail.len() / 2;

        let mut retain_total = 0;
        // let mut last_retain = 0;
        for j in (start_retain / 2)..n_iter {
            let i = j * 2;
            let [x0, y0] = self.trail[i - 2].pos;
            let [x1, y1] = self.trail[i - 1].pos;
            let [x2, y2] = self.trail[i].pos;
            let d0 = vec2(x1 as f32 - x0 as f32, y1 as f32 - y0 as f32);
            let d1 = vec2(x2 as f32 - x1 as f32, y2 as f32 - y1 as f32);

            let (l0, l1) = (d0.length(), d1.length());
            let prod = d0.dot(d1);
            let angle_cos = (prod / (l0 * l1)).min(1.0).max(-1.0);

            let (need_retain, retained_area) =
                need_retain(l0, l1, i, angle_cos, self.trail[i - 1].area);

            if need_retain {
                // last_retain = i;
                let total_area = self.trail[i - 1].area + retained_area;
                self.trail[i - 2].area += total_area / 2.0;
                self.trail[i - 1].area = 0.0;
                self.trail[i].area += total_area / 2.0;
                self.trail[i - 1].keep = false;
                retain_total += 1;
            }
        }
        // println!("{:>5}/{:>5}", last_retain, self.trail.len());
        self.trail.retain(|trail_point| trail_point.keep);

        retain_total
    }

    pub fn update_pos(&mut self, dt: f64) {
        self.pos = vec2_add(self.pos, vec2_scale(self.vel, dt));
        let size = self.size();
        self.trail.push_front(TrailPoint::new(self.pos, size, true));
    }

    pub fn size(&self) -> f32 {
        (self.mass as f32).powf(1.0 / 3.0)
    }

    pub fn sides(&self) -> u8 {
        (self.size().powf(0.8).ceil() as u8).max(3)
    }
}
