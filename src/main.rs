// extern crate rand;

// use std::time::{Instant};
use std::ops::{Add, Sub, Mul};
use std::collections::vec_deque::VecDeque;

use macroquad::*;
use vecmath::*;
use itertools::Itertools;
// use rand::Rng;

#[cfg(feature = "raindow_trail")]
const COLORS: [Color; 23] = [LIGHTGRAY, DARKGRAY, YELLOW, GOLD, ORANGE, PINK, RED, MAROON, GREEN, LIME, DARKGREEN, SKYBLUE, BLUE, DARKBLUE, PURPLE, VIOLET, DARKPURPLE, BEIGE, BROWN, DARKBROWN, WHITE, BLACK, MAGENTA];

const MICRO_MULT: f64 = 1000000.0;

const DRAW_END: bool = true;
const TRAIL_END: usize = 100;

// const G: f64 = 5.0;
// const GRAV_POWER: f64 = 2.0;
const G: f64 = 1000.0;
const GRAV_POWER: i32 = 3;

const MAX_COORD: f32 = 1000.0;
const MAX_INV_LEN: f64 = 1.0;

// const FIXED_DT: bool = true;
const DT: f64 = 0.004;

fn lerp<T: Copy + Add<Output = T> + Mul<Output = T> + Sub<Output = T>>(v0: T, v1: T, t: T) -> T {
    v0 + t * (v1 - v0)
}

fn lerp_mass_vec(v1: Vector2<f64>, v2: Vector2<f64>, m1: f64, m2: f64) -> Vector2<f64> {
    vec2_scale(vec2_add(vec2_scale(v1, m1), vec2_scale(v2, m2)), 1.0 / (m1 + m2))
}

fn text_line(string: &str, line_no: u8) {
    draw_text(string, 10.0 , 5.0 + 25.0 * (line_no - 1) as f32, 25.0, BLACK);
}

#[cfg(feature = "tail_area_optimize")]
fn triangle_area(length1: f32, length2: f32, angle_sin: f32) -> f32 {
    length1 * length2 * angle_sin / 2.0
}

#[cfg(feature = "tail_area_optimize")]
fn need_retain(l0: f32, l1: f32, _: usize, angle_cos: f32) -> bool {
    if angle_cos < 0.8 {
        return false;
    }
    let angle_sin = (1.0 - angle_cos.powi(2)).sqrt();

    triangle_area(l0, l1, angle_sin) < 5.0
}

#[cfg(not(feature = "tail_area_optimize"))]
fn need_retain(l0: f32, l1: f32, i: usize, angle_cos: f32) -> bool {
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
    need_retain
}

#[derive(Copy, Clone)]
struct TrailPoint {
    pos: Vector2<f64>,
    size: f32,
    keep: bool,
}

impl TrailPoint {
    fn new(pos: Vector2<f64>, size: f32, keep: bool) -> TrailPoint {
        TrailPoint {
            pos: pos,
            size: size,
            keep: keep,
        }
    }
}

impl From<TrailPoint> for (Vector2<f64>, f32, bool) {
    fn from(e: TrailPoint) -> (Vector2<f64>, f32, bool) {
        let TrailPoint { pos, size, keep } = e;
        (pos, size, keep)
    }
}

struct Particle {
    pos: Vector2<f64>,
    vel: Vector2<f64>,
    mass: f64,
    trail: VecDeque<TrailPoint>,
}

impl Particle {
    fn new() -> Particle {
        Particle {
            pos: [0.0, 0.0],
            vel: [0.0, 0.0],
            mass: 1.0,
            trail: VecDeque::with_capacity(256),
        }
    }

    fn draw(self: &Self) {

        if DRAW_END {
            let size = self.size();
            draw_poly(self.pos[0] as f32, self.pos[1] as f32, self.sides(), size, 0.0, WHITE);
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
    fn draw_trail(self: &Self) {
        let len = self.trail.len();
        for i in 1..len {
            let [x1, y1] = self.trail[i - 1].pos;
            let [x2, y2] = self.trail[i].pos;

            draw_line(x1 as f32, y1 as f32, x2 as f32, y2 as f32, 2.0, COLORS[(len - i) % COLORS.len()]);
        }
    }

    #[cfg(not(feature = "raindow_trail"))]
    fn draw_trail(self: &Self) {
        let len = self.trail.len();
        for i in 1..len {
            let [x1, y1] = self.trail[i - 1].pos;
            let [x2, y2] = self.trail[i].pos;

            draw_line(x1 as f32, y1 as f32, x2 as f32, y2 as f32, 1.0, RED);
        }
    }

    fn optimize_trail(self: &mut Self) -> usize {
        let start_retain = 40;
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

            let need_retain = need_retain(l0, l1, i, angle_cos);

            if need_retain {
                // last_retain = i;
                retain_total += 1;
            }
            self.trail[i - 1].keep = !need_retain;
        }
        // println!("{:>5}/{:>5}", last_retain, self.trail.len());
        self.trail.retain(|trail_point| trail_point.keep);

        retain_total
    }

    fn update_pos(self: &mut Self, dt: f64) {
        self.pos = vec2_add(self.pos, vec2_scale(self.vel, dt));
        let size = self.size();
        self.trail.push_front(TrailPoint::new(self.pos, size, true));
    }

    fn size(self: &Self) -> f32 {
        (self.mass as f32).powf(1.0 / 3.0)
    }

    fn sides(self: &Self) -> u8 {
        (self.size().powf(0.8).ceil() as u8).max(3)
    }
}

fn window_conf() -> Conf {
    Conf {
        fullscreen: true,
        window_width: 1920,
        window_height: 1080,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    // let mut now = Instant::now();
    // let mut nanoseconds;
    // let mut min_fps = std::f64::INFINITY;

    // let mut rng = rand::thread_rng();

    let (mut width, mut height) = (screen_width(), screen_height());

    let mut mouse_pos = (width / 2.0, height / 2.0);
    let mut mouse_pos_prev;
    let mut mouse_vel;

    // let mut n_init_on_step: usize = 0;

    let mut n_updates: usize = 100;
    let mut time_speed: f64 = 1.0;

    let mut draw_debug = false;

    let init_size = 50;
    let init_speed = 20.0;
    let init_radius = 100.0;

    let mut particles = Vec::with_capacity(init_size);

    for i in 0..init_size {
        let k = i as f64 / init_size as f64 * 2.0 * std::f64::consts::PI;

        let mut p = Particle::new();
        p.pos = [k.cos() * init_radius + width as f64 / 2.0, k.sin() * init_radius + height as f64 / 2.0];
        p.vel = [(k + std::f64::consts::PI / 2.0).cos() * init_speed, (k + std::f64::consts::PI / 2.0).sin() * init_speed];
        particles.push(p);
    }

    loop {
        width = screen_width();
        height = screen_height();

        // nanoseconds = now.elapsed().as_nanos() as f64;
        // let dt = nanoseconds / NANO_MULT;
        // let fps = NANO_MULT / nanoseconds;
        // min_fps = min_fps.min(fps);
        // now = Instant::now();

        mouse_pos_prev = mouse_pos;
        mouse_pos = mouse_position();
        mouse_vel = {
            let (x, y) = mouse_pos;
            let (px, py) = mouse_pos_prev;
            (x - px, y - py)
        };

        #[cfg(not(target_arch = "wasm32"))]
        if is_key_down(KeyCode::Escape) | is_key_down(KeyCode::Q) {
            break;
        }
        if is_key_pressed(KeyCode::Left) {
            time_speed /= 1.1;
        }
        if is_key_pressed(KeyCode::Right) {
            time_speed *= 1.1;
        }
        if is_key_down(KeyCode::Down) {
            n_updates = n_updates.checked_sub(1).unwrap_or(0);
        }
        if is_key_down(KeyCode::Up) {
            n_updates += 1;
        }
        // if is_key_down(KeyCode::Minus) {
            // n_init_on_step = 0.max(n_init_on_step - 1);
        // }
        // if is_key_down(KeyCode::Equal) {
            // n_init_on_step += 1;
        // }
        if is_key_pressed(KeyCode::D) {
            draw_debug = !draw_debug;
        }

        if is_mouse_button_down(MouseButton::Left) {
            let (px, py) = mouse_pos_prev;
            let (px, py) = (px as f64, py as f64);
            let (x, y) = mouse_pos;
            let (x, y) = (x as f64, y as f64);
            let (vx, vy) = mouse_vel;
            let (vx, vy) = (vx as f64, vy as f64);

            let n_particles = 10;
            for i in 0..n_particles {
                let k = i as f64 / (n_particles - 1) as f64;

                let mut p = Particle::new();
                p.pos = [lerp(px, x, k), lerp(py, y, k)];
                p.vel = [vx as f64, vy as f64];
                particles.push(p);
            }
        }

        let particles_time = get_time();

        // for _ in 0..n_init_on_step {
            // let mut p = Particle::new();
            // p.pos = [rng.gen_range(0.0, width as f64), rng.gen_range(0.0, height as f64)];
            // p.vel = [rng.gen_range(-200.0, 200.0), rng.gen_range(-200.0, 200.0)];
            // particles.push(p);
        // }

        clear_background(GRAY);

        // let (x, y) = mouse_pos;
        // draw_circle(x, y, 10.0, WHITE);

        let mut particles_time_vel = 0.0;
        for _ in 0..n_updates {
            let current_dt;
            // if FIXED_DT {
                current_dt = DT;
            // } else {
                // current_dt = dt * time_speed / n_updates as f64;
            // }

            let particles_now_vel = get_time();
            (0..particles.len()).tuple_combinations().for_each(|(i1, i2)| {
                let (part1, part2) = particles.split_at_mut(i2);
                let (p1, p2) = (&mut part1[i1], &mut part2[0]);

                if p1.mass < 0.0 || p2.mass < 0.0 {
                    return
                }

                let d12 = vec2_sub(p2.pos, p1.pos);
                let inv_len = vec2_inv_len(d12);
                if inv_len * (p1.mass * p2.mass).powf(1.0 / GRAV_POWER as f64) > MAX_INV_LEN {
                    p1.pos = lerp_mass_vec(p1.pos, p2.pos, p1.mass, p2.mass);
                    p1.vel = lerp_mass_vec(p1.vel, p2.vel, p1.mass, p2.mass);
                    p1.mass = p1.mass + p2.mass;
                    p2.mass = -0.00001;
                } else {
                    let a = vec2_scale(d12, G * inv_len.powi(GRAV_POWER));
                    p1.vel = vec2_add(p1.vel, vec2_scale(a, current_dt * p2.mass));
                    p2.vel = vec2_add(p2.vel, vec2_scale(a, -current_dt * p1.mass));
                }
            });
            particles_time_vel += get_time() - particles_now_vel;

            particles.retain(|particle| {
                let [x, y] = particle.pos;
                let (x, y) = (x as f32, y as f32);
                particle.mass > 0.0 &&
                    x > -MAX_COORD &&
                    x < width + MAX_COORD &&
                    y > -MAX_COORD &&
                    y < height + MAX_COORD
            });

            particles.iter_mut().for_each(|p| {
                p.update_pos(current_dt);
                // p.vel = vec2_scale(p.vel, 0.99999);
            });
        }

        let particles_time_retain = get_time();
        let mut retain_count = 0;
        particles.iter_mut().for_each(|p| {
            retain_count += p.optimize_trail();
        });
        let particles_time_retain = get_time() - particles_time_retain;

        let particles_time_trail = get_time();
        particles.iter().for_each(|p| {
            p.draw_trail();
        });
        let particles_time_trail = get_time() - particles_time_trail;

        let particles_time_draw = get_time();
        particles.iter().for_each(|p| {
            p.draw();
        });
        let particles_time_draw = get_time() - particles_time_draw;

        let particles_time = get_time() - particles_time;

        let all_time = get_frame_time() as f64;

        if draw_debug {
            text_line(format!("N PARTICLES: {}", particles.len()).as_str(), 1);
            text_line(format!("FPS: {}", get_fps()).as_str(), 2);
            text_line(format!("ALL TIME:  {:>7.0}", all_time * MICRO_MULT).as_str(), 3);
            text_line(format!("PARTICLES: {:>7.0}", particles_time.min(all_time) * MICRO_MULT).as_str(), 4);
            text_line(format!("         - {:>7.0} grav   ({:.3} part)", particles_time_vel * MICRO_MULT, particles_time_vel / particles_time ).as_str(), 5);
            text_line(format!("         - {:>7.0} retain ({:.3} part)", particles_time_retain * MICRO_MULT, particles_time_retain / particles_time).as_str(), 6);
            text_line(format!("         - {:>7.0} trail  ({:.3} part)", particles_time_trail * MICRO_MULT, particles_time_trail / particles_time).as_str(), 7);
            text_line(format!("         - {:>7.0} draw   ({:.3} part)", particles_time_draw * MICRO_MULT, particles_time_draw / particles_time).as_str(), 8);
            text_line(format!("REST:      {:>7.0}", (all_time - particles_time).max(0.0) * MICRO_MULT).as_str(), 9);
            text_line(format!("MASSES: {:?}", particles.iter().map(|p| p.mass).collect::<Vec<f64>>()).as_str(), 10);
            text_line(format!("TRAILS: {:?}", particles.iter().map(|p| p.trail.len()).sum::<usize>()).as_str(), 11);
            text_line(format!("SPEED: {}", time_speed).as_str(), 12);
            text_line(format!("UPDATE STEPS: {}", n_updates).as_str(), 13);
            // text_line(format!("N INIT NEW PARTICLES: {}", n_init_on_step).as_str(), 14);
            text_line(format!("N RETAINED: {}", retain_count).as_str(), 15);
        }

        next_frame().await;
    }
}