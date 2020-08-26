extern crate rand;

use std::time::{Instant};
use std::ops::{Add, Sub, Mul};
use std::collections::vec_deque::VecDeque;

use macroquad::*;
use vecmath::*;
use itertools::Itertools;
use rand::Rng;

const NANO_MULT: f64 = 1000000000.0;

const G: f64 = 5.0;
const GRAV_POWER: f64 = 2.0;
// const G: f64 = 1000.0;
// const GRAV_POWER: f64 = 3.0;

fn lerp<T: Copy + Add<Output = T> + Mul<Output = T> + Sub<Output = T>>(v0: T, v1: T, t: T) -> T {
    v0 + t * (v1 - v0)
}

fn lerp_mass_vec(v1: Vector2<f64>, v2: Vector2<f64>, m1: f64, m2: f64) -> Vector2<f64> {
    vec2_scale(vec2_add(vec2_scale(v1, m1), vec2_scale(v2, m2)), 1.0 / (m1 + m2))
}

fn text_line(string: &str, line_no: u8) {
    draw_text(string, 10.0 , 5.0 + 25.0 * (line_no - 1) as f32, 25.0, BLACK);
}

struct Particle {
    pos: Vector2<f64>,
    vel: Vector2<f64>,
    mass: f64,
    trail: VecDeque<(Vector2<f64>, f32)>,
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
        let trail_len = self.trail.len();
        let trail_end = 100;

        if trail_len > trail_end {
            for i in 1..trail_len {
                let ([x1, y1], _s1) = self.trail[i - 1];
                let ([x2, y2], s2) = self.trail[i];

                if i == trail_end {
                    let sides = (s2.powf(0.8).ceil() as u8).max(3);
                    draw_poly(x2 as f32, y2 as f32, sides, s2, 0.0, if self.mass < 0.0 {GREEN} else {WHITE});
                    break;
                }

                draw_line(x1 as f32, y1 as f32, x2 as f32, y2 as f32, 1.0, RED);
            }
        }
        //  else {
        //     let [x, y] = self.pos;
        //     let size = self.size();
        //     draw_poly(x as f32, y as f32, (size.ceil() as u8).max(3), size, 0.0, WHITE);
        // }
    }

    fn update_pos(self: &mut Self, dt: f64) {
        self.pos = vec2_add(self.pos, vec2_scale(self.vel, dt));
        let size = self.size();
        self.trail.push_front((self.pos, size));
    }

    fn size(self: &Self) -> f32 {
        (self.mass as f32).powf(1.0 / 3.0)
    }
}

fn window_conf() -> Conf {
    Conf {
        // fullscreen: true,
        window_width: 1920,
        window_height: 1080,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut now = Instant::now();
    let mut nanoseconds;
    let mut min_fps = std::f64::INFINITY;

    let mut rng = rand::thread_rng();

    let (mut width, mut height) = (screen_width(), screen_height());

    let mut mouse_pos = (width / 2.0, height / 2.0);
    let mut mouse_pos_prev;
    let mut mouse_vel;

    let mut n_init_on_step = 100;
    let mut n_updates = 1;
    let mut time_speed = 1.0;

    let init_size = 100;
    let init_speed = 100.0;
    
    let mut particles = Vec::with_capacity(init_size);

    for i in 0..init_size {
        let k = i as f64 / init_size as f64 * 2.0 * std::f64::consts::PI;

        let mut p = Particle::new();
        p.pos = [k.cos() * 100.0 + width as f64 / 2.0, k.sin() * 100.0 + height as f64 / 2.0];
        p.vel = [(k + std::f64::consts::PI / 2.0).cos() * init_speed, (k + std::f64::consts::PI / 2.0).sin() * init_speed];
        particles.push(p);
    }

    loop {
        width = screen_width();
        height = screen_height();
        
        nanoseconds = now.elapsed().as_nanos() as f64;
        let dt = nanoseconds / NANO_MULT;
        let fps = NANO_MULT / nanoseconds;
        min_fps = min_fps.min(fps);
        now = Instant::now();

        mouse_pos_prev = mouse_pos;
        mouse_pos = mouse_position();
        mouse_vel = {
            let (x, y) = mouse_pos;
            let (px, py) = mouse_pos_prev;
            (x - px, y - py)
        };

        if is_key_down(KeyCode::Escape) | is_key_down(KeyCode::Q) {
            break;
        }
        if is_key_pressed(KeyCode::Left) {
            time_speed /= 1.1;
        }
        if is_key_pressed(KeyCode::Right) {
            time_speed *= 1.1;
        }
        if is_key_pressed(KeyCode::Down) {
            n_updates = 0.max(n_updates - 1);
        }
        if is_key_pressed(KeyCode::Up) {
            n_updates += 1;
        }
        if is_key_down(KeyCode::Minus) {
            n_init_on_step = 0.max(n_init_on_step - 1);
        }
        if is_key_down(KeyCode::Equal) {
            n_init_on_step += 1;
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

        let particles_now = Instant::now();

        for _ in 0..n_init_on_step {
            let mut p = Particle::new();
            p.pos = [rng.gen_range(0.0, width as f64), rng.gen_range(0.0, height as f64)];
            p.vel = [rng.gen_range(-200.0, 200.0), rng.gen_range(-200.0, 200.0)];
            particles.push(p);
        }

        clear_background(GRAY);

        let (x, y) = mouse_pos;
        draw_circle(x, y, 10.0, WHITE);

        let mut particles_time_vel = 0;
        for _ in 0..n_updates {
            let dt = dt * time_speed / n_updates as f64;

            let particles_now_vel = Instant::now();
            (0..particles.len()).tuple_combinations().for_each(|(i1, i2)| {
                let (part1, part2) = particles.split_at_mut(i2);
                let (p1, p2) = (&mut part1[i1], &mut part2[0]);
    
                if p1.mass < 0.0 || p2.mass < 0.0 {
                    return
                }
    
                let d12 = vec2_sub(p2.pos, p1.pos);
                let inv_len = vec2_inv_len(d12);
                if inv_len > 0.05 {
                    p1.pos = lerp_mass_vec(p1.pos, p2.pos, p1.mass, p2.mass);
                    p1.vel = lerp_mass_vec(p1.vel, p2.vel, p1.mass, p2.mass);
                    p1.mass = p1.mass + p2.mass;
                    p2.mass = -0.00001;
                } else {
                    let a = vec2_scale(d12, G * inv_len.powf(GRAV_POWER));
                    p1.vel = vec2_add(p1.vel, vec2_scale(a, dt * p2.mass));
                    p2.vel = vec2_add(p2.vel, vec2_scale(a, -dt * p1.mass));
                }
            });
            particles_time_vel += particles_now_vel.elapsed().as_micros();
    
            particles.retain(|p| {
                let [x, y] = p.pos;
                p.mass > 0.0 && x > -1000.0 && x < width as f64 + 1000.0 && y > -1000.0 && y < height as f64 + 1000.0
            });
    
            particles.iter_mut().for_each(|p| {
                p.update_pos(dt);
                // p.vel = vec2_scale(p.vel, 1.0001);
            });
        }

        let particles_time = particles_now.elapsed().as_micros();

        particles.iter().for_each(|p| {
            p.draw();
        });

        text_line(format!("N PARTICLES: {}", particles.len()).as_str(), 1);
        text_line(format!("FPS: {:.1} (MIN: {:.1})", fps, min_fps).as_str(), 2);
        text_line(format!("ALL TIME:  {:>7}", nanoseconds as u128 / 1000).as_str(), 3);
        text_line(format!("PARTICLES: {:>7} ({} - {})", particles_time, particles_time_vel, particles_time_vel as f32 / particles_time as f32).as_str(), 4);
        text_line(format!("REST:      {:>7}", nanoseconds as u128  / 1000 - particles_time).as_str(), 5);
        text_line(format!("MASSES: {:?}", particles.iter().map(|p| p.mass).collect::<Vec<f64>>()).as_str(), 6);
        text_line(format!("TRAILS: {:?}", particles.iter().map(|p| p.trail.len()).collect::<Vec<usize>>()).as_str(), 7);
        text_line(format!("SPEED: {}", time_speed).as_str(), 8);
        text_line(format!("UPDATE STEPS: {}", n_updates).as_str(), 9);
        text_line(format!("N INIT NEW PARTICLES: {}", n_init_on_step).as_str(), 10);

        next_frame().await;
    }
}