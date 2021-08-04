// #![feature(trace_macros)]

use std::ops::{Add, Mul, Sub};

use itertools::Itertools;
use macroquad::prelude::*;
use macroquad::ui::{hash, root_ui, widgets};
use vecmath::{vec2_add, vec2_inv_len, vec2_scale, vec2_sub, Vector2};

mod input;
mod labels;
mod particle;
mod time;
mod trail_point;

use particle::Particle;
use time::Timings;

const MICRO_MULT: f64 = 1000000.0;

fn get_time() -> f64 {
    macroquad::prelude::get_time() * MICRO_MULT
}

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
    vec2_scale(
        vec2_add(vec2_scale(v1, m1), vec2_scale(v2, m2)),
        1.0 / (m1 + m2),
    )
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
    let (mut width, mut height) = (screen_width(), screen_height());

    let mut mouse_pos = (width / 2.0, height / 2.0);
    let mut mouse_pos_prev;
    let mut mouse_vel;

    let mut n_init_on_step: usize = 0;

    let mut n_updates: usize = 400;
    let mut time_speed: f64 = 1.0;

    let init_size = 200;
    let init_speed = 40.0;
    let init_radius = 300.0;

    let mut particles = Vec::with_capacity(init_size);

    let (cw, ch) = (width as f64 / 2.0, height as f64 / 2.0);

    let mut debug_window = true;

    for i in 0..init_size {
        let k = i as f64 / init_size as f64 * 2.0 * std::f64::consts::PI;

        let mut p = Particle::new();
        p.pos = [k.cos() * init_radius + cw, k.sin() * init_radius + ch];
        p.vel = [
            (k + std::f64::consts::PI / 2.0).cos() * init_speed,
            (k + std::f64::consts::PI / 2.0).sin() * init_speed,
        ];
        particles.push(p);
    }

    let mut p = Particle::new();
    p.pos = [cw, ch];
    p.mass = 170.0;
    particles.push(p);

    let mut timings = Timings::default();

    loop {
        width = screen_width();
        height = screen_height();

        mouse_pos_prev = mouse_pos;
        mouse_pos = mouse_position();
        mouse_vel = {
            let (x, y) = mouse_pos;
            let (px, py) = mouse_pos_prev;
            (x - px, y - py)
        };

        // trace_macros!(true);
        // handle_input!(
        //     is_key_down => {
        //         KeyCode::Escape => break
        //         KeyCode::Q => break
        //     }
        // );
        // trace_macros!(false);

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
            n_updates = n_updates.saturating_sub(1);
        }
        if is_key_down(KeyCode::Up) {
            n_updates += 1;
        }
        if is_key_pressed(KeyCode::Minus) {
            n_init_on_step = n_init_on_step.saturating_sub(1);
        }
        if is_key_pressed(KeyCode::Equal) {
            n_init_on_step += 1;
        }
        if is_key_pressed(KeyCode::D) {
            debug_window = !debug_window;
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

        timings.particles.start();

        let vel_range = 50.0;
        for _ in 0..n_init_on_step {
            let mut p = Particle::new();
            p.pos = [
                rand::gen_range(0.0, width as f64),
                rand::gen_range(0.0, height as f64),
            ];
            p.vel = [
                rand::gen_range(-vel_range, vel_range),
                rand::gen_range(-vel_range, vel_range),
            ];
            p.mass = 10.0;
            particles.push(p);
        }

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
            (0..particles.len())
                .tuple_combinations()
                .for_each(|(i1, i2)| {
                    let (part1, part2) = particles.split_at_mut(i2);
                    let (p1, p2) = (&mut part1[i1], &mut part2[0]);

                    if p1.mass < 0.0 || p2.mass < 0.0 {
                        return;
                    }

                    let d12 = vec2_sub(p2.pos, p1.pos);
                    let inv_len = vec2_inv_len(d12);
                    if inv_len * (p1.mass * p2.mass).powf(1.0 / GRAV_POWER as f64) > MAX_INV_LEN {
                        p1.pos = lerp_mass_vec(p1.pos, p2.pos, p1.mass, p2.mass);
                        p1.vel = lerp_mass_vec(p1.vel, p2.vel, p1.mass, p2.mass);
                        p1.mass += p2.mass;
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
                particle.mass > 0.0
                    && x > -MAX_COORD
                    && x < width + MAX_COORD
                    && y > -MAX_COORD
                    && y < height + MAX_COORD
            });

            particles.iter_mut().for_each(|p| {
                p.update_pos(current_dt);
                // p.vel = vec2_scale(p.vel, 1.000001);
                // p.pos = vec2_scale(p.pos, 1.000001);
                // p.mass = p.mass.min(p.mass.powf(0.9999999));
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

        timings.particles.stop();

        let all_time = get_frame_time() as f64 * MICRO_MULT;

        if debug_window {
            widgets::Window::new(
                hash!(),
                glam::f32::Vec2::new(20., 20.),
                glam::Vec2::new(350., 650.)
            )
                .label("Debug (Press D to hide)")
                .titlebar(true)
                .movable(true)
                .ui(&mut *root_ui(), |ui| {
                    let masses: Vec<f64> = particles.iter().map(|p| p.mass).collect();
                    let trails: usize = particles.iter().map(|p| p.trail.len()).sum();
                    let particles_time = timings.particles.time();

                    labels!(ui,
                        "N PARTICLES: {}", particles.len();
                        "FPS: {}", get_fps();
                        "ALL TIME:  {:>7.0}", all_time;
                        "PARTICLES: {:>7.0}", particles_time.min(all_time);
                        "         - {:>7.0} grav   ({:.3} part)", particles_time_vel, particles_time_vel / particles_time;
                        "         - {:>7.0} retain ({:.3} part)", particles_time_retain, particles_time_retain / particles_time;
                        "         - {:>7.0} trail  ({:.3} part)", particles_time_trail, particles_time_trail / particles_time;
                        "         - {:>7.0} draw   ({:.3} part)", particles_time_draw, particles_time_draw / particles_time;
                        "REST:      {:>7.0}", (all_time - particles_time).max(0.0);
                        "MASSES: {:?}", masses;
                        "TRAILS: {:?}", trails;
                        "SPEED: {}", time_speed;
                        "UPDATE STEPS: {}", n_updates;
                        "N INIT NEW PARTICLES: {}", n_init_on_step;
                        "N RETAINED: {}", retain_count;
                    );
                });

            //     WindowParams {
            //         label: "Debug".to_string(),
            //         close_button: true,
            //         movable: false,
            //         titlebar: true,
            //     },
            //     |ui| {
            //         let masses: Vec<f64> = particles.iter().map(|p| p.mass).collect();
            //         let trails: usize = particles.iter().map(|p| p.trail.len()).sum();
            //         let particles_time = timings.particles.time();

            //         lables!(ui,
            //             "N PARTICLES: {}", particles.len();
            //             "FPS: {}", get_fps();
            //             "ALL TIME:  {:>7.0}", all_time;
            //             "PARTICLES: {:>7.0}", particles_time.min(all_time);
            //             "         - {:>7.0} grav   ({:.3} part)", particles_time_vel, particles_time_vel / particles_time;
            //             "         - {:>7.0} retain ({:.3} part)", particles_time_retain, particles_time_retain / particles_time;
            //             "         - {:>7.0} trail  ({:.3} part)", particles_time_trail, particles_time_trail / particles_time;
            //             "         - {:>7.0} draw   ({:.3} part)", particles_time_draw, particles_time_draw / particles_time;
            //             "REST:      {:>7.0}", (all_time - particles_time).max(0.0);
            //             "MASSES: {:?}", masses;
            //             "TRAILS: {:?}", trails;
            //             "SPEED: {}", time_speed;
            //             "UPDATE STEPS: {}", n_updates;
            //             "N INIT NEW PARTICLES: {}", n_init_on_step;
            //             "N RETAINED: {}", retain_count;
            //         );
            //     },
            // );
        }

        // draw_megaui();

        next_frame().await;
    }
}
