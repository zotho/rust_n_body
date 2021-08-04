const MICRO_MULT: f64 = 1000000.0;

fn get_time() -> f64 {
    macroquad::prelude::get_time() * MICRO_MULT
}

#[derive(Default, Debug)]
pub struct Time {
    started: bool,
    time: f64,
}

impl Time {
    pub fn start(&mut self) {
        assert!(!self.started);
        self.started = true;
        self.time = get_time();
    }

    pub fn stop(&mut self) {
        assert!(self.started);
        self.started = false;
        self.time = get_time() - self.time;
    }

    pub fn time(&self) -> f64 {
        assert!(!self.started);
        self.time
    }
}

#[derive(Default, Debug)]
pub struct Timings {
    pub all: Time,
    pub particles: Time,
    pub particles_vel: Time,
    pub particles_retain: Time,
    pub particles_trail: Time,
    pub particles_draw: Time,
}
