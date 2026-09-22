//! saturation-lab: open-loop load generator. Arrivals on a fixed schedule regardless of what the server is doing, and every stall shows up in the numbers instead of hiding in them.

pub mod clock;
pub mod config;
pub mod driver;
pub mod recorder;
pub mod runner;
pub mod target;
