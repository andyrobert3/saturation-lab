//! Driver: turns a fixed schedule into traffic. Open mode sleeps until absolute deadlines, never in a loop — a late wakeup is caught up in a burst carrying the true intended times, so harness lag is data instead of being absorbed into later requests.

use crate::{
    clock::RunClock,
    config::{Arrival, Mode, RunConfig},
    recorder::{Record, RecordSink},
    target::{Status, Target},
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::Gamma;
use serde::Serialize;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use tokio::{sync::Semaphore, task::JoinSet};

#[derive(Default, Debug, Clone, Copy, Serialize)]
pub struct DriverCounters {
    pub intended: u64,
    pub fired: u64,
    pub clipped: u64,
}

/// Keep fractional nanoseconds so rounding cannot accumulate across arrivals.
fn interval_ns(rate_hz: f64) -> f64 {
    (1e9 / rate_hz).max(1.0)
}

/// Absolute intended arrival times: the deterministic ladder for Uniform, an
/// exponential-inter-arrival (Poisson) draw for Poisson. Both are reproducible
/// given the seed; the schedule owns its position, so a late wakeup is caught
/// up in a burst carrying the true intended times.
struct ArrivalSchedule {
    next_ns: u64,
    end_ns: u64,
    i: u64,               // Uniform ladder counter
    interval_ns: f64,     // Uniform: 1e9/rate
    rate: f64,            // Poisson: arrivals per second
    rng: Option<StdRng>,  // Poisson only
    gamma: Option<Gamma<f64>>, // Gamma only: inter-arrival seconds
}

impl ArrivalSchedule {
    fn new(cfg: &RunConfig, end_ns: u64, seed: u64) -> Self {
        match cfg.arrival {
            Arrival::Uniform => Self {
                next_ns: 0,
                end_ns,
                i: 0,
                interval_ns: interval_ns(cfg.rate),
                rate: 0.0,
                rng: None,
                gamma: None,
            },
            Arrival::Poisson => Self {
                next_ns: 0,
                end_ns,
                i: 0,
                interval_ns: 0.0,
                rate: cfg.rate,
                rng: Some(StdRng::seed_from_u64(seed)),
                gamma: None,
            },
            // Shape 1/cv² and scale cv²/rate give mean 1/rate and the requested cv.
            Arrival::Gamma { cv } => Self {
                next_ns: 0,
                end_ns,
                i: 0,
                interval_ns: 0.0,
                rate: cfg.rate,
                rng: Some(StdRng::seed_from_u64(seed)),
                gamma: Gamma::new(1.0 / (cv * cv), cv * cv / cfg.rate).ok(),
            },
        }
    }

    /// Next intended deadline, or None once past end_ns.
    fn next_intended(&mut self) -> Option<u64> {
        if self.next_ns >= self.end_ns {
            return None;
        }
        let due = self.next_ns;
        match self.rng.as_mut() {
            None => {
                self.i += 1;
                self.next_ns =
                    (self.i as f64 * self.interval_ns).round().min(u64::MAX as f64) as u64;
            }
            Some(rng) => {
                // Exponential inter-arrival: u uniform in (0,1), x = -ln(u)/λ
                // has density λ·e^(−λx) — a Poisson process with rate λ.
                let dt = match &self.gamma {
                    Some(gamma) => rng.sample(gamma) * 1e9,
                    None => {
                        let u: f64 = rng.sample(rand::distributions::Open01);
                        (-u.ln() / self.rate) * 1e9
                    }
                };
                self.next_ns = self.next_ns.saturating_add(dt.max(1.0) as u64);
            }
        }
        Some(due)
    }
}

pub async fn run(
    cfg: &RunConfig,
    clock: Arc<RunClock>,
    sink: RecordSink,
    target: Target,
    seed: u64,
) -> DriverCounters {
    match cfg.mode {
        Mode::Open => run_open(cfg, clock, sink, target, seed).await,
        Mode::Closed => run_closed(cfg, clock, sink, target).await,
    }
}

async fn sleep_until_ns(clock: &RunClock, due_ns: u64) {
    let due = clock.start_instant() + std::time::Duration::from_nanos(due_ns);
    tokio::time::sleep_until(due.into()).await;
}

async fn run_open(
    cfg: &RunConfig,
    clock: Arc<RunClock>,
    sink: RecordSink,
    target: Target,
    seed: u64,
) -> DriverCounters {
    let end_ns = cfg.duration_ns();
    let cap = Arc::new(Semaphore::new(cfg.max_in_flight));
    let mut counters = DriverCounters::default();
    let mut tasks = JoinSet::new();
    let mut sched = ArrivalSchedule::new(cfg, end_ns, seed);
    let mut id: u64 = 0;

    while let Some(intended_ns) = sched.next_intended() {
        if clock.now_ns() < intended_ns {
            sleep_until_ns(&clock, intended_ns).await;
        }
        // Behind schedule -> fires immediately; that is the catch-up burst,
        // carrying the true intended time. No inner while-loop.
        counters.intended += 1;
        match cap.clone().try_acquire_owned() {
            Ok(permit) => {
                counters.fired += 1;
                let sink = sink.clone();
                let clock = clock.clone();
                let target = target.clone();
                tasks.spawn(async move {
                    let actual_ns = clock.now_ns();
                    let outcome = target.send(&clock).await;
                    let completed_ns = clock.now_ns();
                    sink.send(Record {
                        req_id: id,
                        intended_ns,
                        actual_ns,
                        first_output_ns: outcome.first_output_ns,
                        last_output_ns: outcome.last_output_ns,
                        completed_ns,
                        output_tokens: outcome.output_tokens,
                        status: outcome.status,
                        http_status: outcome.http_status,
                        bytes: outcome.bytes,
                        output_events: outcome.output_events,
                    });
                    drop(permit);
                });
            }
            Err(_) => {
                counters.clipped += 1;
                // A clipped request was never sent, so it carries no lag sample.
                sink.send(Record {
                    req_id: id,
                    intended_ns,
                    actual_ns: 0,
                    first_output_ns: 0,
                    last_output_ns: 0,
                    completed_ns: 0,
                    output_tokens: 0,
                    status: Status::Clipped,
                    http_status: 0,
                    bytes: 0,
                    output_events: Vec::new(),
                });
            }
        }
        id += 1;
    }

    while let Some(joined) = tasks.join_next().await {
        let _ = joined;
    }
    counters
}

async fn run_closed(
    cfg: &RunConfig,
    clock: Arc<RunClock>,
    sink: RecordSink,
    target: Target,
) -> DriverCounters {
    let end_ns = cfg.duration_ns();
    let next_id = Arc::new(AtomicU64::new(0));
    let mut tasks = JoinSet::new();

    for _ in 0..cfg.concurrency {
        let sink = sink.clone();
        let clock = clock.clone();
        let target = target.clone();
        let next_id = next_id.clone();
        tasks.spawn(async move {
            // Closed loop: each request only exists because the previous one finished, so intended == actual by construction.
            while clock.now_ns() < end_ns {
                let id = next_id.fetch_add(1, Ordering::Relaxed);
                let actual_ns = clock.now_ns();
                let outcome = target.send(&clock).await;
                let completed_ns = clock.now_ns();
                sink.send(Record {
                    req_id: id,
                    intended_ns: actual_ns,
                    actual_ns,
                    first_output_ns: outcome.first_output_ns,
                    last_output_ns: outcome.last_output_ns,
                    completed_ns,
                    output_tokens: outcome.output_tokens,
                    status: outcome.status,
                    http_status: outcome.http_status,
                    bytes: outcome.bytes,
                    output_events: outcome.output_events,
                });
            }
        });
    }

    while let Some(joined) = tasks.join_next().await {
        let _ = joined;
    }

    DriverCounters {
        intended: next_id.load(Ordering::Relaxed),
        fired: next_id.load(Ordering::Relaxed),
        clipped: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cfg(rate: f64, arrival: Arrival) -> RunConfig {
        RunConfig {
            name: "t".into(),
            target_url: "http://localhost:8000".into(),
            model: "m".into(),
            prompt: "p".into(),
            mode: Mode::Open,
            rate,
            arrival,
            seed: 0,
            duration_s: 1.0,
            warmup_s: 0.0,
            max_in_flight: 32,
            concurrency: 1,
            request_timeout_s: 30.0,
            channel_capacity: 1024,
            target_notes: None,
        }
    }

    #[test]
    fn uniform_schedule_matches_the_ladder() {
        let cfg = test_cfg(2.0, Arrival::Uniform);
        let mut sched = ArrivalSchedule::new(&cfg, u64::MAX / 4, 0);
        let times: Vec<u64> = (0..6).map(|_| sched.next_intended().unwrap()).collect();
        assert_eq!(
            times,
            vec![
                0,
                500_000_000,
                1_000_000_000,
                1_500_000_000,
                2_000_000_000,
                2_500_000_000
            ]
        );
    }

    #[test]
    fn poisson_schedule_is_reproducible_with_seed() {
        let collect = |seed: u64| {
            let cfg = test_cfg(10.0, Arrival::Poisson);
            let mut sched = ArrivalSchedule::new(&cfg, 1_000_000_000, seed);
            std::iter::from_fn(|| sched.next_intended()).collect::<Vec<u64>>()
        };
        assert_eq!(collect(7), collect(7));
        assert_ne!(collect(7), collect(8));
    }

    #[test]
    fn gamma_schedule_has_the_requested_mean_and_cv() {
        let cfg = test_cfg(50.0, Arrival::Gamma { cv: 2.0 });
        let mut sched = ArrivalSchedule::new(&cfg, 2_000_000_000_000, 1);
        let times: Vec<f64> = std::iter::from_fn(|| sched.next_intended())
            .map(|t| t as f64)
            .collect();
        let gaps: Vec<f64> = times.windows(2).map(|w| w[1] - w[0]).collect();
        let mean = gaps.iter().sum::<f64>() / gaps.len() as f64;
        let var = gaps.iter().map(|g| (g - mean).powi(2)).sum::<f64>() / gaps.len() as f64;
        assert!((mean / 20_000_000.0 - 1.0).abs() < 0.03, "mean gap {mean}");
        assert!((var.sqrt() / mean / 2.0 - 1.0).abs() < 0.05, "cv {}", var.sqrt() / mean);
    }
}
