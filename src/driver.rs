//! Driver: turns a fixed schedule into traffic. Open mode sleeps until absolute deadlines, never in a loop — a late wakeup is caught up in a burst carrying the true intended times, so harness lag is data instead of being absorbed into later requests.

use crate::{
    clock::RunClock,
    config::{Mode, RunConfig},
    recorder::{Record, RecordSink},
    target::{Status, Target},
};
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

/// Round each absolute ideal time, rather than multiplying a rounded period.
fn intended_ns(i: u64, interval: f64) -> u64 {
    (i as f64 * interval).round().min(u64::MAX as f64) as u64
}

pub async fn run(
    cfg: &RunConfig,
    clock: Arc<RunClock>,
    sink: RecordSink,
    target: Target,
) -> DriverCounters {
    match cfg.mode {
        Mode::Open => run_open(cfg, clock, sink, target).await,
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
) -> DriverCounters {
    let interval = interval_ns(cfg.rate);
    let end_ns = cfg.duration_ns();
    let cap = Arc::new(Semaphore::new(cfg.max_in_flight));
    let mut counters = DriverCounters::default();
    let mut tasks = JoinSet::new();
    let mut id: u64 = 0;

    while intended_ns(id, interval) < end_ns {
        if clock.now_ns() < intended_ns(id, interval) {
            sleep_until_ns(&clock, intended_ns(id, interval)).await;
        }
        // Catch-up burst: release everything that fell due while we slept; queueing it would convert open loop into closed.
        while intended_ns(id, interval) < end_ns && clock.now_ns() >= intended_ns(id, interval) {
            counters.intended += 1;
            match cap.clone().try_acquire_owned() {
                Ok(permit) => {
                    counters.fired += 1;
                    let sink = sink.clone();
                    let clock = clock.clone();
                    let target = target.clone();
                    tasks.spawn(async move {
                        let actual_ns = clock.now_ns();
                        let outcome = target.send().await;
                        let completed_ns = clock.now_ns();
                        sink.send(Record {
                            req_id: id,
                            intended_ns: intended_ns(id, interval),
                            actual_ns,
                            completed_ns,
                            status: outcome.status,
                            http_status: outcome.http_status,
                            bytes: outcome.bytes,
                        });
                        drop(permit);
                    });
                }
                Err(_) => {
                    counters.clipped += 1;
                    // A clipped request was never sent, so it carries no lag sample.
                    sink.send(Record {
                        req_id: id,
                        intended_ns: intended_ns(id, interval),
                        actual_ns: 0,
                        completed_ns: 0,
                        status: Status::Clipped,
                        http_status: 0,
                        bytes: 0,
                    });
                }
            }
            id += 1;
        }
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
                let outcome = target.send().await;
                let completed_ns = clock.now_ns();
                sink.send(Record {
                    req_id: id,
                    intended_ns: actual_ns,
                    actual_ns,
                    completed_ns,
                    status: outcome.status,
                    http_status: outcome.http_status,
                    bytes: outcome.bytes,
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

    #[test]
    fn intended_times_scale_linearly_and_saturate_instead_of_wrapping() {
        let interval = interval_ns(1.0);
        assert_eq!(intended_ns(0, interval), 0);
        assert_eq!(intended_ns(3, interval), 3_000_000_000);
        // an absurd rate still yields an interval of at least 1 ns
        assert_eq!(interval_ns(1e12), 1.0);
        // u64::MAX × 1s overflows u64; the product must saturate, not wrap
        assert_eq!(intended_ns(u64::MAX, interval), u64::MAX);
    }

    #[test]
    fn fractional_rate_does_not_drift_across_one_second() {
        let interval = interval_ns(70.0);
        assert_eq!(intended_ns(70, interval), 1_000_000_000);
        assert_eq!(
            (0..)
                .take_while(|&i| intended_ns(i, interval) < 1_000_000_000)
                .count(),
            70
        );
    }

    #[test]
    fn duration_boundary_defines_the_arrival_count() {
        let interval = interval_ns(2.1);
        let arrivals: Vec<u64> = (0..)
            .map(|i| intended_ns(i, interval))
            .take_while(|&ns| ns < 1_000_000_000)
            .collect();
        assert_eq!(arrivals, vec![0, 476_190_476, 952_380_952]);
    }

    #[test]
    fn long_fractional_schedule_uses_the_absolute_formula() {
        let rate = 3.3;
        let interval = interval_ns(rate);
        for i in [1, 10, 1_000, 1_000_000] {
            assert_eq!(
                intended_ns(i, interval),
                (i as f64 * 1e9 / rate).round() as u64
            );
        }
    }
}
