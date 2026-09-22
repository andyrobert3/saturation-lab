//! Time discipline: every duration is measured on the monotonic clock; the wall clock is read once at run start for provenance (when a run happened, never how long anything took).

use std::{
    process::Command,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

pub struct RunClock {
    anchor: Instant,
    wall: SystemTime,
}

impl RunClock {
    pub fn new() -> Self {
        Self {
            anchor: Instant::now(),
            wall: SystemTime::now(),
        }
    }

    /// Nanoseconds since the run anchor: the single source of every timestamp in the CSV.
    pub fn now_ns(&self) -> u64 {
        self.anchor.elapsed().as_nanos() as u64
    }

    /// Absolute deadlines: `sleep_until(anchor + t)` targets fixed points, so a late wakeup is caught up rather than accumulated into every later one.
    pub fn start_instant(&self) -> Instant {
        self.anchor
    }

    /// Wall-clock instant of the anchor, as epoch ns, for provenance only.
    pub fn wall_unix_ns(&self) -> u64 {
        self.wall
            .duration_since(UNIX_EPOCH)
            .expect("system clock is after 1970")
            .as_nanos() as u64
    }

    /// UTC stamp for the run directory name. Shells out to `date`: one call per run, so a date library is not worth it.
    pub fn utc_stamp() -> String {
        Command::new("date")
            .args(["-u", "+%Y%m%dT%H%M%SZ"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "00000000T000000Z".to_string())
    }
}

impl Default for RunClock {
    fn default() -> Self {
        Self::new()
    }
}

/// What `sleep_until` actually costs on this machine, measured with the same timer the driver uses, before the anchor. Median, not max: one unlucky sample must not raise the budget for the whole run.
pub async fn wakeup_floor_ns() -> u64 {
    const SAMPLES: usize = 32;
    const STEP: Duration = Duration::from_millis(2);

    let mut overshoot = Vec::with_capacity(SAMPLES);
    let mut deadline = tokio::time::Instant::now() + STEP;
    for _ in 0..SAMPLES {
        tokio::time::sleep_until(deadline).await;
        overshoot.push(
            tokio::time::Instant::now()
                .saturating_duration_since(deadline)
                .as_nanos() as u64,
        );
        deadline += STEP;
    }
    overshoot.sort_unstable();
    overshoot[SAMPLES / 2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn now_ns_is_monotonic_and_anchored_near_zero() {
        let clock = RunClock::new();
        let a = clock.now_ns();
        std::thread::sleep(std::time::Duration::from_millis(5));
        let b = clock.now_ns();
        assert!(a < b, "clock went backwards: {a} then {b}");
        assert!(a < 50_000_000, "anchor should start near zero, got {a}");
    }

    #[test]
    fn wall_clock_lands_in_this_century() {
        // catches a units mistake (ms/us/ns)
        let secs = RunClock::new().wall_unix_ns() / 1_000_000_000;
        assert!((1_700_000_000..2_100_000_000).contains(&secs), "got {secs}");
    }

    #[tokio::test]
    async fn wakeup_floor_is_measured_and_plausible() {
        let floor = wakeup_floor_ns().await;
        // catches the calibration measuring the wrong thing, not a machine-specific number
        assert!(floor < 50_000_000, "floor {floor}ns is not a wakeup delay");
    }
}
