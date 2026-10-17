//! Runner: wire-up and verdict. Brings clock, target, recorder and driver together, writes provenance, then judges the run.

use crate::{
    clock::{RunClock, wakeup_floor_ns},
    config::{Arrival, Mode, RunConfig},
    driver::{self, DriverCounters},
    recorder::{Recorder, RecorderSummary},
    target::{self, Target},
};
use anyhow::{Context, Result};
use serde::Serialize;
use std::{fs, path::PathBuf, process::Command, sync::Arc};

pub const TOOL_VERSION: &str = env!("CARGO_PKG_VERSION");
pub const GIT_SHA: &str = env!("GIT_SHA");

pub const MIN_LAG_BUDGET_NS: u64 = 1_000_000;
pub const WAKEUP_FLOOR_MULTIPLIER: u64 = 5;
pub const MAX_ERROR_RATE: f64 = 0.01;
pub const MAX_LATE_DISPATCH_FRACTION: f64 = 0.01;

#[derive(Serialize)]
pub struct Provenance {
    pub tool: String,
    pub git_sha: String,
    pub hostname: String,
    pub anchor_wall_unix_ns: Option<u64>,
    pub run_started_utc: String,
    pub target_url: String,
    pub target_version: String,
    pub arrival: Arrival,
    pub seed: u64,
}

#[derive(Serialize)]
pub struct ConfigFile {
    pub config: RunConfig,
    pub provenance: Provenance,
}

#[derive(Serialize)]
pub struct Summary {
    pub mode: Mode,
    pub arrival: Arrival,
    pub seed: u64,
    pub git_sha: String,
    pub target_url: String,
    pub requested_rate_hz: f64,
    pub sent_rate_hz: f64,
    pub served_rate_hz: f64,
    pub output_token_rate_hz: f64,
    pub counters: DriverCounters,
    pub records: RecorderSummary,
    pub wakeup_floor_ns: u64,
    pub send_lag_budget_ns: u64,
    pub failure_reasons: Vec<String>,
}

#[derive(Serialize)]
pub struct RunOutcome {
    pub dir: PathBuf,
    pub summary: Summary,
    pub failed: bool,
}

/// A dispatch is late past max(10% of interval, 5× measured wakeup floor, 1ms).
pub fn lag_budget_ns(interval_ns: u64, wakeup_floor_ns: u64) -> u64 {
    ((interval_ns as f64 * 0.10) as u64)
        .max(wakeup_floor_ns.saturating_mul(WAKEUP_FLOOR_MULTIPLIER))
        .max(MIN_LAG_BUDGET_NS)
}

fn hostname() -> String {
    Command::new("hostname")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn create_run_dir(runs_root: &std::path::Path, utc: &str, name: &str) -> Result<PathBuf> {
    fs::create_dir_all(runs_root).with_context(|| format!("creating {}", runs_root.display()))?;
    let base = format!("{utc}_{name}");
    let mut candidate = runs_root.join(&base);
    let mut suffix = 1;
    while candidate.exists() {
        candidate = runs_root.join(format!("{base}-{suffix}"));
        suffix += 1;
    }
    fs::create_dir(&candidate).with_context(|| format!("creating {}", candidate.display()))?;
    Ok(candidate)
}

pub async fn run(cfg: RunConfig, runs_root: &std::path::Path) -> Result<RunOutcome> {
    let utc = RunClock::utc_stamp();
    let host = hostname();
    let version = target::probe_version(&cfg).await;
    let dir = create_run_dir(runs_root, &utc, &cfg.name)?;

    // Calibrate and complete every fallible setup step before the measurement anchor.
    let floor = wakeup_floor_ns().await;
    let budget = match cfg.mode {
        Mode::Open => lag_budget_ns((1e9 / cfg.rate).round().max(1.0) as u64, floor),
        Mode::Closed => 0,
    };
    let config_path = dir.join("config.json");
    let records_path = dir.join("records.csv");
    let events_path = dir.join("stream_events.csv");
    let (recorder, sink) = Recorder::new(
        &records_path,
        &events_path,
        cfg.warmup_ns(),
        cfg.duration_ns(),
        budget,
        cfg.channel_capacity,
    )?;
    let target = Target::new(&cfg)?;
    let resolved_seed = if cfg.seed != 0 {
        cfg.seed
    } else {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos() as u64)
            .unwrap_or(0)
    };
    let mut config_file = ConfigFile {
        config: cfg.clone(),
        provenance: Provenance {
            tool: TOOL_VERSION.into(),
            git_sha: GIT_SHA.into(),
            hostname: host,
            anchor_wall_unix_ns: None,
            run_started_utc: utc,
            target_url: cfg.target_url.clone(),
            target_version: version,
            arrival: cfg.arrival,
            seed: resolved_seed,
        },
    };
    write_json(&config_path, &config_file)?;

    let recorder_handle = tokio::spawn(recorder.run());
    let clock = Arc::new(RunClock::new());
    config_file.provenance.anchor_wall_unix_ns = Some(clock.wall_unix_ns());
    let counters = driver::run(&cfg, clock, sink, target, resolved_seed).await;
    let records = recorder_handle.await.context("recorder task panicked")?;
    write_json(&config_path, &config_file)?;

    let failure_reasons = failure_reasons(&cfg, &counters, &records);
    let window_s = cfg.duration_s - cfg.warmup_s;

    let summary = Summary {
        mode: cfg.mode,
        arrival: cfg.arrival,
        seed: resolved_seed,
        git_sha: GIT_SHA.into(),
        target_url: cfg.target_url.clone(),
        requested_rate_hz: cfg.rate,
        sent_rate_hz: counters.fired as f64 / cfg.duration_s,
        served_rate_hz: records.completed_in_window_ok as f64 / window_s,
        output_token_rate_hz: records.output_tokens_in_window as f64 / window_s,
        counters,
        records,
        wakeup_floor_ns: floor,
        send_lag_budget_ns: budget,
        failure_reasons: failure_reasons.clone(),
    };
    let failed = !failure_reasons.is_empty();

    let summary_path = dir.join("summary.json");
    write_json(&summary_path, &summary)?;

    Ok(RunOutcome {
        dir,
        summary,
        failed,
    })
}

fn write_json(path: &std::path::Path, value: &impl Serialize) -> Result<()> {
    fs::write(path, serde_json::to_string_pretty(value)?)
        .with_context(|| format!("writing {}", path.display()))
}

pub fn failure_reasons(
    cfg: &RunConfig,
    counters: &DriverCounters,
    records: &RecorderSummary,
) -> Vec<String> {
    let mut reasons = Vec::new();

    if counters.intended != counters.fired + counters.clipped {
        reasons.push(format!(
            "accounting identity failed: intended {} != fired {} + clipped {}",
            counters.intended, counters.fired, counters.clipped
        ));
    }
    if records.dropped > 0 {
        reasons.push(format!(
            "{} records dropped (channel full) — data lost",
            records.dropped
        ));
    }
    if counters.clipped > 0 {
        reasons.push(format!(
            "{} requests clipped by the in-flight cap — the offered load exceeded what the harness could keep in flight",
            counters.clipped
        ));
    }
    if cfg.mode == Mode::Open && records.late_dispatch_fraction > MAX_LATE_DISPATCH_FRACTION {
        reasons.push(format!(
            "{:.2}% of dispatches ({}/{}) exceeded the send-lag budget — the harness changed the offered load shape",
            records.late_dispatch_fraction * 100.0,
            records.late_dispatches,
            records.send_lag_ns.count
        ));
    }
    if records.in_window == 0 {
        reasons.push("no requests in the measured window".into());
    }
    if records.in_window_timeouts > 0 {
        reasons.push(format!(
            "{} request(s) timed out in the measured window",
            records.in_window_timeouts
        ));
    }
    if records.in_window_stream_errors > 0 {
        reasons.push(format!(
            "{} malformed or incomplete stream(s) in the measured window",
            records.in_window_stream_errors
        ));
    }
    let http_errors = records
        .in_window_errors
        .saturating_sub(records.in_window_timeouts)
        .saturating_sub(records.in_window_stream_errors);
    let error_rate = http_errors as f64 / records.in_window.max(1) as f64;
    if error_rate > MAX_ERROR_RATE {
        reasons.push(format!(
            "HTTP error rate {:.2}% exceeds {:.2}%",
            error_rate * 100.0,
            MAX_ERROR_RATE * 100.0
        ));
    }

    reasons
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::{Mode, RunConfig},
        recorder::Stats,
    };

    fn open_cfg(rate: f64) -> RunConfig {
        RunConfig {
            name: "t".into(),
            target_url: "http://localhost:1".into(),
            model: "m".into(),
            prompt: "p".into(),
            mode: Mode::Open,
            rate,
            arrival: Arrival::Uniform,
            seed: 0,
            duration_s: 5.0,
            warmup_s: 0.0,
            max_in_flight: 32,
            concurrency: 1,
            request_timeout_s: 30.0,
            channel_capacity: 1024,
            target_notes: None,
        }
    }

    fn clean_records() -> RecorderSummary {
        RecorderSummary {
            recorded: 50,
            dropped: 0,
            in_window: 50,
            in_window_ok: 50,
            in_window_errors: 0,
            in_window_timeouts: 0,
            in_window_stream_errors: 0,
            completed_in_window_ok: 50,
            output_tokens: 800,
            output_tokens_in_window: 800,
            late_dispatches: 0,
            late_dispatch_fraction: 0.0,
            warmup_rows: 0,
            by_status: Default::default(),
            successful_e2e_ns: Stats {
                count: 50,
                p50: 1_000_000,
                p95: 1_900_000,
                p99: 2_000_000,
                p999: 3_000_000,
                max: 4_000_000,
            },
            successful_ttft_ns: Stats::default(),
            successful_tpot_ns: Stats::default(),
            itl_ns: Stats::default(),
            send_lag_ns: Stats {
                count: 50,
                p50: 100_000,
                p95: 190_000,
                p99: 200_000,
                p999: 300_000,
                max: 400_000,
            },
        }
    }

    #[test]
    fn budget_is_the_max_of_all_three_floors() {
        assert_eq!(lag_budget_ns(1_000_000_000, 1_000), 100_000_000); // 10% of interval
        assert_eq!(lag_budget_ns(100_000, 500_000), 2_500_000); // 5× measured floor wins
        assert_eq!(lag_budget_ns(1_000, 1_000), 1_000_000); // 1ms constant floor wins
    }

    #[test]
    fn a_clean_run_has_no_reasons() {
        let cfg = open_cfg(10.0);
        let counters = DriverCounters {
            intended: 50,
            fired: 50,
            clipped: 0,
        };
        assert!(failure_reasons(&cfg, &counters, &clean_records()).is_empty());
    }

    #[test]
    fn accounting_identity_must_hold() {
        let cfg = open_cfg(10.0);
        let counters = DriverCounters {
            intended: 50,
            fired: 30,
            clipped: 10,
        };
        let reasons = failure_reasons(&cfg, &counters, &clean_records());
        assert!(
            reasons.iter().any(|r| r.contains("accounting")),
            "{reasons:?}"
        );
    }

    #[test]
    fn dropped_and_clipped_are_failures() {
        let cfg = open_cfg(10.0);
        let counters = DriverCounters {
            intended: 50,
            fired: 40,
            clipped: 10,
        };
        let mut records = clean_records();
        records.dropped = 3;
        let reasons = failure_reasons(&cfg, &counters, &records);
        assert!(reasons.iter().any(|r| r.contains("dropped")), "{reasons:?}");
        assert!(reasons.iter().any(|r| r.contains("clipped")), "{reasons:?}");
    }

    #[test]
    fn more_than_one_percent_late_dispatches_condemns_the_run() {
        let cfg = open_cfg(10.0);
        let counters = DriverCounters {
            intended: 50,
            fired: 50,
            clipped: 0,
        };
        let mut records = clean_records();
        records.send_lag_ns.count = 100;
        records.late_dispatches = 2;
        records.late_dispatch_fraction = 0.02;
        let reasons = failure_reasons(&cfg, &counters, &records);
        assert!(
            reasons.iter().any(|r| r.contains("dispatches")),
            "{reasons:?}"
        );
    }

    #[test]
    fn exactly_one_percent_late_dispatches_is_allowed() {
        let cfg = open_cfg(20.0);
        let counters = DriverCounters {
            intended: 200,
            fired: 200,
            clipped: 0,
        };
        let mut records = clean_records();
        records.send_lag_ns.count = 200;
        records.late_dispatches = 2;
        records.late_dispatch_fraction = 0.01;
        assert!(failure_reasons(&cfg, &counters, &records).is_empty());
    }

    #[test]
    fn exactly_one_percent_http_errors_is_allowed() {
        let cfg = open_cfg(20.0);
        let counters = DriverCounters {
            intended: 200,
            fired: 200,
            clipped: 0,
        };
        let mut records = clean_records();
        records.in_window = 200;
        records.in_window_ok = 198;
        records.in_window_errors = 2;
        assert!(failure_reasons(&cfg, &counters, &records).is_empty());
    }

    #[test]
    fn a_slow_run_is_not_condemned_by_the_constant_wakeup_delay() {
        let cfg = open_cfg(40.0); // interval 25ms, 10% = 2.5ms
        let counters = DriverCounters {
            intended: 120,
            fired: 120,
            clipped: 0,
        };
        let mut records = clean_records();
        records.send_lag_ns = Stats {
            count: 120,
            p50: 1_841_151,
            p95: 2_200_000,
            p99: 2_373_631,
            p999: 2_500_000,
            max: 2_500_000,
        };
        let budget = lag_budget_ns(25_000_000, 1_800_000);
        assert_eq!(
            budget, 9_000_000,
            "the measured floor must raise the budget"
        );
        assert!(failure_reasons(&cfg, &counters, &records).is_empty());
    }

    #[test]
    fn an_empty_window_is_a_failure() {
        let cfg = open_cfg(10.0);
        let counters = DriverCounters {
            intended: 50,
            fired: 50,
            clipped: 0,
        };
        let mut records = clean_records();
        records.in_window = 0;
        records.in_window_ok = 0;
        let reasons = failure_reasons(&cfg, &counters, &records);
        assert!(
            reasons.iter().any(|r| r.contains("no requests")),
            "{reasons:?}"
        );
    }

    #[test]
    fn error_rate_past_one_percent_fails() {
        let cfg = open_cfg(10.0);
        let counters = DriverCounters {
            intended: 50,
            fired: 50,
            clipped: 0,
        };
        let mut records = clean_records();
        records.in_window_ok = 47;
        records.in_window_errors = 3; // 3/50 = 6%
        let reasons = failure_reasons(&cfg, &counters, &records);
        assert!(
            reasons.iter().any(|r| r.contains("HTTP error rate")),
            "{reasons:?}"
        );
    }

    #[test]
    fn any_timeout_in_the_measured_window_fails() {
        let cfg = open_cfg(10.0);
        let counters = DriverCounters {
            intended: 50,
            fired: 50,
            clipped: 0,
        };
        let mut records = clean_records();
        records.in_window_ok = 49;
        records.in_window_errors = 1;
        records.in_window_timeouts = 1;
        let reasons = failure_reasons(&cfg, &counters, &records);
        assert!(
            reasons.iter().any(|r| r.contains("timed out")),
            "{reasons:?}"
        );
        assert!(
            !reasons.iter().any(|r| r.contains("HTTP error rate")),
            "{reasons:?}"
        );
    }

    #[test]
    fn any_stream_error_in_the_measured_window_fails() {
        let cfg = open_cfg(10.0);
        let counters = DriverCounters {
            intended: 50,
            fired: 50,
            clipped: 0,
        };
        let mut records = clean_records();
        records.in_window_ok = 49;
        records.in_window_errors = 1;
        records.in_window_stream_errors = 1;
        let reasons = failure_reasons(&cfg, &counters, &records);
        assert!(
            reasons
                .iter()
                .any(|reason| reason.contains("incomplete stream")),
            "{reasons:?}"
        );
        assert!(
            !reasons
                .iter()
                .any(|reason| reason.contains("HTTP error rate")),
            "{reasons:?}"
        );
    }

    #[test]
    fn closed_mode_is_exempt_from_the_lag_verdict() {
        let mut cfg = open_cfg(10.0);
        cfg.mode = Mode::Closed;
        cfg.rate = 0.0;
        let counters = DriverCounters {
            intended: 12,
            fired: 12,
            clipped: 0,
        };
        let mut records = clean_records();
        records.send_lag_ns.p50 = 50_000_000;
        records.late_dispatch_fraction = 1.0;
        assert!(failure_reasons(&cfg, &counters, &records).is_empty());
    }

    #[test]
    fn run_dirs_never_collide() {
        let root = std::env::temp_dir().join(format!("satlab-rundir-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let first = create_run_dir(&root, "20260913T000000Z", "demo").unwrap();
        let second = create_run_dir(&root, "20260913T000000Z", "demo").unwrap();
        assert_ne!(first, second);
        assert!(first.exists() && second.exists());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn provenance_is_null_until_the_exact_anchor_is_known() {
        let dir = std::env::temp_dir().join(format!("satlab-provenance-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        let mut file = ConfigFile {
            config: open_cfg(1.0),
            provenance: Provenance {
                tool: "test".into(),
                git_sha: "test".into(),
                hostname: "test".into(),
                anchor_wall_unix_ns: None,
                run_started_utc: "20260913T000000Z".into(),
                target_url: "http://localhost:1".into(),
                target_version: "test".into(),
                arrival: Arrival::Uniform,
                seed: 0,
            },
        };

        write_json(&path, &file).unwrap();
        let initial: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert!(initial["provenance"]["anchor_wall_unix_ns"].is_null());

        let exact_anchor = RunClock::new().wall_unix_ns();
        file.provenance.anchor_wall_unix_ns = Some(exact_anchor);
        write_json(&path, &file).unwrap();
        let completed: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(completed["provenance"]["anchor_wall_unix_ns"], exact_anchor);
        let _ = fs::remove_dir_all(&dir);
    }
}
