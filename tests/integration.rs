//! End-to-end tests. Every one of these runs a real mock server in a real
//! process over a real socket, because the things worth testing here — pacing,
//! the in-flight cap, connection reuse, classification of failure — do not
//! exist at the unit level.

use satlab::{
    clock::RunClock,
    config::{Mode, RunConfig},
    driver,
    recorder::{Recorder, RecorderSummary},
    target::{Status, Target, probe_version},
};
use std::{
    io::{BufRead, BufReader},
    process::{Child, Command, Stdio},
    sync::Arc,
    time::Duration,
};

/// The mock bin, started on an OS-assigned port. Killed on drop.
struct MockServer {
    child: Child,
    base_url: String,
}

impl MockServer {
    /// `capacity = n / d` requests per second, by construction.
    fn start(n: usize, d_secs: f64) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_mock"))
            .args([n.to_string(), d_secs.to_string(), "0".to_string()])
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn mock");

        // "mock: capacity = 1/0.3 = 3.3 req/s, listening on 127.0.0.1:52341"
        let mut line = String::new();
        let read = BufReader::new(child.stdout.take().expect("stdout is piped"))
            .read_line(&mut line)
            .is_ok();
        let addr = line
            .rsplit("listening on ")
            .next()
            .filter(|_| read && line.contains("listening on"))
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| {
                let _ = child.kill();
                panic!("mock did not announce an address; got {line:?}");
            });

        Self {
            child,
            base_url: format!("http://{addr}"),
        }
    }
}

impl Drop for MockServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn config_against(mock: &MockServer) -> RunConfig {
    RunConfig {
        name: "itest".into(),
        target_url: mock.base_url.clone(),
        model: "mock".into(),
        prompt: "hello".into(),
        mode: Mode::Open,
        rate: 10.0,
        duration_s: 1.0,
        warmup_s: 0.0,
        max_in_flight: 8,
        concurrency: 1,
        request_timeout_s: 5.0,
        channel_capacity: 1024,
        target_notes: None,
    }
}

#[derive(Debug)]
struct Row {
    _req_id: u64,
    intended: u64,
    actual: u64,
    completed: u64,
    status: String,
    _http_status: u16,
    bytes: u64,
    _warmup: u8,
}

fn parse_row(line: &str) -> Row {
    let f: Vec<&str> = line.split(',').collect();
    assert_eq!(f.len(), 8, "unexpected CSV row: {line}");
    Row {
        _req_id: f[0].parse().unwrap(),
        intended: f[1].parse().unwrap(),
        actual: f[2].parse().unwrap(),
        completed: f[3].parse().unwrap(),
        status: f[4].to_string(),
        _http_status: f[5].parse().unwrap(),
        bytes: f[6].parse().unwrap(),
        _warmup: f[7].parse().unwrap(),
    }
}

struct Completed {
    summary: RecorderSummary,
    counters: driver::DriverCounters,
    rows: Vec<Row>,
}

/// Runs the whole measurement path — driver, real sockets, recorder, CSV — and
/// hands back what a reader of the run directory would see.
async fn drive(cfg: &RunConfig, tag: &str) -> Completed {
    drive_after_setup_delay(cfg, tag, Duration::ZERO).await
}

async fn drive_after_setup_delay(cfg: &RunConfig, tag: &str, setup_delay: Duration) -> Completed {
    let target = Target::new(cfg).unwrap();

    let path = std::env::temp_dir().join(format!("satlab-it-{}-{tag}.csv", std::process::id()));
    let budget = satlab::runner::lag_budget_ns((1e9 / cfg.rate).round().max(1.0) as u64, 0);
    let (recorder, sink) = Recorder::new(
        &path,
        cfg.warmup_ns(),
        cfg.duration_ns(),
        budget,
        cfg.channel_capacity,
    )
    .unwrap();

    let writer = tokio::spawn(recorder.run());
    tokio::time::sleep(setup_delay).await;
    let clock = Arc::new(RunClock::new());
    let counters = driver::run(cfg, clock, sink, target).await;
    let summary = writer.await.unwrap();

    let text = std::fs::read_to_string(&path).unwrap();
    let _ = std::fs::remove_file(&path);

    Completed {
        summary,
        counters,
        rows: text.lines().skip(1).map(parse_row).collect(),
    }
}

#[tokio::test]
async fn successful_request_reports_bytes_and_status() {
    let mock = MockServer::start(4, 0.01);
    let target = Target::new(&config_against(&mock)).unwrap();

    let out = target.send().await;
    assert_eq!(out.status, Status::Ok, "got {out:?}");
    assert_eq!(out.http_status, 200);
    assert!(out.bytes > 0, "response body must be read, got 0 bytes");
}

#[tokio::test]
async fn non_2xx_is_http_error_not_ok() {
    let mock = MockServer::start(4, 0.01);
    let mut cfg = config_against(&mock);
    cfg.target_url.push_str("/no-such-prefix"); // /no-such-prefix/v1/completions -> 404
    let target = Target::new(&cfg).unwrap();

    let out = target.send().await;
    assert_eq!(out.status, Status::HttpError, "got {out:?}");
    assert_eq!(out.http_status, 404);
}

#[tokio::test]
async fn slow_response_past_the_timeout_is_classified_as_timeout() {
    let mock = MockServer::start(1, 2.0);
    let mut cfg = config_against(&mock);
    cfg.request_timeout_s = 0.3;
    let target = Target::new(&cfg).unwrap();

    let out = target.send().await;
    assert_eq!(out.status, Status::Timeout, "got {out:?}");
    assert_eq!(out.http_status, 0);
}

#[tokio::test]
async fn dead_target_is_an_error_but_never_aborts_the_run() {
    let mock = MockServer::start(1, 0.01);
    let cfg = config_against(&mock);
    let target = Target::new(&cfg).unwrap();

    assert_eq!(target.send().await.status, Status::Ok);
    drop(mock); // connection refused from here on

    for attempt in 1..=3 {
        let out = target.send().await;
        assert_eq!(out.status, Status::HttpError, "attempt {attempt}: {out:?}");
        assert_eq!(out.http_status, 0, "no HTTP status was received");
    }
}

#[tokio::test]
async fn version_probe_reads_the_body_or_gives_up_quietly() {
    let mock = MockServer::start(1, 0.01);
    let cfg = config_against(&mock);
    assert!(
        probe_version(&cfg).await.contains("mock"),
        "probe should return the target's own report"
    );

    let mut cfg = config_against(&mock);
    cfg.target_url = "http://127.0.0.1:1".into(); // nothing listens here
    assert_eq!(probe_version(&cfg).await, "unknown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn open_loop_keeps_time_when_the_target_can_serve_it() {
    let mock = MockServer::start(1, 0.1); // one slot, 100ms service
    let mut cfg = config_against(&mock);
    cfg.rate = 5.0; // one arrival per 200ms: comfortably inside capacity
    cfg.duration_s = 2.0;
    cfg.max_in_flight = 32;

    let done = drive(&cfg, "keep-time").await;

    assert_eq!(done.counters.intended, 10);
    assert_eq!(done.counters.fired, 10);
    assert_eq!(done.counters.clipped, 0, "10 arrivals, one 100ms slot");
    assert_eq!(done.rows.len(), 10, "no records lost");
    assert!(
        done.rows.iter().all(|r| r.status == "ok"),
        "{:?}",
        done.rows
    );
    assert!(
        done.rows.iter().all(|r| r.bytes > 0),
        "body read on every row"
    );

    // At a rate the target can serve, dispatch lag should remain comfortably
    // below the budget used by the late-dispatch fraction verdict.
    let budget_ns = (0.10 * 200_000_000.0f64) as u64; // 10% of a 200ms interval
    let p50_lag = done.summary.send_lag_ns.p50;
    assert!(
        p50_lag < budget_ns,
        "harness was {p50_lag}ns late at 5 req/s, over its {budget_ns}ns budget"
    );

    // The dominant term must be the server's own service time, not our noise.
    let p50 = done.summary.successful_latency_ns.p50;
    assert!(
        (80_000_000..200_000_000).contains(&p50),
        "p50 {p50}ns should sit near the mock's 100ms"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn served_rate_counts_only_successes_completed_before_the_deadline() {
    let mock = MockServer::start(1, 0.1); // one slot: ~10 req/s capacity
    let mut cfg = config_against(&mock);
    cfg.rate = 20.0;
    cfg.duration_s = 2.0;
    cfg.max_in_flight = 64; // retain the backlog instead of clipping it
    cfg.request_timeout_s = 10.0;

    let done = drive(&cfg, "completion-window").await;

    assert_eq!(done.counters.intended, 40);
    assert_eq!(done.counters.clipped, 0);
    assert_eq!(
        done.summary.in_window_ok, 40,
        "every offered request eventually succeeds"
    );
    assert_eq!(done.summary.successful_latency_ns.count, 40);
    assert!(
        done.summary.completed_in_window_ok < done.summary.in_window_ok,
        "responses drained after the deadline must not count as served in-window"
    );
    let served_rate = done.summary.completed_in_window_ok as f64 / cfg.duration_s;
    assert!(
        (8.0..=10.5).contains(&served_rate),
        "one 100ms slot should plateau near 10 req/s, got {served_rate:.1}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn setup_delay_before_clock_creation_is_not_send_lag() {
    let mock = MockServer::start(1, 0.01);
    let mut cfg = config_against(&mock);
    cfg.rate = 1.0;
    cfg.duration_s = 0.25;

    let setup_delay = Duration::from_millis(200);
    let done = drive_after_setup_delay(&cfg, "anchor-after-setup", setup_delay).await;

    assert_eq!(done.rows.len(), 1);
    assert!(
        done.rows[0].actual < setup_delay.as_nanos() as u64 / 2,
        "request-zero lag {}ns includes the artificial setup delay",
        done.rows[0].actual
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn open_loop_records_clipped_arrivals_instead_of_queueing_them() {
    let mock = MockServer::start(1, 0.1);
    let mut cfg = config_against(&mock);
    cfg.rate = 50.0; // an arrival every 20ms against 100ms of service
    cfg.duration_s = 2.0;
    cfg.max_in_flight = 2; // ...with room for two

    let done = drive(&cfg, "clipped").await;

    assert_eq!(done.counters.intended, 100);
    assert_eq!(
        done.counters.fired + done.counters.clipped,
        100,
        "every intended arrival is accounted for, sent or clipped"
    );
    assert!(
        done.counters.clipped > 0,
        "cap 2, service 100ms, arrivals every 20ms"
    );
    assert_eq!(done.summary.by_status["clipped"], done.counters.clipped);
    assert_eq!(
        done.rows.len() as u64,
        done.counters.fired + done.counters.clipped,
        "clipped arrivals are data, not gaps"
    );
    assert_eq!(done.summary.dropped, 0);
    assert_eq!(
        done.rows.iter().filter(|r| r.status == "clipped").count() as u64,
        done.counters.clipped
    );
    assert!(
        done.rows
            .iter()
            .filter(|r| r.status == "clipped")
            .all(|r| r.actual == 0 && r.completed == 0),
        "a clipped row was never sent, so it has no times"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn closed_loop_is_self_limiting_and_reports_no_lag() {
    let mock = MockServer::start(1, 0.05); // one slot: ~20 req/s ceiling
    let mut cfg = config_against(&mock);
    cfg.mode = Mode::Closed;
    cfg.concurrency = 4;
    cfg.duration_s = 1.0;

    let done = drive(&cfg, "closed").await;

    assert_eq!(
        done.counters.clipped, 0,
        "a closed loop cannot overrun itself"
    );
    assert!(
        done.rows.iter().all(|r| r.intended == r.actual),
        "closed loop has no separate intention"
    );
    assert_eq!(
        done.summary.send_lag_ns.p99, 0,
        "lag is zero by construction"
    );

    let achieved = done.counters.fired as f64 / cfg.duration_s;
    assert!(
        (10.0..25.0).contains(&achieved),
        "one 50ms slot serves ~20/s; got {achieved:.1}"
    );

    // The foil's whole point: throughput stays safe, and the queueing shows up
    // as latency instead. It cannot tell you what happens past capacity.
    let p50 = done.summary.successful_latency_ns.p50;
    assert!(
        p50 > 50_000_000,
        "four clients sharing one slot must queue; p50 was {p50}ns"
    );
}
