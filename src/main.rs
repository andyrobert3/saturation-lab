//! CLI: load config, apply overrides, run, print verdict, set exit code.

use clap::Parser;
use satlab::{config, runner};

#[derive(Parser)]
#[command(version, about = "open-loop HTTP load generator")]
struct Cli {
    /// Path to config.json
    #[arg(short, long, default_value = "config.json")]
    config: String,

    /// Override rate (open mode)
    #[arg(long)]
    rate: Option<f64>,

    /// Override duration_s
    #[arg(long)]
    duration: Option<f64>,

    /// Where run directories are written
    #[arg(long, default_value = "runs")]
    runs_dir: String,
}

fn ms(ns: u64) -> f64 {
    ns as f64 / 1e6
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let cli = Cli::parse();
    let path = std::path::Path::new(&cli.config);
    let mut cfg = match config::RunConfig::load(path) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("error: {e:#}");
            std::process::exit(2);
        }
    };

    if let Some(rate) = cli.rate {
        cfg.rate = rate;
    }
    if let Some(duration) = cli.duration {
        cfg.duration_s = duration;
    }
    // Overrides can invalidate what the file validated.
    if let Err(e) = cfg.validate() {
        eprintln!("error: {e:#}");
        std::process::exit(2);
    }
    let outcome = runner::run(cfg, std::path::Path::new(&cli.runs_dir)).await;
    let outcome = match outcome {
        Ok(outcome) => outcome,
        Err(e) => {
            eprintln!("error: {e:#}");
            std::process::exit(2);
        }
    };

    report(&outcome);
    if outcome.failed {
        std::process::exit(1);
    }
}

fn report(outcome: &runner::RunOutcome) {
    let s = &outcome.summary;
    let statuses: Vec<String> = s
        .records
        .by_status
        .iter()
        .map(|(status, n)| format!("{status}={n}"))
        .collect();
    println!("run: {} ({})", outcome.dir.display(), s.mode);
    println!("  target:      {}  (git {})", s.target_url, s.git_sha);
    println!("  requested:   {:.1} req/s", s.requested_rate_hz);
    println!(
        "  sent:        {:.1} req/s (fired {})",
        s.sent_rate_hz, s.counters.fired
    );
    println!(
        "  served:      {:.1} req/s (completed ok in window {})",
        s.served_rate_hz, s.records.completed_in_window_ok
    );
    println!(
        "  records:     {} ({} warmup, {} in window, {} completed ok in window)  clipped {}  dropped {}  late dispatches {:.2}%",
        s.records.recorded,
        s.records.warmup_rows,
        s.records.in_window,
        s.records.completed_in_window_ok,
        s.counters.clipped,
        s.records.dropped,
        s.records.late_dispatch_fraction * 100.0
    );
    println!("  statuses:    {}", statuses.join(", "));
    let l = s.records.successful_latency_ns;
    println!(
        "  successful latency: p50 {:.1}ms  p99 {:.1}ms  p999 {:.1}ms  max {:.1}ms",
        ms(l.p50),
        ms(l.p99),
        ms(l.p999),
        ms(l.max)
    );
    let g = s.records.send_lag_ns;
    println!(
        "  send lag:    p50 {:.3}ms  p99 {:.3}ms  p999 {:.3}ms  max {:.3}ms  (floor {:.3}ms, budget {:.3}ms)",
        ms(g.p50),
        ms(g.p99),
        ms(g.p999),
        ms(g.max),
        ms(s.wakeup_floor_ns),
        ms(s.send_lag_budget_ns)
    );
    if s.failure_reasons.is_empty() {
        println!("PASS");
    } else {
        println!("FAILED:");
        for reason in &s.failure_reasons {
            println!("  - {reason}");
        }
    }
}
