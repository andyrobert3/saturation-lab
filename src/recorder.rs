//! Recorder: single owner of records.csv and the histograms. Requests write via try_send, which never blocks a fire — a full channel drops rows, and drops are counted, because a load generator that waits for its own disk is measuring itself.

use crate::target::{OutputEvent, Status};
use anyhow::{Context, Result};
use hdrhistogram::Histogram;
use serde::Serialize;
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};
use tokio::sync::mpsc;

pub const CSV_HEADER: &str = "req_id,intended_ns,actual_ns,first_output_ns,last_output_ns,completed_ns,output_tokens,status,http_status,bytes,warmup";
pub const EVENTS_CSV_HEADER: &str = "req_id,event_index,emitted_ns,token_count,request_warmup";

const FLUSH_EVERY: usize = 1000;

#[derive(Debug, Clone)]
pub struct Record {
    pub req_id: u64,
    pub intended_ns: u64,
    pub actual_ns: u64,
    pub first_output_ns: u64,
    pub last_output_ns: u64,
    pub completed_ns: u64,
    pub output_tokens: u64,
    pub status: Status,
    pub http_status: u16,
    pub bytes: u64,
    pub output_events: Vec<OutputEvent>,
}

#[derive(Clone)]
pub struct RecordSink {
    tx: mpsc::Sender<Record>,
    dropped: Arc<AtomicU64>,
}

impl RecordSink {
    /// Fire-and-forget: on a full channel the row is dropped and counted, never waited on.
    pub fn send(&self, record: Record) {
        if self.tx.try_send(record).is_err() {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct Stats {
    pub count: u64,
    pub p50: u64,
    pub p95: u64,
    pub p99: u64,
    pub p999: u64,
    pub max: u64,
}

impl Stats {
    fn of(histogram: &Histogram<u64>) -> Self {
        if histogram.is_empty() {
            return Stats::default();
        }
        Stats {
            count: histogram.len(),
            p50: histogram.value_at_quantile(0.50),
            p95: histogram.value_at_quantile(0.95),
            p99: histogram.value_at_quantile(0.99),
            p999: histogram.value_at_quantile(0.999),
            max: histogram.max(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct RecorderSummary {
    pub recorded: u64,
    pub dropped: u64,
    pub in_window: u64,
    pub in_window_ok: u64,
    pub in_window_errors: u64,
    pub in_window_timeouts: u64,
    pub in_window_stream_errors: u64,
    pub completed_in_window_ok: u64,
    pub output_tokens: u64,
    pub output_tokens_in_window: u64,
    pub late_dispatches: u64,
    pub late_dispatch_fraction: f64,
    pub warmup_rows: u64,
    pub by_status: BTreeMap<String, u64>,
    pub successful_e2e_ns: Stats,
    pub successful_ttft_ns: Stats,
    pub successful_tpot_ns: Stats,
    pub itl_ns: Stats,
    pub send_lag_ns: Stats,
}

pub struct Recorder {
    rx: mpsc::Receiver<Record>,
    writer: BufWriter<File>,
    events_writer: BufWriter<File>,
    e2e: Histogram<u64>,
    ttft: Histogram<u64>,
    tpot: Histogram<u64>,
    itl: Histogram<u64>,
    send_lag: Histogram<u64>,
    warmup_ns: u64,
    duration_ns: u64,
    lag_budget_ns: u64,
    dropped: Arc<AtomicU64>,
    summary: RecorderSummary,
}

impl Recorder {
    pub fn new(
        path: &Path,
        events_path: &Path,
        warmup_ns: u64,
        duration_ns: u64,
        lag_budget_ns: u64,
        capacity: usize,
    ) -> Result<(Self, RecordSink)> {
        let mut writer = BufWriter::new(
            File::create(path).with_context(|| format!("creating {}", path.display()))?,
        );
        writeln!(writer, "{CSV_HEADER}")?;
        let mut events_writer = BufWriter::new(
            File::create(events_path)
                .with_context(|| format!("creating {}", events_path.display()))?,
        );
        writeln!(events_writer, "{EVENTS_CSV_HEADER}")?;

        let (tx, rx) = mpsc::channel(capacity);
        let dropped = Arc::new(AtomicU64::new(0));
        let recorder = Self {
            rx,
            writer,
            events_writer,
            e2e: Histogram::<u64>::new(3).context("building E2E histogram")?,
            ttft: Histogram::<u64>::new(3).context("building TTFT histogram")?,
            tpot: Histogram::<u64>::new(3).context("building TPOT histogram")?,
            itl: Histogram::<u64>::new(3).context("building ITL histogram")?,
            send_lag: Histogram::<u64>::new(3).context("building send-lag histogram")?,
            warmup_ns,
            duration_ns,
            lag_budget_ns,
            dropped: dropped.clone(),
            summary: RecorderSummary {
                recorded: 0,
                dropped: 0,
                in_window: 0,
                in_window_ok: 0,
                in_window_errors: 0,
                in_window_timeouts: 0,
                in_window_stream_errors: 0,
                completed_in_window_ok: 0,
                output_tokens: 0,
                output_tokens_in_window: 0,
                late_dispatches: 0,
                late_dispatch_fraction: 0.0,
                warmup_rows: 0,
                by_status: BTreeMap::new(),
                successful_e2e_ns: Stats::default(),
                successful_ttft_ns: Stats::default(),
                successful_tpot_ns: Stats::default(),
                itl_ns: Stats::default(),
                send_lag_ns: Stats::default(),
            },
        };
        Ok((recorder, RecordSink { tx, dropped }))
    }

    pub async fn run(mut self) -> RecorderSummary {
        let mut rows_since_flush = 0usize;

        while let Some(record) = self.rx.recv().await {
            self.write_row(&record);
            self.write_events(&record);
            self.observe(&record);
            rows_since_flush += 1;
            if rows_since_flush >= FLUSH_EVERY {
                self.writer.flush().ok();
                self.events_writer.flush().ok();
                rows_since_flush = 0;
            }
        }

        self.writer.flush().ok();
        self.events_writer.flush().ok();
        self.summary.successful_e2e_ns = Stats::of(&self.e2e);
        self.summary.successful_ttft_ns = Stats::of(&self.ttft);
        self.summary.successful_tpot_ns = Stats::of(&self.tpot);
        self.summary.itl_ns = Stats::of(&self.itl);
        self.summary.send_lag_ns = Stats::of(&self.send_lag);
        self.summary.late_dispatch_fraction =
            self.summary.late_dispatches as f64 / self.summary.send_lag_ns.count.max(1) as f64;
        self.summary.dropped = self.dropped.load(Ordering::Relaxed);
        self.summary
    }

    /// Derived, never measured: the harness owns intended_ns, so warmup is a comparison on data it already has.
    fn is_warmup(&self, record: &Record) -> bool {
        record.intended_ns < self.warmup_ns
    }

    fn write_row(&mut self, record: &Record) {
        let _ = writeln!(
            self.writer,
            "{},{},{},{},{},{},{},{},{},{},{}",
            record.req_id,
            record.intended_ns,
            record.actual_ns,
            record.first_output_ns,
            record.last_output_ns,
            record.completed_ns,
            record.output_tokens,
            record.status.as_str(),
            record.http_status,
            record.bytes,
            self.is_warmup(record) as u8
        );
    }

    fn write_events(&mut self, record: &Record) {
        for (index, event) in record.output_events.iter().enumerate() {
            let _ = writeln!(
                self.events_writer,
                "{},{},{},{},{}",
                record.req_id,
                index,
                event.emitted_ns,
                event.token_count,
                self.is_warmup(record) as u8
            );
        }
    }

    fn observe(&mut self, record: &Record) {
        self.summary.recorded += 1;
        *self
            .summary
            .by_status
            .entry(record.status.as_str().to_string())
            .or_insert(0) += 1;
        if record.status == Status::Ok {
            self.summary.output_tokens_in_window += record
                .output_events
                .iter()
                .filter(|event| {
                    event.emitted_ns >= self.warmup_ns && event.emitted_ns < self.duration_ns
                })
                .map(|event| event.token_count)
                .sum::<u64>();
        }
        if record.status == Status::Ok
            && record.last_output_ns >= self.warmup_ns
            && record.last_output_ns < self.duration_ns
        {
            self.summary.completed_in_window_ok += 1;
        }
        if self.is_warmup(record) {
            self.summary.warmup_rows += 1;
            return;
        }
        self.summary.in_window += 1;
        match record.status {
            Status::Ok => self.summary.in_window_ok += 1,
            Status::HttpError => self.summary.in_window_errors += 1,
            Status::Timeout => {
                self.summary.in_window_errors += 1;
                self.summary.in_window_timeouts += 1;
            }
            Status::StreamError => {
                self.summary.in_window_errors += 1;
                self.summary.in_window_stream_errors += 1;
            }
            Status::Clipped => {}
        }
        if record.status == Status::Ok {
            self.summary.output_tokens += record.output_tokens;
            let _ = self
                .e2e
                .record(record.last_output_ns.saturating_sub(record.actual_ns));
            let _ = self
                .ttft
                .record(record.first_output_ns.saturating_sub(record.actual_ns));
            if record.output_tokens > 1 {
                let _ = self.tpot.record(
                    record.last_output_ns.saturating_sub(record.first_output_ns)
                        / (record.output_tokens - 1),
                );
            }
            for pair in record.output_events.windows(2) {
                let _ = self
                    .itl
                    .record(pair[1].emitted_ns.saturating_sub(pair[0].emitted_ns));
            }
        }
        if record.status != Status::Clipped {
            let lag = record.actual_ns.saturating_sub(record.intended_ns);
            let _ = self.send_lag.record(lag);
            if lag > self.lag_budget_ns {
                self.summary.late_dispatches += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};

    fn record(id: u64, intended_ns: u64) -> Record {
        Record {
            req_id: id,
            intended_ns,
            actual_ns: intended_ns + 5,
            first_output_ns: intended_ns + 10,
            last_output_ns: intended_ns + 50,
            completed_ns: intended_ns + 55,
            output_tokens: 2,
            status: Status::Ok,
            http_status: 200,
            bytes: 17,
            output_events: vec![
                OutputEvent {
                    emitted_ns: intended_ns + 10,
                    token_count: 1,
                },
                OutputEvent {
                    emitted_ns: intended_ns + 50,
                    token_count: 1,
                },
            ],
        }
    }

    fn new_recorder(
        path: &Path,
        warmup_ns: u64,
        duration_ns: u64,
        lag_budget_ns: u64,
        capacity: usize,
    ) -> Result<(Recorder, RecordSink)> {
        Recorder::new(
            path,
            &path.with_file_name("stream_events.csv"),
            warmup_ns,
            duration_ns,
            lag_budget_ns,
            capacity,
        )
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("satlab-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[tokio::test]
    async fn every_row_lands_in_the_csv() {
        let dir = temp_dir("rec");
        let path = dir.join("records.csv");
        let (recorder, sink) = new_recorder(&path, 0, u64::MAX, 10, 1024).unwrap();
        let handle = tokio::spawn(recorder.run());
        for id in 0..1000 {
            sink.send(record(id, id * 1000));
        }
        drop(sink);
        let summary = handle.await.unwrap();

        let lines: Vec<String> = BufReader::new(std::fs::File::open(&path).unwrap())
            .lines()
            .map(|l| l.unwrap())
            .collect();
        assert_eq!(lines.len(), 1001, "header + 1000 rows");
        assert_eq!(summary.recorded, 1000);
        assert_eq!(summary.dropped, 0);
    }

    #[tokio::test]
    async fn full_channel_drops_are_counted() {
        let dir = temp_dir("drop");
        let path = dir.join("records.csv");
        let (recorder, sink) = new_recorder(&path, 0, u64::MAX, 10, 1).unwrap();
        // recorder never runs until the end: capacity 1, so all but the first send drop
        for id in 0..100 {
            sink.send(record(id, id * 1000));
        }
        drop(sink);
        let summary = tokio::spawn(recorder.run()).await.unwrap();
        assert_eq!(summary.recorded, 1);
        assert_eq!(summary.dropped, 99);
    }

    #[tokio::test]
    async fn warmup_rows_are_flagged_but_not_in_stats() {
        let dir = temp_dir("warmup");
        let path = dir.join("records.csv");
        let (recorder, sink) = new_recorder(&path, 50_000, u64::MAX, 10, 1024).unwrap();
        sink.send(record(0, 0)); // warmup: intended < warmup_ns
        sink.send(record(1, 100_000)); // in window
        drop(sink);
        let summary = tokio::spawn(recorder.run()).await.unwrap();
        assert_eq!(summary.warmup_rows, 1);
        assert_eq!(summary.in_window, 1);
        assert_eq!(summary.successful_e2e_ns.count, 1);
        assert_eq!(summary.send_lag_ns.count, 1);
    }

    #[tokio::test]
    async fn clipped_rows_are_recorded_but_never_lag() {
        let dir = temp_dir("clip");
        let path = dir.join("records.csv");
        let (recorder, sink) = new_recorder(&path, 0, u64::MAX, 10, 1024).unwrap();
        sink.send(Record {
            intended_ns: 7000,
            actual_ns: 0,
            first_output_ns: 0,
            last_output_ns: 0,
            completed_ns: 0,
            output_tokens: 0,
            status: Status::Clipped,
            http_status: 0,
            bytes: 0,
            output_events: Vec::new(),
            req_id: 7,
        });
        drop(sink);
        let summary = tokio::spawn(recorder.run()).await.unwrap();
        assert_eq!(summary.by_status["clipped"], 1);
        assert_eq!(summary.in_window_errors, 0);
        assert_eq!(
            summary.send_lag_ns.count, 0,
            "clipped rows carry no lag sample"
        );
    }

    #[tokio::test]
    async fn completion_throughput_uses_completion_time_not_arrival_cohort() {
        let dir = temp_dir("completion-window");
        let path = dir.join("records.csv");
        let (recorder, sink) = new_recorder(&path, 100, 1_000, 100, 16).unwrap();
        sink.send(Record {
            intended_ns: 50,
            last_output_ns: 200,
            ..record(0, 50)
        });
        sink.send(Record {
            intended_ns: 900,
            last_output_ns: 1_100,
            ..record(1, 900)
        });
        drop(sink);
        let summary = tokio::spawn(recorder.run()).await.unwrap();
        assert_eq!(
            summary.in_window_ok, 1,
            "latency cohort follows intended time"
        );
        assert_eq!(
            summary.completed_in_window_ok, 1,
            "throughput follows completion time"
        );
    }

    #[tokio::test]
    async fn late_fraction_excludes_warmup_and_clipped_rows() {
        let dir = temp_dir("late");
        let path = dir.join("records.csv");
        let (recorder, sink) = new_recorder(&path, 100, 1_000, 10, 16).unwrap();
        sink.send(Record {
            actual_ns: 50,
            ..record(0, 0)
        });
        sink.send(Record {
            actual_ns: 105,
            ..record(1, 100)
        });
        sink.send(Record {
            actual_ns: 120,
            ..record(2, 100)
        });
        sink.send(Record {
            intended_ns: 100,
            actual_ns: 0,
            first_output_ns: 0,
            last_output_ns: 0,
            completed_ns: 0,
            output_tokens: 0,
            status: Status::Clipped,
            http_status: 0,
            bytes: 0,
            output_events: Vec::new(),
            req_id: 3,
        });
        drop(sink);
        let summary = tokio::spawn(recorder.run()).await.unwrap();
        assert_eq!(summary.late_dispatches, 1);
        assert_eq!(summary.send_lag_ns.count, 2);
        assert_eq!(summary.late_dispatch_fraction, 0.5);
    }

    #[tokio::test]
    async fn timeout_is_separate_from_successful_e2e_latency() {
        let dir = temp_dir("timeout");
        let path = dir.join("records.csv");
        let (recorder, sink) = new_recorder(&path, 100, 1_000, 10, 16).unwrap();
        sink.send(Record {
            status: Status::Timeout,
            ..record(0, 0)
        });
        sink.send(Record {
            status: Status::Timeout,
            ..record(1, 100)
        });
        drop(sink);
        let summary = tokio::spawn(recorder.run()).await.unwrap();
        assert_eq!(summary.warmup_rows, 1);
        assert_eq!(summary.in_window_timeouts, 1);
        assert_eq!(summary.successful_e2e_ns.count, 0);
    }

    #[tokio::test]
    async fn streaming_metrics_and_token_window_follow_output_events() {
        let dir = temp_dir("streaming-metrics");
        let path = dir.join("records.csv");
        let (recorder, sink) = new_recorder(&path, 100, 1_000, 10, 16).unwrap();
        sink.send(Record {
            req_id: 1,
            intended_ns: 100,
            actual_ns: 105,
            first_output_ns: 200,
            last_output_ns: 800,
            completed_ns: 810,
            output_tokens: 4,
            status: Status::Ok,
            http_status: 200,
            bytes: 100,
            output_events: vec![
                OutputEvent {
                    emitted_ns: 200,
                    token_count: 1,
                },
                OutputEvent {
                    emitted_ns: 400,
                    token_count: 2,
                },
                OutputEvent {
                    emitted_ns: 800,
                    token_count: 1,
                },
            ],
        });
        drop(sink);
        let summary = tokio::spawn(recorder.run()).await.unwrap();
        assert_eq!(summary.successful_ttft_ns.p50, 95);
        assert_eq!(summary.successful_e2e_ns.p50, 695);
        assert_eq!(summary.successful_tpot_ns.p50, 200);
        assert_eq!(
            summary.itl_ns.count, 2,
            "ITL is per streamed output, not per token"
        );
        assert_eq!(summary.output_tokens_in_window, 4);
    }
}
