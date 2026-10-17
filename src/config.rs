//! Run configuration: what to offer, to whom, for how long. Validation collects every problem before failing, so a config typo costs one edit, not one edit per run.

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// Arrivals on a fixed schedule, independent of responses. The point of this tool.
    Open,
    /// `concurrency` workers, each `send -> await -> send`. Self-limiting: a labelled foil, not a substitute.
    Closed,
}

fn default_channel_capacity() -> usize {
    65536
}

fn default_concurrency() -> usize {
    1
}

/// How arrivals are spaced in open mode. `uniform` keeps today's deterministic
/// ladder; `poisson` draws exponential inter-arrivals so the load generator is
/// comparable to vLLM/SGLang benchmarks and can show burst absorption.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Arrival {
    #[default]
    Uniform,
    Poisson,
    /// Gamma inter-arrivals with this coefficient of variation: `{"gamma": {"cv": 2.0}}`. cv 1 is Poisson; larger is burstier.
    Gamma { cv: f64 },
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RunConfig {
    /// Becomes part of the run directory name.
    pub name: String,
    /// Base URL, e.g. `http://localhost:8000`. No trailing `/v1`.
    pub target_url: String,
    pub model: String,
    pub prompt: String,
    pub mode: Mode,

    /// Requests per second. Open mode only; in closed mode the arrival rate is an output, not an input.
    #[serde(default)]
    pub rate: f64,

    /// Arrival spacing in open mode; Uniform = current behavior.
    #[serde(default)]
    pub arrival: Arrival,

    /// Poisson RNG seed. 0 = drawn from the OS clock and recorded in provenance; any other value makes the arrival sequence reproducible.
    #[serde(default)]
    pub seed: u64,

    pub duration_s: f64,
    /// Prefix of the run excluded from statistics (the *server* reaching steady state).
    #[serde(default)]
    pub warmup_s: f64,

    /// In-flight cap in open mode; also sizes the connection pool. Full cap records `clipped` rather than queueing — queueing here would convert the generator into a closed loop.
    pub max_in_flight: usize,

    #[serde(default = "default_concurrency")]
    pub concurrency: usize,

    pub request_timeout_s: f64,

    /// Record channel capacity. Too small and a fast run drops records it could have kept; too large only costs memory.
    #[serde(default = "default_channel_capacity")]
    pub channel_capacity: usize,

    /// Free text written verbatim into `config.json` — launch flags, prefix-caching setting. Un-audited provenance is how benchmarks lie.
    #[serde(default)]
    pub target_notes: Option<String>,
}

impl RunConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("cannot read config {}: {e}", path.display()))?;
        let cfg: Self = serde_json::from_str(&text)
            .map_err(|e| anyhow::anyhow!("cannot parse config {}: {e}", path.display()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Every problem at once, or silence.
    pub fn validate(&self) -> Result<()> {
        let mut bad: Vec<String> = Vec::new();

        if self.name.is_empty() {
            bad.push("name: must not be empty (it becomes part of the run directory)".into());
        } else if !self
            .name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
        {
            bad.push(format!(
                "name: {:?} must be ASCII letters, digits, '.', '_' or '-' only (it becomes part of the run directory)",
                self.name
            ));
        }
        if !(self.target_url.starts_with("http://") || self.target_url.starts_with("https://")) {
            bad.push(format!(
                "target_url: {:?} must start with http:// or https://",
                self.target_url
            ));
        }
        if let Arrival::Gamma { cv } = self.arrival
            && !(cv.is_finite() && cv > 0.0)
        {
            bad.push(format!("arrival.gamma.cv: {cv} must be a positive number"));
        }
        if self.model.is_empty() {
            bad.push("model: must not be empty".into());
        }
        if self.prompt.is_empty() {
            bad.push("prompt: must not be empty".into());
        }
        if !self.duration_s.is_finite() || self.duration_s <= 0.0 {
            bad.push(format!("duration_s: {} must be > 0", self.duration_s));
        }
        if !self.warmup_s.is_finite() || self.warmup_s < 0.0 {
            bad.push(format!("warmup_s: {} must be >= 0", self.warmup_s));
        }
        // Strictly <, not <=: a run that is entirely warmup has nothing left to measure.
        if self.warmup_s >= self.duration_s {
            bad.push(format!(
                "warmup_s: {} must be < duration_s {} (otherwise the whole run is warmup and there is nothing left to measure)",
                self.warmup_s, self.duration_s
            ));
        }
        if !self.request_timeout_s.is_finite() || self.request_timeout_s <= 0.0 {
            bad.push(format!(
                "request_timeout_s: {} must be > 0 (an unbounded request hides a hung target)",
                self.request_timeout_s
            ));
        }
        if self.channel_capacity == 0 {
            bad.push("channel_capacity: must be > 0".into());
        }

        match self.mode {
            Mode::Open => {
                if !self.rate.is_finite() || self.rate <= 0.0 {
                    bad.push(format!("rate: {} must be > 0 in open mode", self.rate));
                } else if (1e9 / self.rate).round() < 1.0 {
                    bad.push(format!(
                        "rate: {} is above 1e9 req/s, which gives a sub-nanosecond interval",
                        self.rate
                    ));
                }
                if self.max_in_flight == 0 {
                    bad.push("max_in_flight: must be > 0 in open mode".into());
                }
            }
            Mode::Closed => {
                if self.concurrency == 0 {
                    bad.push("concurrency: must be > 0 in closed mode".into());
                }
            }
        }

        if bad.is_empty() {
            Ok(())
        } else {
            bail!("invalid config:\n  - {}", bad.join("\n  - "))
        }
    }

    pub fn duration_ns(&self) -> u64 {
        (self.duration_s * 1e9) as u64
    }

    pub fn warmup_ns(&self) -> u64 {
        (self.warmup_s * 1e9) as u64
    }

    /// URL for the provenance probe.
    pub fn version_url(&self) -> String {
        format!("{}/version", self.target_url.trim_end_matches('/'))
    }

    pub fn completions_url(&self) -> String {
        format!("{}/v1/completions", self.target_url.trim_end_matches('/'))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> RunConfig {
        RunConfig {
            name: "t".into(),
            target_url: "http://localhost:8000".into(),
            model: "m".into(),
            prompt: "p".into(),
            mode: Mode::Open,
            rate: 10.0,
            arrival: Arrival::Uniform,
            seed: 0,
            duration_s: 5.0,
            warmup_s: 1.0,
            max_in_flight: 32,
            concurrency: 1,
            request_timeout_s: 30.0,
            channel_capacity: 1024,
            target_notes: None,
        }
    }

    #[test]
    fn base_config_is_valid() {
        assert!(base().validate().is_ok());
    }

    #[test]
    fn rejects_zero_rate_in_open_mode() {
        let mut c = base();
        c.rate = 0.0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_zero_duration() {
        let mut c = base();
        c.duration_s = 0.0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_warmup_not_shorter_than_duration() {
        let mut c = base();
        c.warmup_s = 5.0;
        assert!(c.validate().is_err());
        c.warmup_s = 6.0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn closed_mode_needs_concurrency_not_rate() {
        let mut c = base();
        c.mode = Mode::Closed;
        c.rate = 0.0;
        assert!(c.validate().is_ok(), "closed mode does not use rate");
        c.concurrency = 0;
        assert!(c.validate().is_err());
        c.concurrency = 8;
        assert!(c.validate().is_ok());
    }

    #[test]
    fn rejects_path_traversal_in_name() {
        let mut c = base();
        c.name = "../etc".into();
        assert!(c.validate().is_err());
        c.name = "ok-name_1.2".into();
        assert!(c.validate().is_ok());
    }

    #[test]
    fn rejects_zero_channel_capacity_and_zero_timeout() {
        let mut c = base();
        c.channel_capacity = 0;
        assert!(c.validate().is_err());
        let mut c = base();
        c.request_timeout_s = 0.0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn reports_every_problem_at_once() {
        let mut c = base();
        c.rate = 0.0;
        c.duration_s = 0.0;
        c.channel_capacity = 0;
        let msg = c.validate().unwrap_err().to_string();
        assert!(msg.contains("rate"), "{msg}");
        assert!(msg.contains("duration_s"), "{msg}");
        assert!(msg.contains("channel_capacity"), "{msg}");
    }

    #[test]
    fn urls_are_joined_without_doubling_slashes() {
        let mut c = base();
        c.target_url = "http://localhost:8000/".into();
        assert_eq!(c.completions_url(), "http://localhost:8000/v1/completions");
        assert_eq!(c.version_url(), "http://localhost:8000/version");
    }

    #[test]
    fn json_without_optional_fields_still_loads() {
        let text = r#"{
            "name": "t", "target_url": "http://x", "model": "m", "prompt": "p",
            "mode": "open", "rate": 5.0, "duration_s": 2.0, "max_in_flight": 8,
            "request_timeout_s": 10.0
        }"#;
        let c: RunConfig = serde_json::from_str(text).unwrap();
        assert!(c.validate().is_ok());
        assert_eq!(c.channel_capacity, 65536, "default applied");
        assert_eq!(c.warmup_s, 0.0);
        assert!(c.target_notes.is_none());
    }
}
