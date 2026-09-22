//! The HTTP target: one request shape, fixed and hardcoded, so measured service time describes the server rather than the prompt. `ignore_eos` is the field that buys this — without it the model stops when it likes and output length becomes a property of the prompt.

use crate::config::RunConfig;
use anyhow::{Context, Result};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Ok,
    /// Reached the server; it answered badly or not at all.
    HttpError,
    /// No answer inside `request_timeout_s`.
    Timeout,
    /// Never sent (cap full): recorded, not dropped — silently discarding clips is how a load generator lies about capacity.
    Clipped,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Status::Ok => "ok",
            Status::HttpError => "http_error",
            Status::Timeout => "timeout",
            Status::Clipped => "clipped",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Outcome {
    pub status: Status,
    pub http_status: u16,
    pub bytes: u64,
}

#[derive(Clone)]
pub struct Target {
    client: reqwest::Client,
    url: String,
    /// Serialized once at construction: re-encoding the same JSON per request is harness CPU spent on nothing.
    body: Vec<u8>,
}

impl Target {
    pub fn new(cfg: &RunConfig) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs_f64(cfg.request_timeout_s))
            // Sized to the in-flight cap so a legal request never waits on the pool.
            .pool_max_idle_per_host(cfg.max_in_flight)
            .build()
            .context("building HTTP client")?;

        let body = serde_json::to_vec(&serde_json::json!({
            "model": cfg.model,
            "prompt": cfg.prompt,
            "max_tokens": 16,
            "ignore_eos": true,
            "stream": false,
            "temperature": 0.0,
        }))
        .context("encoding request body")?;

        Ok(Self {
            client,
            url: cfg.completions_url(),
            body,
        })
    }

    /// One attempt. Never errors: a failed request is a data point, not a reason to abandon a run.
    pub async fn send(&self) -> Outcome {
        let resp = match self
            .client
            .post(&self.url)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(self.body.clone())
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                return Outcome {
                    status: if e.is_timeout() {
                        Status::Timeout
                    } else {
                        Status::HttpError
                    },
                    http_status: 0,
                    bytes: 0,
                };
            }
        };

        let http_status = resp.status().as_u16();
        // Read to the end even on bad statuses: an undrained response never returns its connection to the pool.
        let bytes = match resp.bytes().await {
            Ok(b) => b.len() as u64,
            Err(_) => 0,
        };

        Outcome {
            status: if (200..300).contains(&http_status) {
                Status::Ok
            } else {
                Status::HttpError
            },
            http_status,
            bytes,
        }
    }
}

/// Ask the target what it is, once, before the run. Best effort: an unreachable target yields "unknown" and the run proceeds — provenance is worth having, never worth failing a run over.
pub async fn probe_version(cfg: &RunConfig) -> String {
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
    {
        Ok(c) => c,
        Err(_) => return "unknown".into(),
    };

    match client.get(cfg.version_url()).send().await {
        Ok(resp) if resp.status().is_success() => resp
            .text()
            .await
            .map(|t| t.trim().to_string())
            .unwrap_or_else(|_| "unknown".into()),
        Ok(resp) => format!("unknown (HTTP {})", resp.status().as_u16()),
        Err(_) => "unknown".into(),
    }
}
