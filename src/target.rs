//! The HTTP target: one request shape, fixed and hardcoded, so measured service time describes the server rather than the prompt. `ignore_eos` is the field that buys this — without it the model stops when it likes and output length becomes a property of the prompt.

use crate::{clock::RunClock, config::RunConfig};
use anyhow::{Context, Result};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Ok,
    /// Reached the server; it answered badly or not at all.
    HttpError,
    /// No answer inside `request_timeout_s`.
    Timeout,
    /// HTTP succeeded, but the SSE stream was malformed or incomplete.
    StreamError,
    /// Never sent (cap full): recorded, not dropped — silently discarding clips is how a load generator lies about capacity.
    Clipped,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Status::Ok => "ok",
            Status::HttpError => "http_error",
            Status::Timeout => "timeout",
            Status::StreamError => "stream_error",
            Status::Clipped => "clipped",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct OutputEvent {
    pub emitted_ns: u64,
    pub token_count: u64,
}

#[derive(Debug, Clone)]
pub struct Outcome {
    pub status: Status,
    pub http_status: u16,
    pub bytes: u64,
    pub first_output_ns: u64,
    pub last_output_ns: u64,
    pub output_tokens: u64,
    pub output_events: Vec<OutputEvent>,
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
            "stream": true,
            "stream_options": { "include_usage": true },
            "return_token_ids": true,
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
    pub async fn send(&self, clock: &RunClock) -> Outcome {
        let mut resp = match self
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
                    first_output_ns: 0,
                    last_output_ns: 0,
                    output_tokens: 0,
                    output_events: Vec::new(),
                };
            }
        };

        let http_status = resp.status().as_u16();
        if !(200..300).contains(&http_status) {
            let bytes = resp
                .bytes()
                .await
                .map(|body| body.len() as u64)
                .unwrap_or(0);
            return Outcome {
                status: Status::HttpError,
                http_status,
                bytes,
                first_output_ns: 0,
                last_output_ns: 0,
                output_tokens: 0,
                output_events: Vec::new(),
            };
        }

        read_sse(&mut resp, clock, http_status).await
    }
}

async fn read_sse(resp: &mut reqwest::Response, clock: &RunClock, http_status: u16) -> Outcome {
    let mut pending = Vec::new();
    let mut bytes = 0u64;
    let mut events = Vec::new();
    let mut usage_tokens = None;
    let mut saw_done = false;

    loop {
        let chunk = match resp.chunk().await {
            Ok(Some(chunk)) => chunk,
            Ok(None) => break,
            Err(error) => {
                return stream_outcome(
                    if error.is_timeout() {
                        Status::Timeout
                    } else {
                        Status::StreamError
                    },
                    http_status,
                    bytes,
                    events,
                    usage_tokens.unwrap_or(0),
                );
            }
        };
        bytes += chunk.len() as u64;
        pending.extend_from_slice(&chunk);

        while let Some(newline) = pending.iter().position(|byte| *byte == b'\n') {
            let mut line = pending.drain(..=newline).collect::<Vec<_>>();
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            if line.is_empty() || line.starts_with(b":") {
                continue;
            }
            let Some(data) = line.strip_prefix(b"data:") else {
                continue;
            };
            let data = data.strip_prefix(b" ").unwrap_or(data);
            if data == b"[DONE]" {
                saw_done = true;
                continue;
            }
            let Ok(value) = serde_json::from_slice::<serde_json::Value>(data) else {
                return stream_outcome(
                    Status::StreamError,
                    http_status,
                    bytes,
                    events,
                    usage_tokens.unwrap_or(0),
                );
            };
            if let Some(tokens) = value
                .get("usage")
                .and_then(|usage| usage.get("completion_tokens"))
                .and_then(|tokens| tokens.as_u64())
            {
                usage_tokens = Some(tokens);
            }
            let token_count = value
                .get("choices")
                .and_then(|choices| choices.get(0))
                .and_then(|choice| choice.get("token_ids"))
                .and_then(|tokens| tokens.as_array())
                .map_or(0, |tokens| tokens.len() as u64);
            if token_count > 0 {
                events.push(OutputEvent {
                    emitted_ns: clock.now_ns(),
                    token_count,
                });
            }
        }
    }

    let event_tokens: u64 = events.iter().map(|event| event.token_count).sum();
    let status = if saw_done
        && !events.is_empty()
        && usage_tokens.is_some()
        && usage_tokens == Some(event_tokens)
    {
        Status::Ok
    } else {
        Status::StreamError
    };
    stream_outcome(
        status,
        http_status,
        bytes,
        events,
        usage_tokens.unwrap_or(event_tokens),
    )
}

fn stream_outcome(
    status: Status,
    http_status: u16,
    bytes: u64,
    output_events: Vec<OutputEvent>,
    output_tokens: u64,
) -> Outcome {
    Outcome {
        status,
        http_status,
        bytes,
        first_output_ns: output_events.first().map_or(0, |event| event.emitted_ns),
        last_output_ns: output_events.last().map_or(0, |event| event.emitted_ns),
        output_tokens,
        output_events,
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
